"""
precompute_paraphrase_embeddings.py — 準備スクリプト for paraphrase_robustness.py

「実プロンプトへの依存度合い」（言語表現の丸暗記か、意味の理解か）を検定するには、訓練時の
T5埋め込みキャッシュに存在しない、実プロンプトと**意味的に同一だが表現が異なる**paraphrase
文の埋め込みが必要になる。get_action()にキャッシュ外の文字列を渡すと on-the-fly T5-11B
エンコードが走り、既にロード済みの2B policy DiTと同一GPUメモリ上でCUDA OOMする（本レポート
群のバグリスト、および`attractor/precompute_text_directions.py`のdocstring参照）。

したがって本スクリプトは`precompute_text_directions.py`と同じ設計方針を踏襲する：**policyモデル
を一切ロードせず**、T5-11bエンコーダのみをbf16でロードしてparaphrase文をエンコードし、結果を
ディスクに保存する。paraphrase_robustness.py側はこの保存済み埋め込みをキャッシュへ直接注入する
（`preload_dummy_prompt_embedding`と同一パターン）ため、評価実行中にT5を一切ロードしない。

RoboCasaの`PnPCounterToCab`タスクの実プロンプトは、対象物体`obj`に対し常に固定テンプレート
`"pick the {obj} from the counter and place it in the cabinet"`に従う（`robocasa/environments/
kitchen/single_stage/kitchen_pnp.py`の`PnPCounterToCab.get_ep_meta()`で確認済み、事前計算済み
T5キャッシュにも同テンプレートの文字列が多数含まれる）。エピソードごとに実際に出現する物体`obj`
は環境リセット時に決まる（robosuiteの決定論的乱数、`seed`で再現可能）ため、まず対象seed範囲で
env-onlyのリセットを行い（policy不要、軽量）実プロンプトを取得・物体名を抽出し、その物体を
埋め込んだparaphraseテンプレートを構成してからT5でエンコードする。paraphraseテンプレートは
動詞・前置詞を実プロンプトから変えつつ意味を保存する: "take the {obj} off the counter and put
it into the cabinet"。
"""

import argparse
import re
from pathlib import Path

import torch
from transformers import T5EncoderModel, T5TokenizerFast

from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import (
    PolicyEvalConfig,
    create_robocasa_env,
)

MODEL_NAME = "google-t5/t5-11b"
MAX_LENGTH = 512

REAL_PREFIX = "pick the "
REAL_SUFFIX = " from the counter and place it in the cabinet"
PARAPHRASE_TEMPLATE = "take the {obj} off the counter and put it into the cabinet"

REPO_ROOT = Path(__file__).resolve()
while REPO_ROOT.name != "cosmos-policy":
    REPO_ROOT = REPO_ROOT.parent

DEFAULT_OUT_PATH = (
    REPO_ROOT / "cosmos_policy/experiments/robot/robocasa/analysis/results/"
    "intervention_v3_paraphrase_robustness/paraphrase_embeddings_PnPCounterToCab.pt"
)


def extract_object(real_prompt: str) -> str:
    m = re.fullmatch(re.escape(REAL_PREFIX) + r"(.+)" + re.escape(REAL_SUFFIX), real_prompt)
    if m is None:
        raise ValueError(
            f"real prompt {real_prompt!r} does not match the expected PnPCounterToCab template "
            f"{REAL_PREFIX!r} + <obj> + {REAL_SUFFIX!r}"
        )
    return m.group(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--config_file", default="cosmos_policy/config/config.py")
    p.add_argument("--dataset_stats_path", required=True)
    p.add_argument("--t5_text_embeddings_path", required=True)
    p.add_argument("--task_name", default="PnPCounterToCab")
    p.add_argument("--seed", type=int, default=195)
    p.add_argument("--n_episodes", type=int, default=8)
    p.add_argument("--out_path", default=str(DEFAULT_OUT_PATH))
    args = p.parse_args()

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = PolicyEvalConfig(
        config=args.config, ckpt_path=args.ckpt_path, config_file=args.config_file,
        dataset_stats_path=args.dataset_stats_path, t5_text_embeddings_path=args.t5_text_embeddings_path,
        task_name=args.task_name, seed=args.seed,
    )

    episodes = []
    for ep in range(args.n_episodes):
        seed = args.seed + ep
        env, _ = create_robocasa_env(cfg, seed=seed, episode_idx=ep)
        env.reset()
        real_prompt = env.get_ep_meta().get("lang", args.task_name)
        env.close()
        obj = extract_object(real_prompt)
        paraphrase_prompt = PARAPHRASE_TEMPLATE.format(obj=obj)
        episodes.append({
            "episode": ep, "seed": seed, "obj": obj,
            "real_prompt": real_prompt, "paraphrase_prompt": paraphrase_prompt,
        })
        print(f"[ep {ep}] seed={seed} obj={obj!r}\n"
              f"    real:       {real_prompt!r}\n"
              f"    paraphrase: {paraphrase_prompt!r}")

    unique_paraphrases = sorted({e["paraphrase_prompt"] for e in episodes})
    print(f"\nLoading T5 tokenizer/encoder ({MODEL_NAME}, bf16)...")
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
    encoder = T5EncoderModel.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to("cuda").eval()
    print(f"Loaded. Encoding {len(unique_paraphrases)} unique paraphrase string(s)...")

    paraphrase_embeddings = {}
    with torch.inference_mode():
        for phrase in unique_paraphrases:
            enc = tokenizer.batch_encode_plus(
                [phrase], return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LENGTH,
            )
            input_ids = enc.input_ids.cuda()
            attn_mask = enc.attention_mask.cuda()
            hidden = encoder(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state
            length = int(attn_mask.sum().item())
            hidden = hidden.clone()
            hidden[:, length:] = 0  # CosmosT5TextEncoder.encode_prompts と同じゼロパディング規約
            paraphrase_embeddings[phrase] = hidden.detach().cpu()
            print(f"  '{phrase}' -> shape {tuple(hidden.shape)}, valid_len={length}")

    torch.save({"episodes": episodes, "paraphrase_embeddings": paraphrase_embeddings}, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
