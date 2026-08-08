"""precompute_text_directions.py — attractor_verification_report.md §14 future-work item #8
("CFG的な条件付け側介入（テキスト指示側の操作）との比較") のための下準備。

text_conditioning_intervention.py が使う2つの固定「対比」T5埋め込み(shape (1,512,1024))を
一度だけ計算してディスクに保存する、独立した小さいスクリプト。T5-11bエンコーダ(実質
~5.6Bパラメータ)を、~2Bパラメータのpolicy DiT本体と同一プロセス/同一GPUメモリ上で
同時に保持するリスクを避けるため、意図的にpolicyモデルのロードとは完全に別プロセスで
実行する(本スクリプトはpolicyモデルを一切ロードしない)。

CosmosT5TextEncoder(get_t5_emb.py)をそのまま使うと`T5EncoderModel.from_pretrained`が
デフォルトのfp32でロードされ(~22GB、24GB GPUではギリギリ)、後続のpolicyモデルロードと
干渉しうるため、ここではbf16で明示的にロードする(このスクリプト単体でのみT5を保持し、
プロセス終了と同時に解放されるので、後段のpolicyロードとは時間的に完全分離される)。

出力: {name: torch.Tensor of shape (1, 512, 1024), dtype=bfloat16} の辞書を torch.save。
"""

import argparse

import torch
from transformers import T5EncoderModel, T5TokenizerFast

MODEL_NAME = "google-t5/t5-11b"
MAX_LENGTH = 512

# 2つの固定「対比」フレーズ:
#  - neg_open: グリッパー開閉という意味内容に直接関連する対比 (real方向のuncondition側)
#  - neg_filler: 意味的に無関係な対比 (random方向のuncondition側、内容非特異性の統制)
PHRASES = {
    "neg_open": "Keep the gripper open.",
    "neg_filler": "The weather today is sunny.",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_path", required=True)
    args = p.parse_args()

    print(f"Loading T5 tokenizer/encoder ({MODEL_NAME}, bf16)...")
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
    encoder = T5EncoderModel.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to("cuda").eval()
    print("Loaded.")

    out = {}
    with torch.inference_mode():
        for name, phrase in PHRASES.items():
            enc = tokenizer.batch_encode_plus(
                [phrase], return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LENGTH,
            )
            input_ids = enc.input_ids.cuda()
            attn_mask = enc.attention_mask.cuda()
            hidden = encoder(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state
            length = int(attn_mask.sum().item())
            # CosmosT5TextEncoder.encode_prompts と同じゼロパディング規約に合わせる
            hidden = hidden.clone()
            hidden[:, length:] = 0
            out[name] = hidden.detach().cpu()
            print(f"  '{phrase}' -> shape {tuple(hidden.shape)}, valid_len={length}")

    torch.save(out, args.out_path)
    print(f"Saved: {args.out_path}")


if __name__ == "__main__":
    main()
