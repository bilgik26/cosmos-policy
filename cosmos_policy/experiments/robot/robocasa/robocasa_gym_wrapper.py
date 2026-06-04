"""
RoboCasa Gym Wrapper for FPO++ online RL.

Single-env wrapper + multi-process vectorized env.
Adapted from manipulation_experiments/src/dexmg_env.py patterns,
adjusted for RoboCasa observation keys and scene-per-reset behaviour.
"""

from __future__ import annotations

import multiprocessing as mp
import traceback
from collections import OrderedDict
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Worker ↔ manager command tokens
class _Cmd(Enum):
    RESET = "reset"
    STEP = "step"
    CLOSE = "close"
    GET_ATTR = "get_attr"


# RoboCasa observations we care about
_OBS_KEYS = (
    "robot0_agentview_left_image",
    "robot0_agentview_right_image",
    "robot0_eye_in_hand_image",
    "robot0_gripper_qpos",
    "robot0_eef_pos",
    "robot0_eef_quat",
)

# Manipulation action dim used by the policy (7-DOF OSC_POSE)
POLICY_ACTION_DIM = 7
# Full action dim expected by RoboCasa (7 + 5 mobile-base zeros)
ENV_ACTION_DIM = 12
_MOBILE_BASE_ZEROS = np.array([0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Observation processing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _extract_obs(raw_obs: dict, flip_images: bool = True) -> Dict[str, np.ndarray]:
    """Convert a raw robosuite observation dict into the canonical format."""
    def _flip(img: np.ndarray) -> np.ndarray:
        return np.flipud(img).copy() if flip_images else img.copy()

    primary   = _flip(raw_obs["robot0_agentview_left_image"])   # (H, W, 3) uint8
    secondary = _flip(raw_obs["robot0_agentview_right_image"])  # (H, W, 3) uint8
    wrist     = _flip(raw_obs["robot0_eye_in_hand_image"])      # (H, W, 3) uint8

    proprio = np.concatenate([
        raw_obs["robot0_gripper_qpos"],   # (2,)
        raw_obs["robot0_eef_pos"],        # (3,)
        raw_obs["robot0_eef_quat"],       # (4,)
    ]).astype(np.float32)                 # (9,)

    return {
        "primary_image":   primary,
        "secondary_image": secondary,
        "wrist_image":     wrist,
        "proprio":         proprio,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Single-environment wrapper
# ──────────────────────────────────────────────────────────────────────────────

class RoboCasaGymWrapper:
    """
    Thin wrapper around a single robosuite RoboCasa env.

    Differences from DexMimicGen:
    - Observation keys use left/right agentview + eye_in_hand.
    - Actions are 7-dim (policy) → zero-padded to 12-dim for the env.
    - Scene changes on every reset() — task_description is refreshed.
    - Images are returned upside-down by the sim and must be flipped.
    """

    def __init__(
        self,
        task_name: str,
        seed: Optional[int] = None,
        img_res: int = 224,
        flip_images: bool = True,
        obj_instance_split: str = "B",
        layout_and_style_ids: Optional[list] = None,
        max_steps: int = 500,
    ):
        import pickle
        import robocasa  # registers RoboCasa envs with robosuite
        import robosuite

        self.task_name = task_name
        self.flip_images = flip_images
        self._task_description: str = ""
        self._max_steps = max_steps
        self._step_count = 0

        controller_path = (
            "cosmos_policy/experiments/robot/robocasa/robocasa_controller_configs.pkl"
        )
        with open(controller_path, "rb") as fh:
            controller_configs = pickle.load(fh)

        env_kwargs = dict(
            env_name=task_name,
            robots="PandaMobile",
            controller_configs=controller_configs,
            camera_names=[
                "robot0_agentview_left",
                "robot0_agentview_right",
                "robot0_eye_in_hand",
            ],
            camera_widths=img_res,
            camera_heights=img_res,
            has_renderer=False,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_object_obs=True,
            use_camera_obs=True,
            camera_depths=False,
            seed=seed,
            obj_instance_split=obj_instance_split,
            generative_textures=None,
            randomize_cameras=False,
            layout_and_style_ids=layout_and_style_ids,
            translucent_robot=False,
        )
        self.env = robosuite.make(**env_kwargs)
        self.action_dim = POLICY_ACTION_DIM

    # ------------------------------------------------------------------
    def reset(self) -> Tuple[Dict[str, np.ndarray], dict]:
        raw_obs = self.env.reset()
        self._step_count = 0
        # Task description changes with every scene reset
        self._task_description = self.env.get_ep_meta()["lang"]
        obs = _extract_obs(raw_obs, self.flip_images)
        obs["task_description"] = self._task_description
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, dict]:
        """
        Args:
            action: (7,) float32 manipulation action.
        Returns:
            obs, reward, terminated, truncated, info
        """
        if action.shape[-1] == POLICY_ACTION_DIM and self.env.action_dim == ENV_ACTION_DIM:
            action = np.concatenate([action, _MOBILE_BASE_ZEROS])
        raw_obs, _, done, info = self.env.step(action)
        self._step_count += 1
        success = bool(self.env._check_success())
        reward = 1.0 if success else 0.0
        truncated = self._step_count >= self._max_steps
        obs = _extract_obs(raw_obs, self.flip_images)
        obs["task_description"] = self._task_description
        return obs, reward, done or success, truncated, {"success": success}

    @property
    def task_description(self) -> str:
        return self._task_description

    def close(self):
        self.env.close()


# ──────────────────────────────────────────────────────────────────────────────
# Worker process (runs inside subprocess)
# ──────────────────────────────────────────────────────────────────────────────

def _worker_fn(
    task_name: str,
    seed: Optional[int],
    img_res: int,
    flip_images: bool,
    obj_instance_split: str,
    cmd_q: mp.Queue,
    obs_q: mp.Queue,
    max_steps: int = 500,
):
    """Worker process: owns one RoboCasaGymWrapper and processes commands."""
    try:
        env = RoboCasaGymWrapper(
            task_name=task_name,
            seed=seed,
            img_res=img_res,
            flip_images=flip_images,
            obj_instance_split=obj_instance_split,
            max_steps=max_steps,
        )
        obs_q.put(("ready", None))

        while True:
            cmd, payload = cmd_q.get()

            if cmd == _Cmd.RESET:
                obs, info = env.reset()
                obs_q.put(("obs", (obs, info)))

            elif cmd == _Cmd.STEP:
                action = payload
                obs, reward, terminated, truncated, info = env.step(action)
                obs_q.put(("transition", (obs, reward, terminated, truncated, info)))

            elif cmd == _Cmd.GET_ATTR:
                attr_name = payload
                obs_q.put(("attr", getattr(env, attr_name, None)))

            elif cmd == _Cmd.CLOSE:
                env.close()
                obs_q.put(("closed", None))
                break

    except Exception:
        obs_q.put(("error", traceback.format_exc()))


# ──────────────────────────────────────────────────────────────────────────────
# Vectorized environment (N parallel processes)
# ──────────────────────────────────────────────────────────────────────────────

class VectorizedRoboCasaEnv:
    """
    N independent RoboCasa environments running in separate processes.

    Interface mirrors manipulation_experiments' VectorizedEnvWrapper:
        reset() → (stacked_obs_dict, list_of_info)
        step(actions) → (stacked_obs_dict, rewards, dones, truncateds, list_of_info)
        close()

    Observations are stacked along axis=0:
        primary_image:   (N, H, W, 3) uint8
        secondary_image: (N, H, W, 3) uint8
        wrist_image:     (N, H, W, 3) uint8
        proprio:         (N, 9)       float32
        task_description: list[str]   length N
    """

    def __init__(
        self,
        task_name: str,
        num_envs: int,
        base_seed: int = 0,
        img_res: int = 224,
        flip_images: bool = True,
        obj_instance_split: str = "B",
        max_steps: int = 500,
    ):
        self.num_envs = num_envs
        self.action_dim = POLICY_ACTION_DIM

        self._cmd_qs: List[mp.Queue] = []
        self._obs_qs: List[mp.Queue] = []
        self._procs: List[mp.Process] = []

        for i in range(num_envs):
            cq = mp.Queue()
            oq = mp.Queue()
            p = mp.Process(
                target=_worker_fn,
                args=(
                    task_name,
                    base_seed + i,
                    img_res,
                    flip_images,
                    obj_instance_split,
                    cq,
                    oq,
                    max_steps,
                ),
                daemon=True,
            )
            p.start()
            self._cmd_qs.append(cq)
            self._obs_qs.append(oq)
            self._procs.append(p)

        # Wait for all workers to be ready
        for i, oq in enumerate(self._obs_qs):
            tag, payload = oq.get(timeout=120)
            if tag == "error":
                raise RuntimeError(f"Worker {i} failed to start:\n{payload}")
            assert tag == "ready"

    # ------------------------------------------------------------------
    def reset(self):
        for cq in self._cmd_qs:
            cq.put((_Cmd.RESET, None))
        results = [self._recv(i, expected="obs") for i in range(self.num_envs)]
        obs_list, info_list = zip(*results)
        return self._stack_obs(obs_list), list(info_list)

    def step(self, actions: np.ndarray):
        """
        Args:
            actions: (N, 7) float32
        """
        for i, cq in enumerate(self._cmd_qs):
            cq.put((_Cmd.STEP, actions[i]))
        results = [self._recv(i, expected="transition") for i in range(self.num_envs)]
        obs_list, rewards, terminateds, truncateds, info_list = zip(*results)
        return (
            self._stack_obs(obs_list),
            np.array(rewards, dtype=np.float32),
            np.array(terminateds, dtype=bool),
            np.array(truncateds, dtype=bool),
            list(info_list),
        )

    def close(self):
        for cq in self._cmd_qs:
            cq.put((_Cmd.CLOSE, None))
        for i in range(self.num_envs):
            self._recv(i, expected="closed")
        for p in self._procs:
            p.join(timeout=10)

    # ------------------------------------------------------------------
    def _recv(self, worker_idx: int, expected: str):
        tag, payload = self._obs_qs[worker_idx].get(timeout=300)
        if tag == "error":
            raise RuntimeError(f"Worker {worker_idx} error:\n{payload}")
        assert tag == expected, f"Expected '{expected}', got '{tag}'"
        return payload

    @staticmethod
    def _stack_obs(obs_list) -> Dict[str, np.ndarray]:
        """Stack per-env obs dicts into batched arrays."""
        keys = [k for k in obs_list[0] if k != "task_description"]
        stacked: Dict[str, np.ndarray] = OrderedDict()
        for k in keys:
            stacked[k] = np.stack([o[k] for o in obs_list], axis=0)
        stacked["task_description"] = [o["task_description"] for o in obs_list]
        return stacked
