"""LIBERO, LIBERO-Plus and LIBERO-Pro environments under the OpenPI evaluation protocol."""
from pathlib import Path

import numpy as np
import torch

HORIZONS = {'libero_spatial': 220, 'libero_object': 280, 'libero_goal': 300, 'libero_10': 520}
SETTLE_STEPS = 10
SEED = 7


def horizon(suite):
    """Base-suite horizon; LIBERO-Pro suites are named <base>_<perturbation>."""
    base = max((b for b in HORIZONS if suite == b or suite.startswith(b + '_')), key=len)
    return HORIZONS[base]


def make_env(task, image_size=256):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bddl = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
    return OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=image_size, camera_widths=image_size)


def initial_states(benchmark, suite, task_id):
    if benchmark == 'libero':
        from libero.libero import get_libero_path
        task = suite.get_task(task_id)
        return torch.load(Path(get_libero_path('init_states')) / task.problem_folder / task.init_states_file, weights_only=False)
    return suite.get_task_init_states(task_id)


def observation(obs, task):
    from robosuite.utils.transform_utils import quat2axisangle
    return dict(image=obs['agentview_image'][::-1, ::-1].copy(),
                wrist_image=obs['robot0_eye_in_hand_image'][::-1, ::-1].copy(),
                state=np.concatenate((obs['robot0_eef_pos'], quat2axisangle(obs['robot0_eef_quat']), obs['robot0_gripper_qpos'])),
                task=task.language)


def settle(env):
    obs = None
    for _ in range(SETTLE_STEPS):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    return obs
