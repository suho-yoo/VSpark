"""Target rollouts from demonstration states, optionally with calibrated Gaussian action noise (Sec. 3.2)."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from . import libero


def restore(env, episode):
    """Reset the scene from the demonstration XML and set its recorded simulator state."""
    import libero.libero.utils.utils as libero_utils
    from libero.libero import get_libero_path
    tree = ET.fromstring(libero_utils.postprocess_model_xml(episode['model_xml'], {}))
    for element in tree.iter():
        path = element.get('file')
        if path and not Path(path).is_file() and '/assets/' in path:
            element.set('file', str(Path(get_libero_path('assets')) / path.split('/assets/', 1)[1]))
    env.reset()
    env.reset_from_xml_string(ET.tostring(tree, encoding='unicode'))
    env.sim.reset()
    obs = env.set_init_state(np.asarray(episode['state'], dtype=np.float64))
    if not np.allclose(env.sim.get_state().flatten(), episode['state'], atol=1e-7, rtol=0):
        raise RuntimeError('Demonstration state did not restore')
    return obs


def perturb(action, rng, covariance):
    """Zero-mean Gaussian noise on the six arm dimensions; the gripper is left unchanged."""
    out = np.asarray(action, dtype=np.float64).copy()
    if covariance is not None:
        out[:6] += rng.multivariate_normal(np.zeros(6), covariance, check_valid='raise')
    return np.clip(out, -1, 1)


def eligible(target, trace, support):
    """Terminated chunk whose tokens all lie in the draft vocabulary (and, for pi0-FAST, start with 'Action:')."""
    ids = trace['tokens'].tolist()
    if not trace['terminated'] or not ids or any(t not in support for t in ids):
        return False
    if hasattr(target, 'tokenizer'):
        prefix = target.tokenizer.encode('Action:', add_special_tokens=False)
        return ids[:len(prefix)] == prefix
    return True


def rollouts(target, suite, task_id, states_dir, seed, covariance, support):
    """Yield (episode, step, trace, actions) over target rollouts from the demonstration states of one task.

    Episode i starts from demonstration state i with seed `seed + i`; noise is applied to executed actions only.
    """
    from libero.libero import benchmark
    task = benchmark.get_benchmark_dict()[suite]().get_task(task_id)
    source = json.loads((Path(states_dir) / suite / (Path(task.bddl_file).stem + '_demo.json')).read_text())
    env = libero.make_env(task)
    try:
        for index, episode in enumerate(source['episodes']):
            env.seed(seed + index)
            obs = restore(env, episode)
            rng = np.random.default_rng(seed + index)
            if episode['state_index'] == 0:
                obs = libero.settle(env)
            for step in range(0, libero.horizon(suite), target.action_steps):
                trace = target.generate(target.preprocess(libero.observation(obs, task)), record=True)
                try:
                    actions = np.asarray(target.decode(trace['tokens']), dtype=np.float64)
                except (ValueError, AssertionError):
                    break
                if actions.shape != (target.action_steps, 7) or not np.isfinite(actions).all():
                    break
                yield index, step, trace if eligible(target, trace, support) else None, actions
                success = False
                for action in actions:
                    obs, _, success, _ = env.step(perturb(action, rng, covariance).tolist())
                    if success:
                        break
                if success:
                    break
    finally:
        env.close()
