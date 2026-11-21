"""Extract simulator start states from the official LIBERO training demonstrations.

For every demonstration we keep its initial state and the states at 20/40/60/80% of
the trajectory, together with the scene XML. Rollouts never use benchmark initial states.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

SUITES = ('libero_goal', 'libero_spatial', 'libero_object', 'libero_10')
DATASET = dict(repo_id='yifengzhu-hf/LIBERO-datasets', revision='f13aa24a3da8c43c7225569f28c562979fa0e35a')


def extract(path):
    rows = []
    with h5py.File(path, 'r') as f:
        for name in sorted(f['data'], key=lambda n: int(n.split('_')[-1])):
            demo = f['data'][name]
            xml = demo.attrs['model_file']
            xml = xml.decode() if isinstance(xml, bytes) else xml
            states = demo['states']
            indices = [0] + [i for i in (int((len(states) - 1) * q) for q in (.2, .4, .6, .8)) if i]
            for index in indices:
                rows.append(dict(demo=name, state_index=index, state=np.asarray(states[index], dtype=np.float64).tolist(), model_xml=str(xml)))
    return sorted(rows, key=lambda r: (r['state_index'] != 0, int(r['demo'].split('_')[-1]), r['state_index']))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--suites', nargs='+', default=list(SUITES))
    args = p.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    files = HfApi().list_repo_files(**DATASET, repo_type='dataset')
    for suite in args.suites:
        for name in sorted(f for f in files if f.startswith(suite + '/') and f.endswith('_demo.hdf5')):
            local = hf_hub_download(filename=name, repo_type='dataset', **DATASET)
            destination = args.output / suite / (Path(name).stem + '.json')
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(dict(task_file=Path(name).stem, episodes=extract(local))))
            print(destination, flush=True)


if __name__ == '__main__':
    main()
