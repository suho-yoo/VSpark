"""Speedup of VSpark over autoregressive decoding: AR ms/action divided by VSpark ms/action."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ar', type=Path, required=True, help='Output folder of eval_libero.py --ar')
    p.add_argument('--vspark', type=Path, required=True, help='Output folder of eval_libero.py --checkpoint')
    args = p.parse_args()
    ar = json.loads((args.ar / 'summary.json').read_text())
    vspark = json.loads((args.vspark / 'summary.json').read_text())
    print(json.dumps(dict(suite=vspark['suite'], acceptance_length=round(vspark['acceptance_length'], 2),
                          speedup=round(ar['ms_per_action'] / vspark['ms_per_action'], 2),
                          success_rate_ar=ar['success_rate'], success_rate_vspark=vspark['success_rate']), indent=2))


if __name__ == '__main__':
    main()
