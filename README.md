# VSpark: Speculative Decoding for Autoregressive Vision-Language-Action Models

[![arXiv](https://img.shields.io/badge/arXiv-2611.22666-b31b1b.svg)](https://arxiv.org/abs/2611.22666)
[![Project Page](https://img.shields.io/badge/Project-Page-002147.svg)](https://suho-yoo.github.io/VSpark/)

VSpark accelerates autoregressive VLAs with speculative decoding while keeping verification exact for lossless decoding. It trains a one-layer draft on target rollouts with calibrated action noise and aligns it with the target's hidden features. At inference, it reuses the target's predictions along rejected branches as candidates for the next draft tree.

## Installation

```bash
git clone https://github.com/suho-yoo/VSpark.git
cd VSpark
conda create -n vspark python=3.12 -y
conda activate vspark
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Install LIBERO and point it to its assets:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO --config-settings editable_mode=compat
export MUJOCO_GL=egl
```

pi0-FAST uses the PaliGemma tokenizer, so accept its license on Hugging Face (`google/paligemma-3b-pt-224`) and run `huggingface-cli login`.

ActionCodec loads its action tokenizer from the official repository (its dependencies are already in `requirements.txt`):

```bash
git clone https://github.com/ZibinDong/actioncodec.git
git -C actioncodec checkout f034eb9a9da6d6f37b9522b4548779e8a9886b21
```

ActionCodec's released code requires Transformers 4, so run every `--model actioncodec` command in a separate environment:

```bash
conda create -n vspark-actioncodec --clone vspark -y
conda activate vspark-actioncodec
pip install transformers==4.57.1 tokenizers==0.22.1 huggingface_hub==0.35.0
```

Pass `--actioncodec-repo actioncodec` to every command with `--model actioncodec`.

StarVLA needs no extra step: its weights, processor and FAST tokenizer are downloaded on first use.

For LIBERO-Plus or LIBERO-Pro, install [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) or [LIBERO-Pro](https://github.com/Zxy-MLlab/LIBERO-PRO) in place of LIBERO, following their instructions.

## Data and weights

| What | Source |
|---|---|
| pi0-FAST target | [lerobot/pi0fast-libero](https://huggingface.co/lerobot/pi0fast-libero) |
| ActionCodec target | [ZibinDong/SmolVLM2-2.2B-ActionCodec-AR-LIBERO](https://huggingface.co/ZibinDong/SmolVLM2-2.2B-ActionCodec-AR-LIBERO) |
| ActionCodec tokenizer code | [ZibinDong/actioncodec](https://github.com/ZibinDong/actioncodec) |
| StarVLA target | [StarVLA/Qwen2.5-VL-FAST-LIBERO-4in1](https://huggingface.co/StarVLA/Qwen2.5-VL-FAST-LIBERO-4in1) |
| StarVLA processor and action tokenizer | [StarVLA/Qwen2.5-VL-3B-Instruct-Action](https://huggingface.co/StarVLA/Qwen2.5-VL-3B-Instruct-Action), [physical-intelligence/fast](https://huggingface.co/physical-intelligence/fast) |
| LIBERO scenes and initial states | [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) |
| LIBERO-Plus / LIBERO-Pro | [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus), [LIBERO-Pro](https://github.com/Zxy-MLlab/LIBERO-PRO) |
| Training demonstrations (for rollout start states) | [yifengzhu-hf/LIBERO-datasets](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets) |

Models and datasets on Hugging Face are downloaded automatically on first use (the demonstrations by `training/prepare_demo_states.py`) into the default cache; set `HF_HOME` to change it. All other inputs and outputs are passed as command-line paths.

## Checkpoints

Trained drafts are on Hugging Face at [suhoyoo/VSpark](https://huggingface.co/suhoyoo/VSpark). Each draft is paired with its target.

| Target | Draft |
|---|---|
| [pi0-FAST](https://huggingface.co/lerobot/pi0fast-libero) | [vspark_pi0_fast.pt](https://huggingface.co/suhoyoo/VSpark/blob/main/vspark_pi0_fast.pt) |
| [ActionCodec](https://huggingface.co/ZibinDong/SmolVLM2-2.2B-ActionCodec-AR-LIBERO) | [vspark_actioncodec.pt](https://huggingface.co/suhoyoo/VSpark/blob/main/vspark_actioncodec.pt) |
| [StarVLA](https://huggingface.co/StarVLA/Qwen2.5-VL-FAST-LIBERO-4in1) | [vspark_starvla.pt](https://huggingface.co/suhoyoo/VSpark/blob/main/vspark_starvla.pt) |

```bash
huggingface-cli download suhoyoo/VSpark --local-dir checkpoints
```

Target weights are downloaded from Hugging Face on first use.

## Inference

Evaluate VSpark on one LIBERO suite (`libero_goal`, `libero_spatial`, `libero_object`, `libero_10`). `--model` is `pi0_fast`, `actioncodec` or `starvla`:

```bash
CUDA_VISIBLE_DEVICES=0 python inference/eval_libero.py \
  --model pi0_fast \
  --suite libero_goal \
  --checkpoint checkpoints/vspark_pi0_fast.pt \
  --output outputs/pi0_fast/libero_goal/vspark
```

Run the target with plain autoregressive decoding as the reference, then compute the speedup:

```bash
CUDA_VISIBLE_DEVICES=0 python inference/eval_libero.py \
  --model pi0_fast --suite libero_goal --ar \
  --output outputs/pi0_fast/libero_goal/ar

python inference/speedup.py \
  --ar outputs/pi0_fast/libero_goal/ar \
  --vspark outputs/pi0_fast/libero_goal/vspark
```

Each run writes `episodes.jsonl` and `summary.json` with the success rate, the acceptance length τ and the decoding time per action. Following the paper, episodes with invalid action outputs and chunks that reach the token limit are excluded from timing.

**LIBERO-Plus** runs one episode per perturbed task:

```bash
CUDA_VISIBLE_DEVICES=0 python inference/eval_libero.py \
  --model actioncodec --actioncodec-repo actioncodec \
  --benchmark plus --suite libero_goal \
  --checkpoint checkpoints/vspark_actioncodec.pt \
  --output outputs/actioncodec/plus_libero_goal/vspark
```

**LIBERO-Pro** suites are named `<suite>_<perturbation>` with perturbations `env`, `lan`, `object`, `swap` and `task`, e.g. `--benchmark pro --suite libero_goal_lan`. The `env` suites need initial states generated with the benchmark's `official_generate_init_states.py` (seed 28, 50 states per task).

## Training

**1. Demonstration states.** Rollouts start from simulator states of the official LIBERO training demonstrations, never from benchmark initial states:

```bash
python training/prepare_demo_states.py --output data/demo_states
```

**2. Noisy rollouts.** Roll out the target with calibrated Gaussian action noise and record its clean outputs. The noise covariances used in the paper are in `configs/<model>.json`; the per-task chunk counts match the paper (27,548 chunks per target):

```bash
for suite in libero_goal libero_spatial libero_object libero_10; do
  CUDA_VISIBLE_DEVICES=0 python training/collect_rollouts.py \
    --model pi0_fast --suite $suite \
    --states data/demo_states \
    --output data/pi0_fast_rollouts
done
```

Each chunk stores the target's hidden states and tap features (about 20 MB per chunk for pi0-FAST), so plan for several hundred GB of storage.

**3. Train the draft.**

```bash
CUDA_VISIBLE_DEVICES=0 python training/train.py \
  --model pi0_fast \
  --data data/pi0_fast_rollouts \
  --output checkpoints/pi0_fast
```

This runs 10 epochs (17,210 updates) and writes `epoch_010.pt`.

**Calibrating your own noise (optional).** Train a draft on clean rollouts (`--noise none`), then measure its action errors:

```bash
CUDA_VISIBLE_DEVICES=0 python training/calibrate_noise.py \
  --model pi0_fast --checkpoint checkpoints/pi0_fast_clean/epoch_010.pt \
  --states data/demo_states --output configs/pi0_fast_noise.json
```

Pass the result to `collect_rollouts.py --noise configs/pi0_fast_noise.json`.

## Code structure

```
vspark/
├── draft.py       # one-layer parallel draft with a Markov transition head
├── loss.py        # sampled-anchor block loss: CE + feature alignment
├── tree.py        # draft tree, exact early stopping, continuation reuse
├── decode.py      # speculative decoding with exact verification
├── targets.py     # pi0-FAST, ActionCodec and StarVLA with tree verification
├── rollouts.py    # target rollouts from demonstration states with action noise
└── libero.py      # LIBERO / LIBERO-Plus / LIBERO-Pro evaluation protocol
training/          # demonstration states, noise calibration, rollout collection, training
inference/         # evaluation and speedup
configs/           # draft configurations, noise covariances, chunk counts
docs/              # project page (GitHub Pages)
```

## Acknowledgments

VSpark builds on [DSpark](https://arxiv.org/abs/2607.05147) and adapts its anchor sampling from [DeepSpec](https://github.com/deepseek-ai/DeepSpec). The targets come from [LeRobot](https://github.com/huggingface/lerobot) (pi0-FAST), [ActionCodec](https://github.com/ZibinDong/actioncodec) and [StarVLA](https://github.com/starVLA/starVLA). We evaluate on [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) and [LIBERO-Pro](https://github.com/Zxy-MLlab/LIBERO-PRO).

## Citation

```bibtex
@article{yoo2026vspark,
  title   = {VSpark: Speculative Decoding for Autoregressive Vision-Language-Action Models},
  author  = {Yoo, Suho and Jang, Youngjoon and Cho, Hyebin and Lee, Taeyeop and Jeong, Jaewook and Chung, Joon Son},
  journal = {arXiv preprint arXiv:2611.22666},
  year    = {2026}
}
```

## License

The code is released under the Apache-2.0 license (see [LICENSE](LICENSE)). Third-party attributions are listed in [NOTICE](NOTICE). Pretrained target weights and datasets keep their original licenses, and each released draft checkpoint follows the license of its target (see the [model card](https://huggingface.co/suhoyoo/VSpark)).
