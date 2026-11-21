"""Target VLAs with tree verification: pi0-FAST (LeRobot), ActionCodec-AR (SmolVLM) and StarVLA-FAST (Qwen2.5-VL).

All run in bfloat16 with SDPA attention. `prefill`, `consume` and `consume_tree`
return the final hidden state and record the concatenated tap-layer features in
`last_features`, which condition the draft.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

PI0_FAST = dict(repo='lerobot/pi0fast-libero', revision='840f4b503f4c09110421c33c810a85b6684fd658')
ACTIONCODEC = dict(repo='ZibinDong/SmolVLM2-2.2B-ActionCodec-AR-LIBERO', revision='da65dd2b1e09d3c4176b2a0e66f63ff84c387a73')
STARVLA = dict(repo='StarVLA/Qwen2.5-VL-FAST-LIBERO-4in1', revision='2a30a316ab65ab1a3172082fa7ef4b7121beb447')
STARVLA_PROCESSOR = dict(repo='StarVLA/Qwen2.5-VL-3B-Instruct-Action', revision='97163e6190ca87d6abae7a4dd15840cadde2da1d')
FAST_TOKENIZER = dict(repo='physical-intelligence/fast', revision='ec4d7aa71691cac0b8bed6942be45684db2110f4')


class Target:
    end_ids: tuple
    max_tokens: int
    action_steps: int
    tap_indices: tuple

    def features(self, out):
        return torch.cat([out.hidden_states[i] for i in self.tap_indices], -1)

    def embed(self, ids):
        return self.embed_tokens(ids).to(self.dtype)

    def rope(self, positions):
        """Position ids passed to the language model."""
        return positions

    def scores(self, hidden, committed=(), tree=None):
        """Next-token scores at each position of `hidden`.

        `committed` (generated tokens before `hidden`) and `tree` (the verified draft tree)
        give the token history for targets whose decoding applies logits processors.
        """
        return self.head(hidden).float()

    @torch.inference_mode()
    def consume(self, ids, cache, valid):
        """Append tokens with causal masking (autoregressive decoding)."""
        emb = self.embed(ids)
        b, k = ids.shape
        positions = valid.sum(1, keepdim=True) + torch.arange(k, device=ids.device)[None]
        causal = torch.ones(k, k, device=ids.device, dtype=torch.bool).tril()[None].expand(b, -1, -1)
        mask = torch.cat((valid[:, None, :].expand(b, k, -1), causal), -1)
        out = self.lm(inputs_embeds=emb, attention_mask=self.mask4(mask, emb.dtype), position_ids=self.rope(positions),
                      past_key_values=cache, use_cache=True, output_hidden_states=True)
        self.last_features = self.features(out)
        valid = torch.cat((valid, torch.ones_like(ids, dtype=torch.bool)), 1)
        return out.last_hidden_state, out.past_key_values, emb, valid, positions

    @torch.inference_mode()
    def consume_tree(self, ids, cache, valid, ancestors, depths):
        """One target pass over the draft tree: ancestor-only attention and path-depth RoPE."""
        emb = self.embed(ids)
        positions = valid.sum(1, keepdim=True) + depths
        mask = torch.cat((valid[:, None, :].expand(1, ids.shape[1], -1), ancestors[None]), -1)
        out = self.lm(inputs_embeds=emb, attention_mask=self.mask4(mask, emb.dtype), position_ids=self.rope(positions),
                      past_key_values=cache, use_cache=True, output_hidden_states=True)
        self.last_features = self.features(out)
        return out.last_hidden_state, out.past_key_values, positions

    @staticmethod
    def select_cache_path(cache, base_length, path):
        """Keep the committed prefix plus the accepted tree path in the KV cache."""
        for layer in cache.layers:
            keep = torch.cat((torch.arange(base_length, device=layer.keys.device),
                              torch.tensor(path, device=layer.keys.device) + base_length))
            layer.keys = layer.keys.index_select(-2, keep)
            layer.values = layer.values.index_select(-2, keep)
        return cache

    @torch.inference_mode()
    def generate(self, batch, record=False):
        """Greedy autoregressive decoding. With record=True, also returns draft training features."""
        h, cache, emb, valid, pos = self.prefill(batch)
        compact = valid[0]
        hidden, embedding, positions = [h[0, compact].cpu()], [emb[0, compact].cpu()], [pos[0, compact].cpu()]
        features = [self.last_features[0, compact].cpu()]
        tokens = []
        for step in range(self.max_tokens):
            token = int(self.scores(h[:, -1:], tokens)[0, 0].argmax())
            tokens.append(token)
            if token in self.end_ids or step + 1 == self.max_tokens:
                break
            h, cache, emb, valid, pos = self.consume(torch.tensor([[token]], device=self.device), cache, valid)
            if record:
                hidden.append(h[0].cpu())
                embedding.append(emb[0].cpu())
                positions.append(pos[0].cpu())
                features.append(self.last_features[0].cpu())
        result = dict(tokens=torch.tensor(tokens), prefix_length=int(compact.sum()), terminated=tokens[-1] in self.end_ids)
        if record:
            result.update(hidden=torch.cat(hidden), embedding=torch.cat(embedding), positions=torch.cat(positions),
                          features=torch.cat(features), tap_indices=list(self.tap_indices))
        return result


class Pi0Fast(Target):
    def __init__(self, tap_indices, device='cuda'):
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig
        from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy
        assets = Path(snapshot_download(**{'repo_id': PI0_FAST['repo'], 'revision': PI0_FAST['revision']}))
        cfg = PI0FastConfig.from_pretrained(assets)
        cfg.device, cfg.compile_model, cfg.gradient_checkpointing = 'cpu', False, False
        self.policy = PI0FastPolicy(cfg)
        state = self.policy._fix_pytorch_state_dict_keys(load_file(str(assets / 'model.safetensors')), cfg)
        self.policy.load_state_dict({k if k.startswith('model.') else 'model.' + k: v for k, v in state.items()}, strict=True)
        self.policy.to(device).eval().requires_grad_(False)
        cfg.device = device
        self.core = self.policy.model
        vlm = self.core.paligemma_with_expert
        self.lm = vlm.paligemma.model.language_model
        self.lm.config._attn_implementation = 'sdpa'
        self.head = vlm.paligemma.lm_head
        self.embed_tokens = vlm.embed_language_tokens
        self.dtype = self.lm.layers[0].self_attn.q_proj.weight.dtype
        self.tokenizer = self.policy._paligemma_tokenizer
        self.end_ids = (self.tokenizer.convert_tokens_to_ids('|'),)
        self.max_tokens = cfg.max_decoding_steps
        self.action_steps = cfg.n_action_steps
        self.tap_indices = tuple(tap_indices)
        self.device = device
        self.pre, self.post = make_pre_post_processors(cfg, str(assets), preprocessor_overrides={'device_processor': {'device': device}})

    def preprocess(self, observation):
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
        batch = {'observation.state': torch.as_tensor(np.array(observation['state'], copy=True), dtype=torch.float32),
                 'task': observation['task']}
        for source, name in (('image', 'image'), ('wrist_image', 'image2')):
            image = torch.as_tensor(np.array(observation[source], copy=True)).permute(2, 0, 1).float() / 255
            batch['observation.images.' + name] = image
        batch = self.pre(batch)
        tokens, mask = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
        if not bool((tokens[:, 0] == self.tokenizer.bos_token_id).all()):
            if bool(mask[:, -1].any()):
                raise ValueError('Cannot insert the prompt BOS into a full prompt')
            batch[OBS_LANGUAGE_TOKENS] = torch.cat((torch.full_like(tokens[:, :1], self.tokenizer.bos_token_id), tokens[:, :-1]), 1)
            batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.cat((torch.ones_like(mask[:, :1]), mask[:, :-1]), 1)
        return batch

    @staticmethod
    def mask4(mask, dtype):
        from lerobot.policies.common.vla_utils import prepare_attention_masks_4d
        return prepare_attention_masks_4d(mask, dtype=dtype)

    @torch.inference_mode()
    def prefill(self, batch):
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
        images, image_masks = self.policy._preprocess_images(batch)
        tokens, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
        bos = torch.full((1, 1), self.tokenizer.bos_token_id, device=self.device, dtype=torch.long)
        emb, valid, mask, _, _ = self.core.embed_prefix_fast(images, image_masks, torch.cat((tokens, bos), 1),
                                                             torch.cat((masks, torch.ones_like(bos, dtype=torch.bool)), 1))
        emb = emb.to(self.dtype)
        positions = valid.long().cumsum(1) - 1
        out = self.lm(inputs_embeds=emb, attention_mask=self.mask4(mask, emb.dtype), position_ids=positions,
                      use_cache=True, output_hidden_states=True)
        self.last_features = self.features(out)
        return out.last_hidden_state, out.past_key_values, emb, valid, positions

    @torch.inference_mode()
    def decode(self, tokens):
        ids = torch.as_tensor(tokens, device=self.device)[None]
        actions = self.policy.detokenize_actions(ids, self.action_steps, 7)
        return self.post(actions).cpu().numpy()[0]


class ActionCodec(Target):
    def __init__(self, tap_indices, repo, device='cuda'):
        """repo: a clone of github.com/ZibinDong/actioncodec (provides the action tokenizer)."""
        from huggingface_hub import snapshot_download
        from torchvision.transforms.v2 import CenterCrop, Compose, Resize
        from transformers import AutoModelForImageTextToText
        repo = Path(repo)
        sys.path.insert(0, str(repo))
        import actioncodec  # noqa: F401  registers the action tokenizer model type with Transformers
        from utils.vla_tokenizer import VisionLanguageActionProcessor
        spec = importlib.util.spec_from_file_location('actioncodec_scripts_utils', repo / 'scripts' / 'utils.py')
        self.utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.utils)
        assets = snapshot_download(**{'repo_id': ACTIONCODEC['repo'], 'revision': ACTIONCODEC['revision']})
        self.transform = Compose([Resize((224, 224)), CenterCrop(196), Resize((224, 224))])
        self.model = AutoModelForImageTextToText.from_pretrained(
            assets, trust_remote_code=True, dtype=torch.bfloat16, _attn_implementation='sdpa').to(device).eval().requires_grad_(False)
        self.processor = VisionLanguageActionProcessor.from_pretrained(assets, trust_remote_code=True, mode='discrete')
        self.end_ids = (int(self.model.generation_config.eos_token_id),)
        self.lm = self.model.model.text_model
        self.head = self.model.get_output_embeddings()
        self.embed_tokens = self.model.get_input_embeddings()
        self.dtype = torch.bfloat16
        self.max_tokens = 1000
        self.action_steps = 20
        self.tap_indices = tuple(tap_indices)
        self.device = device
        self._inputs = None
        self.lm.register_forward_pre_hook(self._capture, with_kwargs=True)

    def _capture(self, module, args, kwargs):
        self._inputs = kwargs['inputs_embeds']

    def preprocess(self, observation):
        u = self.utils
        state = (np.asarray(observation['state']) - u.STATE_Q01) / (u.STATE_Q99 - u.STATE_Q01) * 2 - 1
        images = [self.transform(torch.from_numpy(np.asarray(observation[k]).copy()).permute(2, 0, 1)) for k in ('image', 'wrist_image')]
        message = u.prompt_template(state, None, observation['task'], mode='discrete', action_vocab_size=2048, action_token_len=16)
        message[1]['content'] = message[1]['content'][1:]
        texts = self.processor.apply_chat_template([message], add_generation_prompt=True)
        batch = self.processor(text=texts, images=[[torch.cat(images, dim=-1).numpy()]], return_tensors='pt', padding=True,
                               padding_side='left', action_processor_kwargs={'embodiment_ids': 0})
        return u.dict_apply(lambda x: x.to(self.device, torch.bfloat16), batch)

    @staticmethod
    def mask4(mask, dtype):
        return torch.zeros(mask.shape, device=mask.device, dtype=dtype).masked_fill(~mask, torch.finfo(dtype).min)[:, None]

    @torch.inference_mode()
    def prefill(self, batch):
        out = self.model(**batch, use_cache=True, output_hidden_states=True, logits_to_keep=1)
        valid = batch['attention_mask'].bool()
        self.last_features = self.features(out)
        return out.hidden_states[-1], out.past_key_values, self._inputs, valid, valid.long().cumsum(1) - 1

    def decode(self, tokens):
        rows = self.processor.batch_decode(torch.as_tensor(tokens, device=self.device)[None], skip_special_tokens=False, actions_only=True)
        if rows[0] is None:
            raise ValueError('Token sequence does not start with a valid action')
        action = np.clip(np.asarray(rows[0], dtype=np.float32), -1, 1)
        action[..., :-1] *= np.maximum(np.abs(self.utils.ACTION_Q99), np.abs(self.utils.ACTION_Q01))[..., :-1]
        action[..., -1] *= -1
        return action


class StarVLA(Target):
    """StarVLA Qwen2.5-VL-FAST (LIBERO 4-in-1): the autoregressive VLM with its released decoding.

    Released generation applies a repetition penalty over the prompt and generated tokens,
    stops at either end-of-text token and caps the full sequence at 2048 tokens.
    """
    ACTION_TOKEN_MIN = 151665
    ACTION_TOKEN_MAX = 153712
    MAX_LENGTH = 2048

    def __init__(self, tap_indices, device='cuda'):
        import yaml
        from huggingface_hub import snapshot_download
        from transformers import AutoProcessor, GenerationConfig, PreTrainedTokenizerFast, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
        from transformers.initialization import no_init_weights
        assets = Path(snapshot_download(**{'repo_id': STARVLA['repo'], 'revision': STARVLA['revision']}))
        processor = Path(snapshot_download(**{'repo_id': STARVLA_PROCESSOR['repo'], 'revision': STARVLA_PROCESSOR['revision']},
                                           ignore_patterns=['*.safetensors']))
        fast = Path(snapshot_download(**{'repo_id': FAST_TOKENIZER['repo'], 'revision': FAST_TOKENIZER['revision']}))
        cfg = Qwen2_5_VLConfig.from_pretrained(processor)
        cfg._attn_implementation = 'sdpa'
        dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with no_init_weights():
                self.model = Qwen2_5_VLForConditionalGeneration(cfg)
        finally:
            torch.set_default_dtype(dtype)
        released = torch.load(assets / 'checkpoints' / 'steps_30000_pytorch_model.pt', map_location='cpu', mmap=True, weights_only=True)
        prefix = 'qwen_vl_interface.model.'
        if any(not k.startswith(prefix) for k in released):
            raise ValueError('Unexpected non-VLM tensors in the StarVLA checkpoint')
        state = self.remap({k[len(prefix):]: v for k, v in released.items()}, self.model.state_dict())
        self.model.load_state_dict(state, strict=True, assign=True)
        self.model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(processor)
        self.generation = GenerationConfig.from_pretrained(processor)
        if self.generation.do_sample and self.generation.top_k != 1:
            raise ValueError('Only deterministic (top-k 1) released decoding is supported')
        release = yaml.safe_load((assets / 'config.yaml').read_text())
        self.prompt = release['datasets']['vla_data']['CoT_prompt']
        self.action_steps = release['framework']['action_model']['action_horizon']
        stats = json.loads((assets / 'dataset_statistics.json').read_text())['franka']['action']
        self.action_low, self.action_high = np.asarray(stats['min']), np.asarray(stats['max'])
        spec = importlib.util.spec_from_file_location('fast_action_tokenizer', fast / 'processing_action_tokenizer.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        options = json.loads((fast / 'processor_config.json').read_text())
        self.fast = module.UniversalActionProcessor(
            bpe_tokenizer=PreTrainedTokenizerFast(tokenizer_file=str(fast / 'tokenizer.json'), clean_up_tokenization_spaces=False),
            scale=options['scale'], vocab_size=options['vocab_size'], min_token=options['min_token'])
        self.lm = self.model.model.language_model
        self.head = self.model.lm_head
        self.embed_tokens = self.model.get_input_embeddings()
        self.dtype = torch.bfloat16
        eos = self.generation.eos_token_id
        self.end_ids = tuple([eos] if isinstance(eos, int) else eos)
        self.max_tokens = self.MAX_LENGTH
        self.tap_indices = tuple(tap_indices)
        self.device = device
        self._inputs = None
        self.lm.register_forward_pre_hook(self._capture, with_kwargs=True)

    @staticmethod
    def remap(state, expected):
        """Released Transformers 4.x key layout to the 5.x Qwen2.5-VL layout."""
        result = {}
        for key, value in state.items():
            candidates = [key]
            if key.startswith('visual.'):
                candidates.append('model.' + key)
            if key.startswith('model.') and not key.startswith(('model.visual.', 'model.language_model.')):
                candidates.append('model.language_model.' + key[len('model.'):])
            matches = [k for k in candidates if k in expected]
            if len(matches) != 1:
                raise ValueError(f'Unexpected StarVLA weight: {key}')
            result[matches[0]] = value
        if set(result) != set(expected):
            raise ValueError(f'Missing StarVLA weights: {sorted(set(expected) - set(result))[:5]}')
        return result

    def _capture(self, module, args, kwargs):
        self._inputs = kwargs['inputs_embeds']

    def preprocess(self, observation):
        """StarVLA's chat prompt with the main and wrist images resized to 224 x 224."""
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        images = [Image.fromarray(np.asarray(observation[k])).resize((224, 224), Image.Resampling.BILINEAR) for k in ('image', 'wrist_image')]
        content = [{'type': 'image', 'image': image} for image in images]
        content.append({'type': 'text', 'text': self.prompt.replace('{instruction}', observation['task'])})
        messages = [{'role': 'user', 'content': content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        return self.processor(text=[text], images=images, videos=videos, padding=True, return_tensors='pt').to(self.device)

    @staticmethod
    def mask4(mask, dtype):
        return torch.zeros(mask.shape, device=mask.device, dtype=dtype).masked_fill(~mask, torch.finfo(dtype).min)[:, None]

    def rope(self, positions):
        """Text positions of the multimodal RoPE: offset by the prompt's rope delta on all three axes."""
        return (positions + self.rope_delta)[None].expand(3, -1, -1)

    @torch.inference_mode()
    def prefill(self, batch):
        out = self.model(**batch, use_cache=True, output_hidden_states=True, logits_to_keep=1)
        valid = batch['attention_mask'].bool()
        self.rope_delta = self.model.model.rope_deltas.clone()
        self.prompt_ids = batch['input_ids'][0]
        self.max_tokens = self.MAX_LENGTH - batch['input_ids'].shape[1]
        self.last_features = self.features(out)
        return out.hidden_states[-1], out.past_key_values, self._inputs, valid, valid.long().cumsum(1) - 1

    def scores(self, hidden, committed=(), tree=None):
        """Released repetition penalty, applied with the exact token history of each position."""
        logits = self.head(hidden).float()
        base = self.prompt_ids.tolist() + list(committed)
        histories = [base] if tree is None else [base + tree.path(i) for i in range(len(tree.tokens))]
        width = max(map(len, histories))
        # Padding repeats a prompt token: the penalty depends on presence, not count.
        ids = torch.tensor([h + [base[0]] * (width - len(h)) for h in histories], device=logits.device)
        rows = logits[0]
        values = rows.gather(1, ids)
        values = torch.where(values < 0, values * self.generation.repetition_penalty, values / self.generation.repetition_penalty)
        return rows.scatter(1, ids, values)[None]

    def decode(self, tokens):
        ids = [int(t) - self.ACTION_TOKEN_MIN for t in tokens if self.ACTION_TOKEN_MIN <= int(t) <= self.ACTION_TOKEN_MAX]
        action = np.asarray(self.fast.decode([ids], time_horizon=self.action_steps, action_dim=7))[0]
        result = action.copy()
        result[:, :6] = .5 * (action[:, :6] + 1) * (self.action_high[:6] - self.action_low[:6]) + self.action_low[:6]
        result[:, -1] = 1. - 2. * (action[:, -1] > .5)
        return result


def load_target(name, tap_indices, actioncodec_repo=None, device='cuda'):
    if name == 'pi0_fast':
        return Pi0Fast(tap_indices, device)
    if name == 'actioncodec':
        if actioncodec_repo is None:
            raise ValueError('--actioncodec-repo is required for ActionCodec')
        return ActionCodec(tap_indices, actioncodec_repo, device)
    if name == 'starvla':
        return StarVLA(tap_indices, device)
    raise ValueError(name)


def model_config(name):
    return json.loads((Path(__file__).resolve().parents[1] / 'configs' / f'{name}.json').read_text())
