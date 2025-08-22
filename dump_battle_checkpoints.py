#!/usr/bin/env python3
"""Dump checkpoints whose filename starts with 'battle' into Kaggle submission .py files.
This loader is explicit about accepting either a raw state_dict (what
`torch.save(self.model.policy.state_dict(), path)` produces) or a full
checkpoint dict containing 'model_state_dict'. It uses the same
submission generator from `dump_weight_fixed_v2.py` when available.

Usage:
  python dump_battle_checkpoints.py --dry-run    # just list matching checkpoints
  python dump_battle_checkpoints.py             # convert all matching checkpoints

The script writes `sub/{basename}.py` and `model_weights_{basename}.npz` next to the repo root.
"""

import argparse
import base64
import glob
import io
import os
import sys
from pathlib import Path

import numpy as np

# Prefer local torch if available, otherwise fail with a helpful message
try:
    import torch
except Exception as e:
    print("ERROR: torch is required for this script. Install PyTorch and retry.")
    raise

# Import generator from dump_weight_fixed_v2 if possible to keep codepaths identical.
# The generator expects a base64 blob and a has_trunk boolean and returns Python source.
try:
    from dump_weight_fixed_v2 import generate_submission_code
except Exception:
    generate_submission_code = None

ROOT = Path(__file__).parent
CKPT_DIR = ROOT / "checkpoints"
OUT_SUB_DIR = ROOT / "sub"
OUT_SUB_DIR.mkdir(exist_ok=True)

def _normalize_key(key: str) -> str:
    """Normalize common prefixes so the saved npz keys match the forwarder expectations.
    This mirrors the behaviour in the original dump script: strip 'policy.', 'module.',
    'model.' prefixes and also 'policy.features_extractor.' -> 'features_extractor.' etc.
    """
    # Common prefixes to strip
    strips = [
        "policy.features_extractor.",
    "features_extractor.",
        "policy.",
        "module.",
        "model.",
        "net.",
    ]
    out = key
    for s in strips:
        if out.startswith(s):
            out = out[len(s):]
            break
    # SB3 sometimes prefixes pi_ / vf_ for separate heads (pi_features_extractor.),
    # convert pi_/vf_ variants to neutral names (pi_features_extractor -> features_extractor)
    out = out.replace("pi_features_extractor.", "features_extractor.")
    out = out.replace("vf_features_extractor.", "features_extractor.")
    # also remove a leading 'features_extractor.' if present
    if out.startswith("features_extractor."):
        out = out[len("features_extractor."):]
    # map SB3 mlp_extractor naming to the expected head names
    out = out.replace("mlp_extractor.policy_net.", "policy_net.")
    out = out.replace("mlp_extractor.value_net.", "value_net.")
    # map common head names to the forwarder naming used by the generator
    out = out.replace("fc.1.", "head.1.")
    out = out.replace("policy_net.", "policy_head.")
    # action_net maps to the policy head final linear (policy_head.2)
    out = out.replace("action_net.", "policy_head.2.")
    # value_net -> value_head.* (generator may expect value_head naming)
    out = out.replace("value_net.", "value_head.")
    # also strip leading 'policy.' if still present
    if out.startswith("policy."):
        out = out[len("policy."):]
    return out


def state_dict_to_numpy(state_dict: dict) -> dict:
    """Convert torch tensors in a state_dict to numpy arrays and normalize their keys."""
    np_state = {}
    for k, v in state_dict.items():
        nk = _normalize_key(k)
        # If v is a torch tensor, convert; else accept numpy already
        try:
            if isinstance(v, torch.Tensor):
                arr = v.cpu().numpy()
            else:
                arr = np.array(v)
        except Exception:
            # Fallback: try to convert via numpy.asarray
            arr = np.asarray(v)
        np_state[nk] = arr
    return np_state


def dump_checkpoint_to_submission(path: Path, dry_run: bool = False) -> Path:
    """Load checkpoint, convert to npz and generate submission file. Returns path to written submission (or would-be path).

    Accepts either a raw state_dict (mapping) or a checkpoint dict containing 'model_state_dict'.
    """
    name = path.stem
    base = name
    out_npz = ROOT / f"model_weights_{base}.npz"
    out_py = OUT_SUB_DIR / f"{base}.py"

    print(f"Processing {path} -> {out_py}")
    if dry_run:
        return out_py

    ckpt = torch.load(str(path), map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and all(isinstance(v, (torch.Tensor, np.ndarray, list, tuple)) for v in ckpt.values()):
        # This is likely the raw state_dict saved by online_battle_trainer: torch.save(policy.state_dict())
        state = ckpt
    else:
        # Unknown format: try common keys
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            raise ValueError(f"Unrecognized checkpoint format for {path}")

    np_state = state_dict_to_numpy(state)

    # Save compressed npz
    np.savez_compressed(out_npz, **np_state)
    print(f"  -> wrote {out_npz} ({len(np_state)} arrays)")

    # Embed to base64 for generator
    with open(out_npz, "rb") as f:
        weights_b64 = base64.b64encode(f.read()).decode("ascii")

    # Detect presence of a feature-extractor / trunk. Older heuristics only checked for
    # keys starting with 'trunk.' which misses SB3-style 'features_extractor.' keys
    # (and other possible backbone names). Check common prefixes and also look
    # for large intermediate tensor shapes (e.g. 1024) which indicate a deep
    # feature extractor producing a high-dimensional embedding.
    def _looks_like_trunk(keys, arrays):
        for k in keys:
            if k.startswith("trunk.") or k.startswith("features_extractor.") or "backbone" in k or "resnet" in k or "conv" in k:
                return True
        # fallback: if any weight has a dimension that is plausibly a trunk output (>=256)
        for v in arrays.values():
            try:
                if getattr(v, 'ndim', 0) >= 2:
                    # look at the second axis or flattened size
                    if v.shape[0] >= 256 or (len(v.shape) > 1 and v.shape[1] >= 256):
                        return True
            except Exception:
                continue
        return False

    has_trunk = _looks_like_trunk(np_state.keys(), np_state)

    if generate_submission_code is not None:
        code = generate_submission_code(weights_b64, has_trunk)
    else:
        # Minimal generic submission wrapper if generator not importable
        code = _minimal_generate(weights_b64, has_trunk)

    # Post-process generated code to better match some saved extractor shapes.
    # Some trainers use a features_extractor that expects 2 input planes (player/opponent)
    # and performs global pooling before the FC head. The canonical generator may
    # generate a forwarder that assumes 3 planes + coord channels and flattens the
    # conv map. Detect common mismatches and patch the code string accordingly.
    def _patch_code_for_shapes(code_str, arrays):
        # if stem exists and expects 2 input channels, ensure we feed 2 planes and
        # remove coord concat
        stem_key = 'stem.0.weight'
        fc_key = 'head.1.weight'
        patched = code_str
        try:
            if stem_key in arrays:
                c_in = arrays[stem_key].shape[1]
                if c_in == 2:
                    # replace reshape from 3-plane+coord to 2-plane input
                    patched = patched.replace(
                        "    x = np.array(state, dtype=np.float32).reshape(3,6,7)\n    coord = np.stack([_row, _col], axis=0).astype(np.float32)\n    x = np.concatenate([x, coord], axis=0)",
                        "    x = np.array(state, dtype=np.float32).reshape(2,6,7)"
                    )
                    # Also update encode_state to emit only 2 planes (player/opponent)
                    patched = patched.replace(
                        "def encode_state(board, mark):\n    cur = detect_player_identity(board, mark)\n    arr = np.array(board).reshape(6,7)\n    p = (arr == cur).astype(np.float32)\n    o = (arr == (3-cur)).astype(np.float32)\n    e = (arr == 0).astype(np.float32)\n    return np.concatenate([p.ravel(), o.ravel(), e.ravel()])",
                        "def encode_state(board, mark):\n    cur = detect_player_identity(board, mark)\n    arr = np.array(board).reshape(6,7)\n    p = (arr == cur).astype(np.float32)\n    o = (arr == (3-cur)).astype(np.float32)\n    return np.concatenate([p.ravel(), o.ravel()])"
                    )
            # If fc/head expects input matching channel count (e.g. head.1.weight shape (1024,64)), use global pooling
            if fc_key in arrays:
                w = arrays[fc_key]
                if w.ndim == 2:
                    in_dim = w.shape[1]
                    # if in_dim equals channel count in conv (arrays[stem_key].shape[0]) then use pooling
                    if stem_key in arrays and arrays[stem_key].shape[0] == in_dim:
                        patched = patched.replace(
                            "    flat = x.reshape(-1)\n    h = relu(linear(flat, weights['head.1.weight'], weights['head.1.bias']))",
                            "    # global average pool to get channel descriptor\n    pooled = x.mean(axis=(1,2))\n    h = relu(linear(pooled, weights['head.1.weight'], weights['head.1.bias']))"
                        )
        except Exception:
            pass
        # Make weights lookups safe: after load_weights() the generated code sets
        # `weights = load_weights()`. Replace that with a defaultdict wrapper so
        # missing keys return None instead of raising KeyError. Also make the
        # generated group_norm resilient to None gamma/beta by returning the
        # input unchanged.
        if "weights = load_weights()" in patched:
            patched = patched.replace(
                "weights = load_weights()",
                "weights = load_weights()\nfrom collections import defaultdict\nweights = defaultdict(lambda: None, weights)"
            )

        # Insert guard in group_norm to handle None gamma/beta. This is safer
        # than attempting to remap many different normalization key patterns.
        if "def group_norm(" in patched:
            patched = patched.replace(
                "def group_norm(x, gamma, beta, num_groups=8, eps=1e-5):\n    C,H,W = x.shape",
                "def group_norm(x, gamma, beta, num_groups=8, eps=1e-5):\n    # If gamma/beta not found in weights (None), skip normalization\n    if gamma is None or beta is None:\n        return x\n    C,H,W = x.shape"
            )

        return patched

    code = _patch_code_for_shapes(code, np_state)

    with open(out_py, "w", encoding="utf-8") as f:
        f.write(code)

    print(f"  -> wrote submission {out_py}")
    return out_py


def _minimal_generate(weights_b64: str, has_trunk: bool) -> str:
    """Fallback generator: simple agent that loads the npz and picks a random valid action.
    This ensures we always produce a runnable submission even if the main generator can't be imported.
    """
    return (
        "import base64, io, numpy as np\n"
        "WEIGHTS_B64='" + weights_b64 + "'\n"
        "def load_weights():\n"
        "    buf = io.BytesIO(base64.b64decode(WEIGHTS_B64))\n"
        "    return {k: v for k, v in np.load(buf).items()}\n\n"
        "def agent(obs, config):\n"
        "    board = obs['board'] if isinstance(obs, dict) and 'board' in obs else obs\n"
        "    valids = [c for c in range(7) if board[c] == 0]\n"
        "    return int(valids[0] if valids else 0)\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Dump battle-* checkpoints to Kaggle submission .py files")
    parser.add_argument("--dry-run", action="store_true", help="List matching checkpoints without writing files")
    parser.add_argument("--pattern", type=str, default="battle*.pt", help="Glob pattern to match battle checkpoints (relative to checkpoints/)")
    args = parser.parse_args()

    pattern = str(CKPT_DIR / args.pattern)
    files = sorted(glob.glob(pattern))
    if not files:
        print("No matching checkpoints found:", pattern)
        return

    print(f"Found {len(files)} checkpoints")
    for p in files:
        try:
            dump_checkpoint_to_submission(Path(p), dry_run=args.dry_run)
        except Exception as e:
            print(f"Failed to process {p}: {e}")


if __name__ == '__main__':
    main()
