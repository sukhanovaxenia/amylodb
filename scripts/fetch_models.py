#!/usr/bin/env python3
"""Download the mining models into a local directory and pin their revisions.

For each model in `amylodb.mining.models.registry.MODELS` this resolves the
commit on the HuggingFace Hub, downloads that exact snapshot into
``<models-dir>/<key>/`` and records ``{repo_id, revision, path}`` in
``<models-dir>/models.lock.json``. A second run reuses the pinned revisions, so
every later mining run uses byte-identical weights until you pass ``--update``.

Then run mining with ``--models-dir <models-dir>`` (and HF_HUB_OFFLINE=1 if you
want to forbid any network access by the model stack).

Consumes: network access to huggingface.co (first run only).
Emits: model snapshots + models.lock.json.

Usage:
    python scripts/fetch_models.py --models-dir models
    python scripts/fetch_models.py --models-dir models --only nli structner
    python scripts/fetch_models.py --models-dir models --update      # move to latest revisions
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

# Weight formats we never need (keeps downloads small).
IGNORE = ["*.h5", "*.msgpack", "*.onnx", "*.ot", "flax_model*", "tf_model*", "*.tflite", "rust_model*"]


def main() -> int:
    from amylodb.mining.models.registry import LOCKFILE_NAME, MODELS, read_lock

    ap = argparse.ArgumentParser(description="Fetch and pin mining models")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--only", nargs="*", choices=sorted(MODELS), help="subset of models")
    ap.add_argument("--update", action="store_true", help="ignore pinned revisions, take latest")
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        print('huggingface_hub is required: pip install -e ".[mining]"', file=sys.stderr)
        return 1

    os.makedirs(args.models_dir, exist_ok=True)
    lock = read_lock(args.models_dir)
    api = HfApi()
    for key in (args.only or sorted(MODELS)):
        repo = MODELS[key]
        pinned = None if args.update else lock.get(key, {}).get("revision")
        info = api.model_info(repo, revision=pinned)
        target = os.path.join(args.models_dir, key)
        print(f"[{key}] {repo} @ {info.sha[:12]} -> {target}")
        path = snapshot_download(repo_id=repo, revision=info.sha, local_dir=target,
                                 ignore_patterns=IGNORE)
        lock[key] = {
            "repo_id": repo,
            "revision": info.sha,
            "path": os.path.abspath(path),
            "fetched_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        }

    # zshot's SMXM linker always loads the bert-large-cased tokenizer into its own
    # cache (MODELS_CACHE_PATH). Pre-fill it inside models_dir so OpenBioNER also
    # runs offline; amylodb points MODELS_CACHE_PATH there at run time.
    if "openbioner" in (args.only or MODELS):
        from transformers import BertTokenizerFast
        zcache = os.path.join(os.path.abspath(args.models_dir), "zshot_cache") + os.sep
        BertTokenizerFast.from_pretrained("bert-large-cased", truncation_side="left", cache_dir=zcache)
        lock["zshot_tokenizer"] = {"repo_id": "bert-large-cased", "path": zcache}
        print(f"[zshot] bert-large-cased tokenizer cached in {zcache}")

    with open(os.path.join(args.models_dir, LOCKFILE_NAME), "w", encoding="utf-8") as fh:
        json.dump(lock, fh, indent=2)
    print(f"Pinned {len(lock)} models in {os.path.join(args.models_dir, LOCKFILE_NAME)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
