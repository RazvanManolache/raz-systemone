"""Upload a finetuned checkpoint to the Hugging Face Hub.

Auth: `hf auth login` once, or set HF_TOKEN. The token never touches disk here.

Example:
  training/.venv/Scripts/python.exe training/upload.py \
      --checkpoint training/runs/v5 --card training/model_cards/v5-xsmall.md \
      --repo RazvanManolache/raz-systemone-nli-xsmall
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import create_repo, upload_folder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="train.py --out dir")
    ap.add_argument("--card", required=True, help="model card markdown -> README.md")
    ap.add_argument("--repo", required=True, help="e.g. user/raz-systemone-nli-xsmall")
    ap.add_argument("--private", action="store_true", help="create a private repo")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    card = Path(args.card)
    for p in (ckpt / "model.safetensors", ckpt / "config.json",
              ckpt / "tokenizer.json", card):
        if not p.is_file():
            sys.exit(f"missing required file: {p}")

    create_repo(args.repo, exist_ok=True, private=args.private)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for f in ("model.safetensors", "config.json", "tokenizer.json",
                  "tokenizer_config.json"):
            src = ckpt / f
            if src.is_file():
                shutil.copy(src, stage / f)
        shutil.copy(card, stage / "README.md")
        upload_folder(folder_path=stage, repo_id=args.repo,
                      allow_patterns=["*.safetensors", "*.json", "*.md"])
    print(f"uploaded: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
