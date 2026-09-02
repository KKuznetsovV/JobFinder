"""Quantize the fine-tuned tier-1 model (LoRA adapter merged into the base
model) to a GGUF file suitable for llama-cpp-python serving.

Run:
    python scripts/tier1/quantize.py --config scripts/tier1/config.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("scripts/tier1/config.yaml"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["output"]["adapter_dir"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=True,
    )

    gguf_dir = Path(cfg["output"]["gguf_dir"])
    gguf_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_gguf(
        str(gguf_dir / cfg["output"]["gguf_name"]),
        tokenizer,
        quantization_method=cfg["output"]["quantization_method"],
    )
    print(f"GGUF model written under {gguf_dir}")


if __name__ == "__main__":
    main()
