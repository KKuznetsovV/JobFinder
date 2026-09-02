"""Fine-tune the tier-1 local classifier: LoRA fine-tuning of a small base
model (Qwen2.5-1.5B-Instruct, 4-bit) via Unsloth + TRL's SFTTrainer, on the
train/val JSONL produced by split_dataset.py.

Mirrors the sibling SmartServe project's approach (same base model, same
LoRA hyperparameters, same prompt-template shape) - see scripts/tier1/config.yaml
for the exact hyperparameters, kept in one place so train.py and this
docstring can't drift out of sync with each other.

Requires the separate training environment (heavy deps not part of the main
app's requirements.txt): pip install -r scripts/tier1/requirements.txt
Needs a CUDA-capable GPU in practice - CPU training of even a 1.5B model is
impractically slow.

Run:
    python scripts/tier1/train.py --config scripts/tier1/config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from jobfinder.ai.tier1_schema import PROMPT_TEMPLATE  # noqa: E402


def _patch_dill_for_python314() -> None:
    """Python 3.14's stdlib pickle.Pickler.save_dict now calls
    self._batch_setitems(items, obj) (an extra `obj` arg, used only for
    error messages), but datasets.utils._dill.Pickler._batch_setitems still
    overrides the old 2-arg signature - breaking every datasets.load_dataset
    call with a TypeError. Remove this once `datasets` ships a fix upstream."""
    import dill
    from datasets.utils import _dill as datasets_dill

    if sys.version_info < (3, 14):
        return

    def _batch_setitems(self, items, obj=None):
        if self._legacy_no_dict_keys_sorting:
            return dill.Pickler._batch_setitems(self, items, obj)
        try:
            items = sorted(items)
        except Exception:
            from datasets.fingerprint import Hasher

            items = sorted(items, key=lambda x: Hasher.hash(x[0]))
        dill.Pickler._batch_setitems(self, items, obj)

    datasets_dill.Pickler._batch_setitems = _batch_setitems


_patch_dill_for_python314()

from datasets import load_dataset  # noqa: E402


def _format_example(example: dict) -> dict:
    return {
        "text": PROMPT_TEMPLATE.format(
            instruction=example["instruction"], input=example["input"], output=example["output"]
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("scripts/tier1/config.yaml"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
        use_gradient_checkpointing=True,
        random_state=cfg["seed"],
    )

    dataset = load_dataset(
        "json",
        data_files={"train": cfg["data"]["train_path"], "validation": cfg["data"]["val_path"]},
    )
    dataset = dataset.map(_format_example)

    training_args = SFTConfig(
        output_dir=cfg["output"]["adapter_dir"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        warmup_steps=cfg["training"]["warmup_steps"],
        num_train_epochs=cfg["training"]["num_epochs"],
        learning_rate=cfg["training"]["learning_rate"],
        eval_steps=cfg["training"]["eval_steps"],
        eval_strategy="steps",
        bf16=cfg["training"]["bf16"],
        seed=cfg["seed"],
        max_seq_length=cfg["max_seq_length"],
        dataset_text_field="text",
        logging_steps=10,
        save_strategy="epoch",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=training_args,
    )
    trainer.train()

    adapter_dir = Path(cfg["output"]["adapter_dir"])
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"LoRA adapter saved to {adapter_dir}")


if __name__ == "__main__":
    main()
