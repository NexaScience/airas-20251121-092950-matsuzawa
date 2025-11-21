# Code unchanged from previous valid version – included for completeness.
from __future__ import annotations

import re
from typing import Dict, List, Tuple

import datasets
import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer

CACHE_DIR = ".cache/"

__all__ = ["build_datasets", "DataCollatorCausalLM"]


def _extract_answer(ans_str: str) -> str:
    if "####" in ans_str:
        return ans_str.split("####")[-1].strip()
    m = re.findall(r"(-?\d+\.?\d*)", ans_str)
    if m:
        return m[-1]
    return ans_str.strip().split("\n")[-1].strip()


def _tokenise_example(ex: Dict[str, str], tokenizer: PreTrainedTokenizer, max_len: int) -> Dict:
    question: str = ex["question"]
    answer_raw: str = ex["answer"]
    answer_clean = _extract_answer(answer_raw)

    prompt = f"Question: {question}\nAnswer:"
    tok_prompt = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=max_len)
    remaining = max_len - len(tok_prompt["input_ids"])
    tok_answer = tokenizer(" " + answer_clean, add_special_tokens=False, truncation=True, max_length=remaining - 1)

    input_ids: List[int] = tok_prompt["input_ids"] + tok_answer["input_ids"] + [tokenizer.eos_token_id]
    attn: List[int] = [1] * len(input_ids)
    labels = [-100] * len(tok_prompt["input_ids"]) + tok_answer["input_ids"] + [tokenizer.eos_token_id]

    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "labels": labels,
        "prompt_length": len(tok_prompt["input_ids"]),
        "answer": answer_clean,
    }


def build_datasets(cfg, tokenizer: PreTrainedTokenizer) -> Tuple[Dataset, Dataset]:
    if cfg.dataset.name.lower() != "gsm8k":
        raise ValueError("Only GSM8K supported in this reference implementation")

    raw = datasets.load_dataset("openai/gsm8k", cfg.dataset.config, cache_dir=CACHE_DIR)
    ds_train = raw[cfg.dataset.splits.train]
    ds_val = raw[cfg.dataset.splits.validation]

    max_len = int(cfg.dataset.preprocessing.max_seq_length)
    ds_train = ds_train.map(lambda ex: _tokenise_example(ex, tokenizer, max_len), remove_columns=ds_train.column_names)
    ds_val = ds_val.map(lambda ex: _tokenise_example(ex, tokenizer, max_len), remove_columns=ds_val.column_names)

    if cfg.mode == "trial":
        ds_train = ds_train.select(range(min(32, len(ds_train))))
        ds_val = ds_val.select(range(min(32, len(ds_val))))

    return ds_train, ds_val


class DataCollatorCausalLM:
    def __init__(self, tokenizer: PreTrainedTokenizer, label_pad_token_id: int = -100):
        self.tok = tokenizer
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.label_pad = label_pad_token_id

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, attn, labels, p_lens, answers = [], [], [], [], []
        for ex in batch:
            pad_len = max_len - len(ex["input_ids"])
            input_ids.append(ex["input_ids"] + [self.pad_id] * pad_len)
            attn.append(ex["attention_mask"] + [0] * pad_len)
            labels.append(ex["labels"] + [self.label_pad] * pad_len)
            p_lens.append(ex["prompt_length"])
            answers.append(ex["answer"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompt_lengths": torch.tensor(p_lens, dtype=torch.long),
            "answers": answers,
        }