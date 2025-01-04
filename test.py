from transformers import GPT2LMHeadModel, GPT2Config, GPT2TokenizerFast, Trainer, TrainingArguments, TextDataset
from datasets import load_dataset
from functools import partial
import torch

class DataCollator:
    def __init__(self, tokenizer: GPT2TokenizerFast):
        self.tokenizer: GPT2TokenizerFast = tokenizer

    def __call__(self, features):
        batch = {}
        # Padding
        input_ids = [feature["input_ids"] for feature in features]
        max_len = max(len(x) for x in input_ids)
        input_ids = [x + [tokenizer.pad_token_id] * (max_len - len(x)) for x in input_ids]
        batch["input_ids"] = torch.tensor(input_ids)
        attention_mask = [feature["attention_mask"] for feature in features]
        max_len = max(len(x) for x in attention_mask)
        attention_mask = [x + [0] * (max_len - len(x)) for x in attention_mask]
        batch["attention_mask"] = torch.tensor(attention_mask)
        labels = [feature["labels"] for feature in features]
        max_len = max(len(x) for x in labels)
        labels = [x + [-100] * (max_len - len(x)) for x in labels]
        batch["labels"] = torch.tensor(labels)
        return batch

def preprocess(batch):
    result = tokenizer(batch["text"], truncation=True)
    batch["input_ids"] = result.input_ids
    batch["attention_mask"] = result.attention_mask
    batch["labels"] = result.input_ids
    return batch



if __name__ == '__main__':
    wikitext = load_dataset("wikitext", "wikitext-2-raw-v1")

    model = GPT2LMHeadModel(
        GPT2Config(n_layer=3, n_head=3, n_positions=512, n_embd=96)
    )

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = model.config.n_positions
    wikitext = wikitext.filter(lambda x: bool(x["text"]))  # Only if not empty
    wikitext = wikitext.map(preprocess, batched=True)
    print(wikitext)
    print(wikitext["train"][0])
    args = TrainingArguments("test/", eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=wikitext["train"],
        eval_dataset=wikitext["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model("final/")