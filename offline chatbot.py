from transformers import GPT2TokenizerFast
import torch
from torch.utils.data import Dataset

tokenizer = GPT2TokenizerFast.from_pretrained("./hf_models/gpt2-small", local_files_only=True)
# Add special tokens if needed
tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # if you want padding

max_len = 512  # depending on GPU/CPU memory and model; GPT-2 small default max is 1024

class ConvDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        # Build a single string: input + target (we'll let model see both, but mask context tokens)
        input_text = ex['input'] + " " + ex['target']
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()  # shape (max_len,)
        attention_mask = encoding['attention_mask'].squeeze()
        # Create labels: mask context tokens with -100 so loss computed only on target
        # Determine split point: length of tokenized input for input part only
        input_part_enc = self.tokenizer(ex['input'], truncation=True, max_length=self.max_len, return_tensors='pt')
        input_part_len = input_part_enc['input_ids'].size(1)
        labels = input_ids.clone()
        # tokens up to input_part_len are context -> mask
        labels[:input_part_len] = -100
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# create dataset
dataset = ConvDataset(examples, tokenizer, max_len=max_len)
