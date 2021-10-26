from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
import random

# subsample the dataset so we don't get OOM
YIELD_RATE = 0.02

if __name__ == "__main__":
    import os
    data_dir = "scrapes/out_python_forkless_open-source_10-9/github_data_dedup"

    tokenizer_dir = f"{data_dir}/tokenizer_preserve_newlines"
    os.makedirs(tokenizer_dir, exist_ok=True)
    
    # datasets are hashed based on arguments, which are sensitive to trailing dashes (even though loading is not)
    data_dir = data_dir.rstrip("/")
    print(data_dir)
    data = load_dataset("code_clippy_dataset", data_dir=data_dir)['train']

    tokenizer = ByteLevelBPETokenizer()

    def data_gen():
        for x in data:
            if random.random() <= YIELD_RATE:
                yield x['text']

    tokenizer.train_from_iterator(data_gen(), vocab_size=50257, special_tokens=["<|endoftext|>", "<|pad|>"])
    tokenizer.save_model(tokenizer_dir)
