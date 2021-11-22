import sys
import argparse
import datasets
import lm_dataformat
import re
from typing import Set, Tuple
import hashlib
import os.path
import pickle

TOKEN_RE = re.compile(r"\W+")

def get_signatures(examples):
    """Convert a batch of code string to a list of tokens, then return a hash signature for the token list."""
    """signature: file extension, md5 digest of the joined tokens, and number of tokens"""
    assert len(examples["text"]) == len(examples["file_name"]), f"{len(examples['text'])} != {len(examples['file_name'])}"
    all_tokens = [TOKEN_RE.split(text) for text in examples["text"]]
    extensions = [os.path.splitext(file_name)[1] for file_name in examples["file_name"]]
    assert len(all_tokens) == len(extensions), f"{len(all_tokens)} != {len(extensions)}"
    signatures = [
        # using the number of tokens as well as the md5 is probably overkill but will help avoid any collisions
        f"{extension}_{hashlib.md5(' '.join(tokens).encode()).hexdigest()}_{len(tokens)}" 
        for tokens, extension in zip(all_tokens, extensions)
    ]
    return {"signature": signatures}

def check_uniques(example, unique_signatures_: Set[Tuple[str, bytes, int]]):
    """If an example is in uniques, return True and remove it from uniques."""
    signature = example["signature"]
    if signature in unique_signatures_:
        unique_signatures_.remove(signature)
        return True
    else:
        return False

def strip_trailing_slash(path):
    while path[-1] == '/':
        path = path[:-1]
    return path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deduplicate a list of files")
    parser.add_argument("--data_dirs", nargs="*")
    parser.add_argument("--signature_file", type=str, required=True)
    parser.add_argument("--archive_commit_freq", type=int, default=10_000)
    parser.add_argument("--num_proc", type=int, default=10)
    parser.add_argument("--source", default="github")
    parser.add_argument("--output_dir_exist_ok", action="store_true")
    args = parser.parse_args()
    print(' '.join(sys.argv))

    def write_signature_file(unique_signatures, processed_data_dirs):
        print(f"writing to signature file {args.signature_file}")
        with open(args.signature_file, 'wb') as f:
            pickle.dump({
                'processed_data_dirs': processed_data_dirs,
                'unique_signatures': unique_signatures,
            }, f)

    if os.path.exists(args.signature_file):
        print(f"loading existing signature file from {args.signature_file}")
        with open(args.signature_file, 'rb') as f:
            d = pickle.load(f)
            unique_signatures, processed_data_dirs = d['unique_signatures'], d['processed_data_dirs']
    else:
        unique_signatures = set()
        processed_data_dirs = []
    print(f"initial number of signatures: {len(unique_signatures)}")
    if processed_data_dirs:
        print(f"previously procssed data_dirs:")
        print('\n'.join(processed_data_dirs))

    for data_dir in args.data_dirs:
        data_dir = strip_trailing_slash(data_dir)

        output_dir = data_dir + "_dedup"
        print(f"deduplicating to directory {output_dir}")
        os.makedirs(output_dir, exist_ok=args.output_dir_exist_ok)

        dataset = datasets.load_dataset("code_clippy_dataset", data_dir=data_dir, split="train", source=args.source)

        dataset = dataset.map(get_signatures, batched=True, num_proc=args.num_proc, desc='computing signatures')
        unique_in_this = set(dataset.unique('signature'))
        marginal_unique_in_this = unique_in_this - unique_signatures

        unique_signatures_updated = unique_in_this | unique_signatures
        n_marginal_unique = len(unique_signatures_updated) - len(unique_signatures)

        assert n_marginal_unique == len(marginal_unique_in_this)

        print(f"{len(unique_signatures)} unique signatures, previously")
        print(f"{len(unique_in_this)} / {len(dataset)} = {len(unique_in_this)/len(dataset)*100:.2f}% in this dataset are unique to this dataset")
        print(f"{n_marginal_unique} / {len(dataset)} = {n_marginal_unique/len(dataset)*100:.2f}% in this dataset are unique overall")
        print(f"{len(unique_signatures)} unique signatures, including this dataset")

        unique_signatures = unique_signatures_updated

        deduplicated_dataset = dataset.filter(check_uniques, fn_kwargs={"unique_signatures_": marginal_unique_in_this})

        assert len(deduplicated_dataset) == n_marginal_unique
        print(f"{len(dataset)} entries before deduplication; read from {data_dir}")
        print(f"{len(deduplicated_dataset)} entries before deduplication; writing to {output_dir}")

        ar = lm_dataformat.Archive(output_dir)

        for i, example in enumerate(deduplicated_dataset):
            code = example["text"]
            del example["text"]
            del example["signature"]
            ar.add_data(code, meta=example)
            if i > 0 and i % args.archive_commit_freq == 0:
                ar.commit()
        ar.commit()
        processed_data_dirs.append(data_dir)
        write_signature_file(unique_signatures, processed_data_dirs)
