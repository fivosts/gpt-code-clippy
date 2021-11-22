import argparse
import datasets
import lm_dataformat
import re
from typing import Set, Tuple
import hashlib
import os.path
import pickle

TOKEN_RE = re.compile(r"\W+")

def get_signatures(examples, unique_signatures_: Set[Tuple[str, bytes, int]]):
    """Convert a batch of code string to a list of tokens, then return a hash signature for the token list."""
    """signature: file extension, md5 digest of the joined tokens, and number of tokens"""
    all_tokens = [TOKEN_RE.split(text) for text in examples["text"]]
    extensions = [os.path.splitext(file_name)[1] for file_name in examples["file_name"]]
    assert len(all_tokens) == len(extensions)
    signatures = [
        # using the number of tokens as well as the md5 is probably overkill but will help avoid any collisions
        (extension, hashlib.md5(' '.join(tokens)).digest(), len(tokens)) 
        for tokens, extension in zip(all_tokens, extensions)
    ]
    unique_signatures_.update(signatures)
    return {"signature": signatures}

def check_uniques(example, unique_signatures_: Set[Tuple[str, bytes, int]]):
    """If an example is in uniques, return True and remove it from uniques."""
    signature = example["signatures"]
    if signature in unique_signatures_:
        unique_signatures_.remove(signature)
        return True
    else:
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deduplicate a list of files")
    parser.add_argument("--data_dirs", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--signature_file", type=str, required=True)
    parser.add_argument("--archive_commit_freq", type=int, default=10_000)
    args = parser.parse_args()

    def write_signature_file(unique_signatures):
        print(f"writing to signature file {args.signature_file}")
        with open(args.signature_file, 'wb') as f:
            pickle.dump(unique_signatures, f)

    if os.path.exists(args.signature_file):
        print(f"loading existing signature file from {args.signature_file}")
        with open(args.signature_file, 'rb') as f:
            unique_signatures = pickle.load(f)
    else:
        unique_signatures = set()
    print(f"initial number of signatures: {len(unique_signatures)}")

    for data_dir in args.data_dirs:
        dataset = datasets.load_dataset("script.py", data_dir=args.data_dir, split="train")
        print(f": {len(dataset)}")

        unique_in_this = set()
        dataset = dataset.map(get_signatures, batched=True, fn_kwargs={"unique_signatures_": unique_in_this})
        unique_signatures_updated = unique_in_this | unique_signatures

        n_marginal_unique = len(unique_signatures_updated) - len(unique_signatures)

        print(f"{len(unique_in_this)} / {len(dataset)} in this dataset unique to the dataset")
        print(f"{n_marginal_unique} / {len(dataset)} in this dataset unique overall")
        print(f"{len(unique_signatures)} unique signatures total")

        # new_unique_signatures will be modified in place
        unique_signatures = unique_signatures_updated.copy()

        deduplicated_dataset = dataset.filter(check_uniques, fn_kwargs={"unique_signatures_": unique_signatures_updated})

        assert len(deduplicated_dataset) == n_marginal_unique

        ar = lm_dataformat.Archive(args.output_dir)

        for i, example in enumerate(deduplicated_dataset):
            code = example["text"]
            del example["text"]
            del example["signature"]
            ar.add_data(code, meta=example)
            if i > 0 and i % args.archive_commit_freq == 0:
                ar.commit()
        ar.commit()
        write_signature_file(unique_signatures)