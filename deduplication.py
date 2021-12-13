import sys
import argparse
import datasets
import lm_dataformat
from typing import Set, Tuple
import hashlib
import os.path
import pickle
import code_clippy_dataset.utils

from code_clippy_dataset.utils import TOKEN_RE, load_dataset_infer, strip_trailing_slash


def get_signatures(examples):
    """Convert a batch of code string to a list of tokens, then return a hash signature for the token list."""
    """signature: file extension, md5 digest of the joined tokens, and number of tokens"""
    assert len(examples["text"]) == len(examples["file_name"]), f"{len(examples['text'])} != {len(examples['file_name'])}"
    all_tokens = [TOKEN_RE.split(text) for text in examples["text"]]
    extensions = [os.path.splitext(file_name)[1] for file_name in examples["file_name"]]
    repos_and_filenames = [f"{r}:{f}" for r, f in zip(examples["repo_name"], examples["file_name"])]
    assert len(all_tokens) == len(extensions), f"{len(all_tokens)} != {len(extensions)}"
    signatures = [
        # using the number of tokens as well as the md5 is probably overkill but will help avoid any collisions
        f"{extension}_{hashlib.md5(' '.join(tokens).encode()).hexdigest()}_{len(tokens)}" 
        for tokens, extension in zip(all_tokens, extensions)
    ]
    return {"signature": signatures, "repo_and_filename": repos_and_filenames}

def check_uniques(example, unique_signatures_: Set[str], unique_repos_and_filenames_: Set[str], exclude_duplicate_repos_and_filenames: bool = False, repos_to_exclude=None, 
                  files_excluded_because_of_repos_=None, files_excluded_because_of_repos_and_filenames_=None,
                 files_excluded_because_of_sigs_=None):
    """Have we seen this example before? Updates unique_signatures_ and unique_repos_and_filesnames"""
    signature = example["signature"]
    repo_and_filename = example["repo_and_filename"]
    if repos_to_exclude is not None and example["repo_name"] in repos_to_exclude:
        # don't add the signatures because we're not going to use it
        if files_excluded_because_of_repos_ is not None:
            files_excluded_because_of_repos_.append(repo_and_filename)
        return False
    found = False
    if signature in unique_signatures_:
        found = True
        if files_excluded_because_of_sigs_ is not None:
            files_excluded_because_of_sigs_.append((repo_and_filename, signature))
    if exclude_duplicate_repos_and_filenames and (repo_and_filename in unique_repos_and_filenames_):
        found = True
        if files_excluded_because_of_repos_and_filenames_ is not None:
            files_excluded_because_of_repos_and_filenames_.append(repo_and_filename)
    if not found:
        unique_signatures_.add(signature)
        unique_repos_and_filenames_.add(repo_and_filename)
    return not found

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deduplicate a list of files")
    parser.add_argument("--data_dirs", nargs="*")
    parser.add_argument("--signature_file", type=str, required=True)
    parser.add_argument("--archive_commit_freq", type=int, default=10_000)
    parser.add_argument("--num_proc", type=int, default=10)
    # parser.add_argument("--source", choices=['github', 'google_code', 'bitbucket', 'gitlab'])
    parser.add_argument("--output_dir_exist_ok", action="store_true")
    #parser.add_argument("--save_format", choices=["lm_archive", "huggingface"], default='huggingface')
    parser.add_argument("--exclude_duplicate_repos_and_filenames", action="store_true")
    parser.add_argument("--repo_exclude_lists", nargs="*")
    parser.add_argument("--infer_repo_exclude_lists", action='store_true', help='rather than using --repo_exclude_lists, load files from "repos_processed.txt" in the directory above each processed_data_dir from previously processed data dirs in the signatures file. This will exclude any repos that have already been processed but were double-included')
    parser.add_argument("--debug", action='store_true', help='skip write and drop into a pdb shell')
    args = parser.parse_args()
    print(' '.join(sys.argv))

    repos_to_exclude = set()

    def augment_repos_to_exclude_from_file(fname):
        with open(fname, 'r') as f:
            repos_to_exclude.update(line.strip() for line in f)

    if args.repo_exclude_lists:
        for fname in args.repo_exclude_lists:
            augment_repos_to_exclude_from_file(fname)

    def write_signature_file(unique_signatures, processed_data_dirs, repos_and_basenames):
        print(f"writing to signature file {args.signature_file}")
        with open(args.signature_file, 'wb') as f:
            pickle.dump({
                'processed_data_dirs': processed_data_dirs,
                'unique_signatures': unique_signatures,
                'repos_and_basenames': repos_and_basenames,
            }, f)

    if os.path.exists(args.signature_file):
        print(f"loading existing signature file from {args.signature_file}")
        with open(args.signature_file, 'rb') as f:
            d = pickle.load(f)
            unique_signatures, processed_data_dirs = d['unique_signatures'], d['processed_data_dirs']
            if 'repos_and_basenames' in d:
                unique_repos_and_basenames = d['repos_and_basenames']
            else:
                unique_repos_and_basenames = set()
            if args.infer_repo_exclude_lists:
                assert not args.repo_exclude_lists
                for path in processed_data_dirs:
                    fname = os.path.join(path.rstrip("/").rstrip("data"), "repos_processed.txt")
                    augment_repos_to_exclude_from_file(fname)
    else:
        unique_signatures = set()
        processed_data_dirs = []
        unique_repos_and_basenames = set()

    print(f"initial number of signatures: {len(unique_signatures)}")

    if processed_data_dirs:
        print(f"previously procssed data_dirs:")
        print('\n'.join(processed_data_dirs))

    print(f"will exclude any repos from set with {len(repos_to_exclude)} repos")

    for data_dir in args.data_dirs:
        data_dir = strip_trailing_slash(data_dir)

        output_dir = data_dir + "_dedup"
        print(f"deduplicating to directory {output_dir}")
        os.makedirs(output_dir, exist_ok=(args.output_dir_exist_ok or args.debug))

        print(f"loading {data_dir}")
        dataset = load_dataset_infer(data_dir)

        print(f"mapping")
        dataset = dataset.map(get_signatures, batched=True, num_proc=args.num_proc, desc='computing signatures')
        # unique_in_this = set(dataset.unique('signature'))
        # unique_randb_in_this = set(dataset.unique('repo_and_filename'))

        # marginal_unique_in_this = unique_in_this - unique_signatures
        # marginal_unique_randb_in_this = unique_randb_in_this - unique_repos_and_basenames

        # unique_signatures_updated = unique_in_this | unique_signatures
        # unique_repos_and_basenames_updated = unique_randb_in_this | unique_repos_and_basenames

        # n_marginal_unique = len(unique_signatures_updated) - len(unique_signatures)

        # assert n_marginal_unique == len(marginal_unique_in_this)

        print(f"{len(unique_signatures):_} unique signatures, previously")
        print(f"{len(unique_repos_and_basenames):_} unique (repo_name, base_filename), previously")

        files_excluded_because_of_repos = []
        files_excluded_because_of_sigs = []
        files_excluded_because_of_repos_and_filenames = []

        deduplicated_dataset = dataset.filter(check_uniques, 
                                              fn_kwargs={"unique_signatures_": unique_signatures,
                                                         "unique_repos_and_filenames_": unique_repos_and_basenames,
                                                         "exclude_duplicate_repos_and_filenames": args.exclude_duplicate_repos_and_filenames,
                                                         "repos_to_exclude": repos_to_exclude,
                                                         "files_excluded_because_of_repos_": files_excluded_because_of_repos,
                                                         "files_excluded_because_of_sigs_": files_excluded_because_of_sigs,
                                                         "files_excluded_because_of_repos_and_filenames_": files_excluded_because_of_repos_and_filenames,
                                                        })

        print(f"{len(unique_signatures):_} unique signatures, afterward")
        print(f"{len(unique_repos_and_basenames):_} unique (repo_name, base_filename), afterward")

        # assert len(deduplicated_dataset) == n_marginal_unique
        print(f"{len(dataset):_} entries before deduplication; read from {data_dir}")
        print(f"{len(deduplicated_dataset):_} entries ({len(deduplicated_dataset)/len(dataset)*100:.2f}%) after deduplication; writing to {output_dir}")
        print(f"excluded {len(files_excluded_because_of_sigs):_} files because their signatures matched previously found signatures")
        print(f"excluded {len(files_excluded_because_of_repos):_} files because their repos were in repo exclusion lists")
        print(f"excluded {len(files_excluded_because_of_repos_and_filenames):_} files because their repos and filenames were in repo and filename exclusion lists")

        # now that we are preprocssing the jupyter notebooks, force saving to a HF dataset
        # if args.save_format == 'lm_archive':
        #     ar = lm_dataformat.Archive(output_dir)

        #     for i, example in enumerate(deduplicated_dataset):
        #         code = example["text"]
        #         del example["text"]
        #         del example["signature"]
        #         del example["repo_and_filename"]
        #         ar.add_data(code, meta=example)
        #         if i > 0 and i % args.archive_commit_freq == 0:
        #             ar.commit()
        #     ar.commit()
        # else:
        deduplicated_dataset.remove_columns(["signature", "repo_and_filename"])
        processed_data_dirs.append(data_dir)
        if args.debug:
            from IPython import embed
            embed()
        else:
            print("writing signature file")
            deduplicated_dataset.save_to_disk(output_dir)
            write_signature_file(unique_signatures, processed_data_dirs, unique_repos_and_basenames)
