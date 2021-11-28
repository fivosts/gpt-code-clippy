import tqdm
import re
import sys
import pprint
import csv
import argparse
from os.path import splitext
from typing import Set, Tuple, List
import itertools
import os

import functools

import numpy as np
from datasets import load_dataset

from code_clippy_dataset.utils import strip_trailing_slash, infer_source_from_data_dir

NON_ALPHA_NUM_RE = re.compile(r"\W+")
NEWLINE_RE = re.compile(r"[\r\n]+")

UNLIMITED_LINE_LENGTH_EXTENSIONS = {'.ipynb'}


def keep_record(record,
                minimum_word_char_fraction: float = None,
                maximum_line_length: float = None,
                maximum_average_line_length: float = None,
                strings_to_exclude: List[str] = None,
                repos_and_basenames_to_exclude: List[Tuple[str, str]] = None,
                ):
    text = record["text"]
    extension = splitext(record["file_name"])[-1]
    basename = record["file_name"].split('/')[-1]
    if repos_and_basenames_to_exclude is not None and (record["repo_name"], basename) in repos_and_basenames_to_exclude:
        # tqdm.tqdm.write(f"excluding {(record['repo_name'], basename)}")
        return False
    if (maximum_line_length is not None or maximum_average_line_length is not None) and extension not in UNLIMITED_LINE_LENGTH_EXTENSIONS:
        line_lengths = np.array([len(l) for l in NEWLINE_RE.split(text)])
        if maximum_line_length is not None and line_lengths.max() > maximum_line_length:
            return False
        if maximum_average_line_length is not None and line_lengths.mean() > maximum_average_line_length:
            return False
    if minimum_word_char_fraction is not None:
        assert 0 <= minimum_word_char_fraction <= 1.0
        num_word_chars = len(re.sub(NON_ALPHA_NUM_RE, "", text))
        word_char_percentage = float(num_word_chars) / len(text)
        if word_char_percentage < minimum_word_char_fraction:
            return False
    if strings_to_exclude is not None:
        for str_to_exclude in strings_to_exclude:
            if str_to_exclude in text:
                return False
    return True

if __name__ == "__main__":
    print(' '.join(sys.argv))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dirs",
        required=True,
        nargs='+',
    )
    parser.add_argument(
        "--maximum_average_line_length",
        type=float,
        default=100,
        help="exclude files that have more than this number of characters on average per line",
    )
    parser.add_argument(
        "--maximum_line_length",
        type=int,
        default=1000,
        help="exclude files that have more than this number of characters in any single line",
    )
    parser.add_argument(
        "--minimum_word_char_fraction",
        type=float,
        default=0.5,
        help="exclude files that have fewer than this fraction of word characters (alphanumeric and _)"
    )
    parser.add_argument(
        "--repos_and_basenames_to_exclude_paths",
        nargs='+',
    )
    parser.add_argument(
        "--strings_to_exclude",
        nargs='*',
        default=["GNU General Public License"],
    )
    parser.add_argument("--source")
    parser.add_argument("--num_proc", type=int, default=20)

    args = parser.parse_args()
    pprint.pprint(vars(args))

    for data_dir in args.data_dirs:
        data_dir = strip_trailing_slash(data_dir)

        if args.source is None:
            source = infer_source_from_data_dir(data_dir)
            print(f"inferred source {source} from path {data_dir}")
        else:
            source = args.source

        dataset = load_dataset("code_clippy_dataset", source=source, data_dir=data_dir)['train']
        len_before_filtering = len(dataset)

        if args.repos_and_basenames_to_exclude_paths:
            repos_and_basenames_to_exclude = set()
            for fname in args.repos_and_basenames_to_exclude_paths:
                with open(fname) as f:
                    this_repos_and_basenames_to_exclude = set()
                    has_extension = False
                    # sanity check the ordering: repo names should be in format name/repo; some filenames should have an extension
                    for repo_name, path in csv.reader(f):
                        assert len(repo_name.split('/')) == 2, repo_name
                        basename = os.path.basename(path)
                        if '.' in basename:
                            has_extension = True
                        this_repos_and_basenames_to_exclude.add((repo_name, basename))
                    assert has_extension
                    repos_and_basenames_to_exclude.update(this_repos_and_basenames_to_exclude)
        else:
            repos_and_basenames_to_exclude = None

        predicate = functools.partial(keep_record, 
            minimum_word_char_fraction=args.minimum_word_char_fraction,
            maximum_line_length=args.maximum_line_length,
            maximum_average_line_length=args.maximum_average_line_length,
            strings_to_exclude=args.strings_to_exclude,
            repos_and_basenames_to_exclude=repos_and_basenames_to_exclude,
        )

        # dataset_filtered = filter(
        #     dataset,
        #     minimum_word_char_fraction=args.minimum_word_char_fraction,
        #     maximum_line_length=args.maximum_line_length,
        #     maximum_average_line_length=args.maximum_average_line_length,
        #     strings_to_exclude=args.strings_to_exclude,
        #     repos_and_basenames_to_exclude=repos_and_basenames_to_exclude,
        #     num_proc=args.num_proc,
        # )
        dataset_filtered = dataset.filter(predicate, num_proc=args.num_proc)

        print(f"retained {len(dataset_filtered)} / {len(dataset)} records ({len(dataset_filtered)/len(dataset)*100:.2f}%)")
        output_dir = data_dir + "_filtered"
        print(f"deduplicating to directory {output_dir}")
        dataset_filtered.save_to_disk(output_dir)