#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
import sys
from collections import Counter
from multiprocessing import Pool
import io
import re
import math

import numpy as np
import scipy
import scipy.stats

import jsonlines
import zstandard as zstd

from tokenizers import ByteLevelBPETokenizer

END_OF_TEXT = "<|endoftext|>"
PAD = "<|pad|>"

SPECIAL_TOKENS = [END_OF_TEXT, PAD]

NON_WORD_REGEX = re.compile(r"\W")

def word_chars(string):
    return re.sub(NON_WORD_REGEX, "", string)

DEFAULT_LANG_EXTS = {
    ".lisp",
    ".lsp",
    ".f",
    ".fs",
    ".sh",
    ".groovy",
    ".r",
    ".pl",
    ".html",
    ".css",
    ".sql",
    ".py",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".jl",
    ".java",
    ".js",
    ".ts",
    ".cs",
    ".go",
    ".rs",
    ".swift",
    ".php",
    ".dart",
    ".kt",
    ".m",
    ".hs",
    ".scala",
    ".sc",
    ".lua",
    ".rb",
}


def read_zst_file(decompressor, opened_file, lang_exts=DEFAULT_LANG_EXTS):
    # TODO filtering
    f = decompressor.stream_reader(opened_file)
    f = io.TextIOWrapper(f, encoding="utf-8")
    f = jsonlines.Reader(f)
    for record in f:
        filename = record["meta"]["file_name"]
        start = filename.rfind(".")
        if filename[start:] in lang_exts:
            yield record["text"]


def main():
    """
    Helper script to encode raw text with the GPT-2 BPE using multiple processes.

    The encoder.json and vocab.bpe files can be obtained here:
    - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
    - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--merge-file",
        help="path to merges.txt",
    )
    parser.add_argument(
        "--vocab-file",
        type=str,
        help="path to vocab.txt",
    )
    parser.add_argument(
        "--input",
        required=True,
        default="-",
        help="input file to filter/encode",
    )
    parser.add_argument(
        "--output",
        required=True,
        default="-",
        help="path to save encoded outputs",
    )
    # parser.add_argument(
    #     "--maximum-line-length",
    #     type=int,
    #     help="exclude files that have a single line with more than this number of tokens",
    # )
    parser.add_argument(
        "--maximum-average-line-length",
        type=int,
        help="exclude files that have more than this number of tokens on average per line",
    )
    parser.add_argument(
        "--minimum-word-char-percentage",
        type=float,
        help="exclude files that have fewer than this percentage of word characters (alphanumeric and _)"
    )
    parser.add_argument(
        "--add-file-separator",
        action='store_true',
        help=f"separate source files using the token for {END_OF_TEXT}",
    )
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    decompressor = zstd.ZstdDecompressor()

    with contextlib.ExitStack() as stack:
        file_input = (stack.enter_context(open(args.input, "rb")) if args.input != '-' else sys.stdin)
        input = read_zst_file(decompressor, file_input)
        output = stack.enter_context(open(args.output, "w", encoding="utf-8")) if args.output != "-" else sys.stdout

        encoder = MultiprocessingEncoder(args)
        pool = Pool(args.workers, initializer=encoder.initializer)
        encoded_lines = pool.imap(encoder.encode_file_string, input, 100)

        stats = Counter()
        word_char_fractions = []
        for i, (enc_lines, file_string, num_lines) in enumerate(encoded_lines, start=1):
            filt = None
            if len(file_string) == 0:
                filt = "empty_file"
            else:
                word_char_fraction = float(len(word_chars(file_string))) / len(file_string)
                word_char_fractions.append(word_char_fraction)
            avg_tokens_per_line = float(sum(len(line) for line in enc_lines)) / num_lines

            if args.minimum_word_char_percentage and word_char_fraction < args.minimum_word_char_percentage:
                filt = "word_char_fraction"
            # elif args.maximum_line_length is not None and any(num_tokens_per_line > args.maximum_line_length):
            #     filt = "max_line_length"
            elif args.maximum_average_line_length is not None and avg_tokens_per_line > args.maximum_average_line_length:
                filt = "max_average_line_length"

            if filt is None:
                for enc_line in enc_lines:
                    print(' '.join(map(str, enc_line)), file=output)
            else:
                stats["num_filtered_" + filt] += 1
            if i % 10000 == 0:
                print("processed {} lines".format(i), file=sys.stderr)

        for k, v in stats.most_common():
            print(f"[{k}] filtered {v} lines / {i} = {v * 100.0 / i:.2f}%", file=sys.stderr)

        print(f"word char fractions min: {np.min(word_char_fractions):.3f}", file=sys.stderr)
        print(f"word char fractions mean: {np.mean(word_char_fractions):.3f}", file=sys.stderr)
        print(f"word char fractions max: {np.max(word_char_fractions):.3f}", file=sys.stderr)

        # percentiles = np.arange(0, 100, 10)
        # percentile_scores = scipy.stats.scoreatpercentile(word_char_fractions, per=percentiles)
        # print("word char fractions percentiles:", file=sys.stderr)
        # print(list(zip(percentiles, percentile_scores)), file=sys.stderr)


class MultiprocessingEncoder(object):

    def __init__(self, args):
        self.args = args

    def initializer(self):
        global bpe
        bpe = ByteLevelBPETokenizer.from_file(
            self.args.vocab_file, self.args.merge_file,
            )
        bpe.add_special_tokens(SPECIAL_TOKENS)

    def encode_line(self, line):
        global bpe
        ids = bpe.encode(line).ids
        return ids

    def encode_file_string(self, file_string):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        # encode the entire file as one line so that we can properly handle merged new lines (eg \n\n as a single token)
        tokens_per_line = [self.encode_line(file_string)]
        num_lines = len(file_string.splitlines())
        if self.args.add_file_separator:
            tokens_per_line.append(self.encode_line(END_OF_TEXT))
        else:
            # want to separate by a blank document
            tokens_per_line.append([])
        return tokens_per_line, file_string, num_lines


if __name__ == "__main__":
    main()
