#!/bin/bash

#root_dir="/private/home/dpf/data/github/out_python_forkless_open-source_10-9/github_data_dedup/tokenized_ours"
root_dir="/private/home/dpf/data/github/out_python_forkless_open-source_10-9/github_data_dedup/tokenized_gpt2"
in_dir="${root_dir}/raw"
out_dir="${root_dir}/bin"
prefix="combined"

mkdir ${out_dir}

python examples/code/split_and_concat_corpora.py $in_dir $prefix

fairseq-preprocess \
    --only-source \
    --trainpref ${in_dir}/${prefix}.train.bpe \
    --validpref ${in_dir}/${prefix}.valid.bpe \
    --testpref ${in_dir}/${prefix}.test.bpe \
    --destdir ${out_dir} \
    --workers 60
