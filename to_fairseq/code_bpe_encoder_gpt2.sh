#!/bin/bash

vocab_file="/private/home/halilakin/src/gpt2_bpe/encoder.json"
merge_file="/private/home/halilakin/src/gpt2_bpe/vocab.bpe"

input_dir="/private/home/dpf/data/github/out_python_forkless_open-source_10-9/github_data_dedup/tokenized_gpt2"
output_dir="/private/home/dpf/data/github/out_python_forkless_open-source_10-9/github_data_dedup/tokenized_gpt2/raw/"

mkdir $output_dir

input_suffix=".jsonl.zst"

for input_file in "${input_dir}"/*.jsonl.zst; do
  output_file=${output_dir}/`basename ${input_file%.jsonl.zst}`
  echo $input_file
  echo $output_file
  python examples/code/code_bpe_encoder.py \
          --vocab-file $vocab_file \
          --merge-file $merge_file \
          --input $input_file \
          --output $output_file \
          --add-file-separator \
          --maximum-average-line-length 100 \
          --minimum-word-char-percentage 0.5
done
