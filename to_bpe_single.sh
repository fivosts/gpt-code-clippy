#!/bin/bash

#SBATCH --time=4320
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --partition=devlab
#SBATCH --mem-per-cpu=10G

input_dir=$1

python code_bpe_encoder.py \
    --merge-file /checkpoint/dpf/data/tokenizers/github-py-redacted+so_psno-True/merges.txt \
    --vocab-file /checkpoint/dpf/data/tokenizers/github-py-redacted+so_psno-True/vocab.json \
    --pretokenizer-split-newlines-only \
    --input-dirs ${input_dir} \
    --output-dir /checkpoint/dpf/data/processed_filenames_redact \
    --use-hf-tokenizer \
    --splits-dir /checkpoint/dpf/data/splits/ \
    --metadata dstars source extension filename \
    --workers 40
