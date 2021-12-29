#!/bin/bash

#SBATCH --time=4320
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --partition=devlab
#SBATCH --mem-per-cpu=10G

input_dir=$1

python code_bpe_encoder.py \
    --merge-file /private/home/halilakin/src/gpt2_bpe/vocab.bpe \
    --vocab-file /private/home/halilakin/src/gpt2_bpe/encoder.json \
    --input-dirs ${input_dir} \
    --output-dir /checkpoint/dpf/data/processed_gpt_redact \
    --use-hf-tokenizer \
    --splits-dir /checkpoint/dpf/data/splits/ \
    --attribute-move-probability 0.0 \
    --workers 40
