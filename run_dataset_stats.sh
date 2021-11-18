#!/bin/bash

data_dir=$1
source=$2

python -u dataset_stats_par.py $data_dir \
  --source $source \
  --tokenizer_names gpt2 codet5 \
  | tee ${data_dir}/size_stats.out
