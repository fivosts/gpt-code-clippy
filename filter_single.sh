#!/bin/bash

#SBATCH --time=4320
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --partition=devlab
#SBATCH --mem-per-cpu=20G

data_dir=$1

# data_dir: should be a data_dedup

dname=`dirname $data_dir`

python -u filter_dataset.py \
  --num_proc 20 \
  --repos_to_exclude_paths ~/projects/JuICe/juice-notebooks/*.csv ~/data/code-search-net/*_{valid,test}_repos.csv \
  --data_dirs $data_dir | tee ${dname}/filter_csn-conservative.out
