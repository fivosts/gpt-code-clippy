#!/bin/bash

for input_dir in \
  /private/home/dpf/data/github/python_forkless_open-source_0stars_20K+/data_dedup\
  /private/home/dpf/data/github/python_forkless_open-source_0stars_20K+_2020/data_dedup\
  /private/home/dpf/data/github/python_forkless_open-source_1star/data_dedup\
  /private/home/dpf/data/github/python_forkless_open-source_2star+/data_dedup\
  /private/home/dpf/data/gitlab/python_open-source/data_dedup\
  /private/home/dpf/data/google-code/python_open-source/data_dedup\
  /private/home/dpf/data/github/javascript_forkless_open-source/data_dedup\
  /private/home/dpf/data/gitlab/javascript_open-source/data_dedup\
  /private/home/dpf/data/google-code/javascript_open-source/data_dedup\
  /private/home/dpf/data/github/jupyter_forkless_open-source_2021_0stars_20K+/data_dedup\
  /private/home/dpf/data/github/jupyter_forkless_open-source_2021_1star+/data_dedup\
  /private/home/dpf/data/github/jupyter_forkless_open-source_to2020_1star+/data_dedup\
  /private/home/dpf/data/bigquery/jupyter/data_dedup\
  /private/home/dpf/data/bigquery/python_exclude_12-15_no-forks/data_dedup\
  /private/home/dpf/data/bitbucket/python_open-source/data_dedup\
  /private/home/dpf/data/bitbucket/javascript_open-source/data_dedup
do
  dname=`dirname $input_dir`
  name=`basename $dname`
  sbatch -J $name -o 'slurm-logs/slurm-%a-%x.out' -e 'slurm-logs/slurm-%a-%x.err' filter_single.sh $input_dir
done
