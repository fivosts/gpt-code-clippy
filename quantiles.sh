#!/bin/bash

python compute_star_quantiles.py \
  /private/home/dpf/data/github/python_forkless_open-source_0stars_20K+/data_dedup_filtered_mwcf-0.4_mll-3000_pandoc_csn\
  /private/home/dpf/data/github/python_forkless_open-source_0stars_20K+_2020/data_dedup_filtered_mwcf-0.4_mll-3000_pandoc_csn\
  /private/home/dpf/data/github/python_forkless_open-source_1star/data_dedup_filtered_mwcf-0.4_mll-3000_pandoc_csn\
  /private/home/dpf/data/github/python_forkless_open-source_2star+/data_dedup_filtered_mwcf-0.4_mll-3000_pandoc_csn\
  | tee ~/data/quantiles/github-python.txt

python compute_star_quantiles.py \
  /private/home/dpf/data/gitlab/python_open-source/data_dedup_filtered_mwcf-0.4_mll-3000_pandoc_csn \
  | tee ~/data/quantiles/gitlab-python.txt

python compute_star_quantiles.py \
  /private/home/dpf/data/google-code/python_open-source/data_dedup_filtered_mwcf-0.4_mll-3000_pandoc_csn\
  | tee ~/data/quantiles/google-code-python.txt

python compute_star_quantiles.py \
  /private/home/dpf/data/github/javascript_forkless_open-source/data_dedup_filtered_mwcf-0.4_mll-3000_pandoc_csn\
  | tee ~/data/quantiles/github-javascript.txt

python compute_star_quantiles.py \
  /private/home/dpf/data/gitlab/javascript_open-source/data_dedup_filtered_mwcf-0.4_mll-3000_pandoc_csn\
  | tee ~/data/quantiles/gitlab-javascript.txt

python compute_star_quantiles.py \
  /private/home/dpf/data/google-code/javascript_open-source/data_dedup_filtered_mwcf-0.4_mll-3000_pandoc_csn\
  | tee ~/data/quantiles/google-code-javascript.txt

python compute_star_quantiles.py \
  /private/home/dpf/data/github/jupyter_forkless_open-source_2021_0stars_20K+/data_dedup_filtered_mwcf-0.4_mll-3000_pandoc_csn\
  /private/home/dpf/data/github/jupyter_forkless_open-source_2021_1star+/data_dedup_filtered_mwcf-0.4_mll-3000_pandoc_csn\
  /private/home/dpf/data/github/jupyter_forkless_open-source_to2020_1star+/data_dedup_filtered_mwcf-0.4_mll-3000_pandoc_csn\
  | tee ~/data/quantiles/github-jupyter.txt
