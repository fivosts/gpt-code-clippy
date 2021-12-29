from code_clippy_dataset.utils import load_dataset_infer
exclude_repos = set()

for filename in [
    '/private/home/dpf/data/github/python_forkless_open-source_2star+/repos_processed.txt',
    '/private/home/dpf/data/github/python_forkless_open-source_1star/repos_processed.txt',
]:
    with open(filename, 'r') as f:
        exclude_repos.update(line.strip() for line in f)

jupyter_bigquery = load_dataset_infer("/private/home/dpf/data/bigquery/jupyter/")
jupyter_bigquery_dedup = load_dataset_infer("/private/home/dpf/data/bigquery/jupyter_dedup/")
jupyter_bigquery_dedup_filter = load_dataset_infer("/private/home/dpf/data/bigquery/jupyter_dedup_filtered_mwcf-0.4_mll-3000_pandoc/")
