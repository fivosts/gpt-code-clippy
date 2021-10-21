import datasets
from collections import defaultdict
dataset = datasets.load_dataset("script.py", data_dir="../../scrapes/out_forkless_open-source_10-1/github_data_1", split="train")
by_variables = defaultdict(list)
def get_variables(examples):
    import re
    variables = [" ".join(re.split(r"\W+", text)) for text in examples["text"]]
    for var, repo_name, file_name in zip(variables, examples["repo_name"], examples["file_name"]):
        by_variables[var].append((repo_name, file_name))
    return {"variables": variables}
dataset_mapped = dataset.map(get_variables, batched=True)

# len(dataset_mapped)
# len(by_variables)
by_variables_lengths = {k: len(v) for k, v in by_variables.items()}
# max(by_variables_lengths.items(), key=lambda t: t[1])
var_lengths_sorted = list(sorted(by_variables_lengths.items(), key=lambda t: t[1], reverse=True))
var_lengths_sorted[:10]
# var_lengths_sorted[10:20]
# var_lengths_sorted[30:40]
# var_lengths_sorted[40:50]
