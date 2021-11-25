import sys
import pprint
import tqdm
from datasets import load_dataset
import humanize


def filter_data(data, filter_fn, prefix="", print_new_repos=False):
    amount_code = 0
    amount_code_lost = 0
    records = 0
    records_lost = 0
    repos_filtered = set()

    for record in tqdm.tqdm(data, ncols=80):
        this_amount = len(record['text'])
        repo_name = record['repo_name']
        amount_code += this_amount
        records += 1
        if filter_fn(record):
            amount_code_lost += this_amount
            records_lost += 1
            if repo_name not in repos_filtered:
                if print_new_repos:
                    tqdm.tqdm.write(f"{len(repos_filtered)}\t{record['repo_name']}\t{record['file_name']}")
                repos_filtered.add(repo_name)
    if prefix:
        prefix = f"{prefix}\t"
    print(f"{prefix}lost {records_lost} / {records} ({records_lost/records*100:.2f}% files")
    print(f"{prefix}lost {humanize.naturalsize(amount_code_lost)} / {humanize.naturalsize(amount_code)} code ({amount_code_lost/amount_code*100:.2}% filtering repos")
    return repos_filtered

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dirs", default=[
        "/private/home/dpf/data/github/python_forkless_open-source_2star+/data_dedup",
        "/private/home/dpf/data/github/python_forkless_open-source_1star/data_dedup",
        "/private/home/dpf/data/gitlab/python_open-source/data_dedup",
        "/private/home/dpf/data/bitbucket/python_open-source/data_dedup",
    ])
    parser.add_argument("--string_to_check", default="GNU General Public License")
    args = parser.parse_args()

    def file_is_gpl(record):
        return args.string_to_check in record['text']

    print(' '.join(sys.argv))
    pprint.pprint(vars(args))

    for data_dir in args.data_dirs:
        source = data_dir.lstrip("/private/home/dpf/data/").split('/')[0]
        print(f"checking data_dir {data_dir}")
        data = load_dataset("code_clippy_dataset", data_dir=data_dir.rstrip("/"), source=source)["train"]

        repos_lost = filter_data(data, file_is_gpl, prefix="file is GPL", print_new_repos=True)

        def repo_contaminated(record):
            return record['repo_name'] in repos_lost

        if bool(repos_lost):
            filter_data(data, repo_contaminated, prefix="repo has GPL file", print_new_repos=False)
        else:
            print(f"nothing found for {data_dir}")
