import argparse
import glob
from datasets import load_from_disk
import random
import sys
import pprint
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--fraction_val", type=float, default=0.01)
    parser.add_argument("--out_prefix", default="/private/home/dpf/data/splits")

    args = parser.parse_args()

    print(' '.join(sys.argv))
    pprint.pprint(vars(args))

    repo_name_to_data = {}

    names = set()

    filenames = [fname for file_glob in args.datasets for fname in glob.glob(file_glob)]
    print("filenames:")
    print(' '.join(filenames))

    for fname in filenames:
        data = load_from_disk(fname)

        for repo_name in data.data['repo_name'].unique():
            repo_name = str(repo_name)
            repo_name = repo_name.lower()
            try:
                owner, name = repo_name.split('/')
            except:
                print(f"can't split {repo_name}; using {repo_name} as key")
                name = repo_name
            names.add(name)
            if repo_name in repo_name_to_data:
                print(f"{repo_name} found in {repo_name_to_data[repo_name]} and {fname}")
            else:
                repo_name_to_data[repo_name] = fname
    print(f"{len(repo_name_to_data):_} distinct owner/repos")
    print(f"{len(names):_} distinct repos")

    names = list(sorted(names))

    random.seed(0)
    random.shuffle(names)

    num_val = int(len(names) * args.fraction_val)

    train_names = names[:-num_val]
    val_names = names[-num_val:]

    print(f"{len(train_names)} train")
    print(f"{len(val_names)} val")

    assert not(set(train_names) & set(val_names))

    with open(os.path.join(args.out_prefix, "train"), 'w') as f:
        for n in train_names:
            f.write(f'{n}\n')

    with open(os.path.join(args.out_prefix, "val"), 'w') as f:
        for n in val_names:
            f.write(f'{n}\n')
