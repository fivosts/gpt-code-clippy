import numpy as np
import sys
from code_clippy_dataset.utils import load_dataset_infer
from datasets import load_from_disk

def zeno(num_vals):
    last = 0
    vals = [last]
    while len(vals) < num_vals:
        last = 1 - ((1 - last) / 2)
        vals.append(last)
    return np.array(vals)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dirs", nargs='+')
    parser.add_argument("--log_spacing", action='store_true')
    parser.add_argument("--buckets", type=int, default=6)

    args = parser.parse_args()

    print(' '.join(sys.argv))

    stars = []
    
    repos_found = set()


    for data_dir in args.data_dirs:
        # datasets are hashed based on arguments, which are sensitive to trailing dashes (even though loading is not)
        data_dir = data_dir.rstrip("/")
        print(data_dir)

        dataset = load_dataset_infer(data_dir)

        for repo_name, this_stars in zip(dataset['repo_name'], dataset['stars']):
            if repo_name not in repos_found:
                repos_found.add(repo_name)
                stars.append(int(this_stars))

    print(f"num repos: {len(repos_found):_}")
    stars = np.array(stars)

    # -1 sometimes indicates no star count available
    stars = stars[stars > -1]

    buckets = args.buckets

    if args.log_spacing:
        qs = zeno(buckets)
    else:
        qs = np.arange(buckets+1)/buckets

    quantiles = np.quantile(stars, qs)
    print(f"min: {stars.min()}")
    print(f"max: {stars.max()}")
    for q, v in zip(qs, np.quantile(stars, qs)):
        print(f"{q:0.3f}: {v}")
