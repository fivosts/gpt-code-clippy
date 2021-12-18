import numpy as np
import sys
from code_clippy_dataset.utils import load_dataset_infer
from datasets import load_from_disk

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dirs", nargs='+')

    args = parser.parse_args()

    print(' '.join(sys.argv))

    stars = []

    for data_dir in args.data_dirs:
        # datasets are hashed based on arguments, which are sensitive to trailing dashes (even though loading is not)
        data_dir = data_dir.rstrip("/")
        print(data_dir)

        dataset = load_dataset_infer(data_dir)

        stars.extend(map(int, dataset.data['stars'].to_numpy()))

    stars = np.array(stars)

    #stars = stars[stars > -1]
    stars = stars[stars > 0]

    buckets = 5

    qs = np.arange(buckets+1)/buckets

    quantiles = np.quantile(stars, qs)
    print(f"min: {stars.min()}")
    print(f"max: {stars.max()}")
    for q, v in zip(qs, np.quantile(stars, qs)):
        print(f"{q:0.3f}: {v}")
