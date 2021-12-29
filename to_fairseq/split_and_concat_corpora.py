import os
import sys
import glob
import subprocess


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_prefix")
    parser.add_argument("--subsample_every_kth_file", type=int)
    parser.add_argument("--valid_and_test_kth", type=int, default=100)
    args = parser.parse_args()

    def get_file_number(fname):
        return int(os.path.basename(fname).split('_')[1])

    files = sorted(glob.glob(os.path.join(args.input_dir, "data_*")), key=get_file_number)

    train_files = []
    valid_files = []
    test_files = []
    for ix, f in enumerate(files):
        mod = ix % args.valid_and_test_kth
        if mod == args.valid_and_test_kth-2:
            valid_files.append(f)
        elif mod == args.valid_and_test_kth-1:
            test_files.append(f)
        else:
            train_files.append(f)

    def subsample(files, kth=4):
        return files[::kth]

    if args.subsample_every_kth_file is not None:
        k = args.subsample_every_kth_file
        train_files = subsample(train_files, k)
        valid_files = subsample(valid_files, k)
        test_files = subsample(test_files, k)

    for split, files in [('train', train_files), ('valid', valid_files), ('test', test_files)]:
        f_concat = ' '.join(files)
        output_fname = os.path.join(args.input_dir, f"{args.prefix}.{split}.bpe")
        with open(os.path.join(args.input_dir, f"{args.prefix}.{split}.fnames"), 'w') as f:
            f.write('\n'.join(files))
        command = f"cat {f_concat} > {output_fname}"
        print(command)
        subprocess.run(command, shell=True)
