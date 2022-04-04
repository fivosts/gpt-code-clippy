import json
import glob
from collections import Counter
import tqdm
import os
import pprint

import humanize

PRINT_FREQUENCY = 100_000
#PRINT_FREQUENCY = 1000

def readable(x, is_size=False):
    if isinstance(x, float):
        return f"{x:.2f}"
    else:
        if is_size:
            return humanize.naturalsize(x)
        else:
            return f"{x:_}"

def readable_counter(counter, is_size, limit=20):
    return {k: readable(v, is_size=is_size) for k, v in counter.most_common(limit)}

def print_counters(file_counter, size_counter, limit=20):
    print("files:")
    pprint.pprint(readable_counter(file_counter, is_size=False, limit=limit))
    pprint.pprint(readable_counter(size_counter, is_size=True, limit=limit))
    print(f"total files: {sum(file_counter.values()):_}")
    print(f"total sizes: {humanize.naturalsize(sum(size_counter.values()))}")
    print()

def load_file(filename, file_counter, size_counter):
    record_count = 0
    with open(filename) as f:
        it = tqdm.tqdm(f, ncols=120, desc=os.path.basename(filename))
        for line in it:
            record = json.loads(line)
            if record['meta']['pile_set_name'] != 'Github':
                continue
            text_size = len(record['text'])
            language = record['meta']["repo_language"]
            file_counter[language] += 1
            size_counter[language] += text_size
            record_count += 1

            if PRINT_FREQUENCY > 0 and record_count % PRINT_FREQUENCY == 0:
                print_counters(file_counter, size_counter)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file_globs", nargs='+')
    args = parser.parse_args()

    file_counter = Counter()
    size_counter = Counter()

    for glob_path in args.file_globs:
        for filename in glob.glob(glob_path):
            print(f"starting file {filename}")
            load_file(filename, file_counter, size_counter)
            print_counters(file_counter, size_counter, limit=None)
