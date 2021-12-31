# Copyright maintained by EleutherAI. Originally from https://github.com/EleutherAI/github-downloader

import chardet
import magic
import lm_dataformat as lmd
import os
import random
import sys
import traceback
import shutil
import csv
import json
from collections import Counter
from multiprocessing import cpu_count, Pool, Process, JoinableQueue
from tqdm import tqdm
import argparse
import subprocess
import functools
import zstandard as zstd

from hacky_linguist import LANGUAGE_EXTENSIONS

from code_clippy_dataset.utils import split_into_chunks, timeout, TimeoutError
from github_utils.utils import GitRepo, get_git_commits, get_git_date
from github_utils.comments import aggregate_comments
from download_repo_text import MIME, MAX_TEXT_SIZE, get_content

class Worker(Process):
    def __init__(self, args, index, input_queue, output_queue,
                 progress_bar=None, **kwargs):
        super().__init__(**kwargs)
        self.args = args
        self.index = index
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.progress_bar = progress_bar

    def run(self):
        #print(f"worker {self.index} starting")
        while True:
            input = self.input_queue.get()
            if self.progress_bar is not None:
                with self.progress_bar.get_lock():
                    self.progress_bar.update(1)
            if input is None:
                self.output_queue.put(None)
                self.input_queue.task_done()
                break

            result = self.compute_result(input)
            self.output_queue.put(result)
            self.input_queue.task_done()
        print(f"worker {self.index} ending")
    
    def compute_result(self, input):
        raise NotImplementedError()

class CommentWorker(Worker):
    def compute_result(self, input):
        owner, repo = input['repo_name'].split('/')
        comment_data = aggregate_comments(owner, repo, approximate_max_comments=self.args.approximate_max_comments)
        input.update(comment_data)
        return input

class FileWorker(Worker):
    def compute_result(self, repo_data):
        args = self.args
        assert args.scratch_dir is not None

        # make output dirs
        tmp_dir = os.path.join(args.scratch_dir, '.tmp')
        os.makedirs(tmp_dir, exist_ok=True)

        repodir = None

        # Dict[pull_request_id: int, Dict[(original_commit_id: str, path: str, original_position: int), List[comment: str]]]
        comments = repo_data['comments']

        # Dict[commit_id, [List[path: str]]]
        paths_by_commit = repo_data['paths_by_commit']

        # int
        num_comments = repo_data['num_comments']

        try:
            name = repo_data['name']
            base_url = f'https://github.com/{name}'

            username, projectname = name.split("/")
            rootfolder = os.path.join(tmp_dir, username)
            repodir = os.path.join(rootfolder, projectname)

            command = f'GIT_TERMINAL_PROMPT=0 git clone {base_url} {projectname}'

            os.makedirs(rootfolder, exist_ok=True)
            p = subprocess.Popen(
                command,
                shell=True,
                cwd=rootfolder,
                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
            )
            try:
                p.wait(args.clone_timeout)
            except subprocess.TimeoutExpired:
                print(f'download for {name} timed out ')
                p.kill()

            git_repo = GitRepo(repodir)

            file_data_by_commit_and_path = {}

            commit_info = {}

            for commit, paths in paths_by_commit:
                for path in paths:
                    # dict with keys {'path', 'parent_content', 'parent_commit', 'child_content', 'child_commit'}
                    file_datum = git_repo.get_file_before_and_after_commit(commit, path)
                    del file_datum['path']
                    file_data_by_commit_and_path[(commit, path)] = file_datum
                commit_info[commit] = git_repo.get_commit_info(commit, paths)
            
            repo_data['file_data'] = file_data_by_commit_and_path
            repo_data['commit_info'] = commit_info

            out_dir = os.path.join(args.output_dir, "data", username)
            os.makedirs(out_dir, exist_ok=True)

            comp = zstd.ZstdCompressor(level=3, threads=4)
            with open(os.path.join(out_dir, f"{projectname}.json.zstd"), 'wb') as f:
                writer = comp.stream_writer(f)
                json.dump(repo_data, writer, indent=4, sort_keys=True)

        except Exception as e:
            if not args.abort_on_errors:
                print(e)
                print(f"error dir: {repodir}")
                return (name, False, e)
            else:
                raise e
        if repodir is not None:
            try:
                shutil.rmtree(repodir, ignore_errors=True)
            except Exception as e:
                print(e)
        return (name, True, None)


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir')
    parser.add_argument('--scratch_dir', default="/scratch/dpf/code-crawl-scratch")
    parser.add_argument('--input_csvs', nargs="+", required=True)
    parser.add_argument('--num_processes', help='number of processes', default=10, type=int)
    parser.add_argument('--clone_timeout', help='timeout for git clone command in seconds',
                        default=480,
                        type=int)
    parser.add_argument('--processing_timeout', help='timeout for processing repo in seconds',
                        default=150,
                        type=int)
    parser.add_argument('--approximate_max_comments', type=int)
    parser.add_argument('-v', '--verbose', help='if flag is present, print errors', action='store_true')
    parser.add_argument('--abort_on_errors', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':

    args = process_args()  # parse args
    verbose = args.verbose

    scratch_dir = args.scratch_dir if args.scratch_dir is not None else args.output_dir

    if not os.path.isdir(args.output_dir):
        raise Exception("output directory {args.output_dir} does not exist; exiting")

    already_processed_file = os.path.join(args.output_dir, "repos_processed.txt")
    if os.path.exists(already_processed_file):
        with open(already_processed_file, 'r') as f:
            already_processed = set(l.strip() for l in f.readlines())
    else:
        already_processed = {}

    all_repo_data = []

    to_process = set()

    for input_csv in args.input_csvs:
        # TODO: load this up and convert the format
        with open(input_csv, 'r') as f:
            csv_reader = csv.DictReader(f)
            this_repo_data = list(csv_reader)

        # possibly filter out repos from the file in --already_scraped_input
        repo_data_filtered = []

        skipped = Counter()

        for t in this_repo_data:

            assert 'name' in t, f"csv keys {t.keys()} do not include 'name'"

            if t['name'] in already_processed: 
                skipped['processed'] += 1
                continue
            if t['name'] in to_process:
                skipped['duplicate'] += 1
                continue

            to_process.add(t['name'])
            repo_data_filtered.append(t)
        print(f"{input_csv}:\tskipping {sum(skipped.values())} repos \t {skipped.most_common()}")
        all_repo_data.extend(repo_data_filtered)

    print(f"{len(all_repo_data)} repos to process")
    already_scraped_output_file = open(already_processed_file, 'a', buffering=1)

    print(args.output_dir)
    output_dir = args.output_dir

    data_dir = os.path.join(output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    all_repo_data.sort(key=lambda t: t['name'])
    random.seed(1)
    random.shuffle(all_repo_data)

    def pipe(in_iterable, WorkerClass, num_workers, pbar_name=None, **worker_kwargs):
        in_queue, out_queue = JoinableQueue(), JoinableQueue()
        workers = []
        running_count = 0
        
        for i in range(num_workers):
            worker = WorkerClass(args, i, in_queue, out_queue, **worker_kwargs)
            worker.start()
            running_count += 1
            workers.append(worker)

        num_jobs = 0

        for x in tqdm.tqdm(in_iterable, ncols=80, desc=f"{pbar_name} inputs"):
            in_queue.put(x)
            num_jobs += 1
        
        for _ in range(num_workers):
            in_queue.put(None)

        with tqdm.tqdm(total=num_jobs, ncols=80, desc=f"{pbar_name} worker") as progress_bar:
            while num_jobs > 0:
                r = out_queue.get()
                if r is None:
                    running_count -= 1
                    #print(f"running count: {running_count}")
                    out_queue.task_done()
                    continue
                num_jobs -= 1
                progress_bar.update(1)
                out_queue.task_done
                yield r

    os.chdir(scratch_dir)

    all_with_comments = pipe(all_repo_data, CommentWorker, 1, pbar_name="comments")


    results = pipe(all_with_comments, FileWorker, args.num_processes, pbar_name="files")

    success_count = 0
    total_count = 0
    for (repo_name, was_success, error) in results:
        if was_success:
            success_count += 1
        total_count += 1
        
        if total_count % 10 == 0:
            print(f"successes: {success_count} / {total_count} ({success_count/total_count*100:.2f}%)")

        already_scraped_output_file.write(repo_name+"\n")
    already_scraped_output_file.close()
