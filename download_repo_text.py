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
from multiprocessing import cpu_count, Pool
from tqdm import tqdm
import argparse
import subprocess
import functools

from hacky_linguist import LANGUAGE_EXTENSIONS
from github_utils.utils import get_git_commits, get_git_date
from code_clippy_dataset.utils import split_into_chunks, timeout, TimeoutError

MAX_TEXT_SIZE = 10_000_000

OPEN_SOURCE_LICENSES = {
    # gitlab
    'mit', 'apache-2.0', 'bsd-3-clause', 'bsd-3-clause-clear', 'bsd-2-clause', 
    # github
    'MIT License',
    'Apache License 2.0',
    'BSD 3-Clause New or Revised License',
    'BSD 2-Clause Simplified License',
    'BSD 3-Clause Clear License',
    # licensee
    'BSD 3-Clause "New" or "Revised" License',
    'BSD 2-Clause "Simplified" License',
    # Google Code
    'mit', 'asf20', 'bsd',
}

LICENSE_STANDARDIZATION = {
    # github
    'MIT License': 'mit',
    'Apache License 2.0': 'apache-2.0',
    'BSD 3-Clause New or Revised License': 'bsd-3-clause',
    'BSD 2-Clause Simplified License': 'bsd-2-clause',
    'BSD 3-Clause Clear License': 'bsd-3-clause-clear',
    # licensee
    'BSD 3-Clause "New" or "Revised" License': 'bsd-3-clause',
    'BSD 2-Clause "Simplified" License': 'bsd-2-clause',
}

MIME = magic.Magic(mime=True)


def filter_by_stars(repo_data, n_stars):
    return [record for record in repo_data if int(record['stargazers']) >= n_stars]


def get_content(f):
    # discerns filetype with mime and reads text from file if possible

    type = None
    try:
        type = MIME.from_file(f)
        if not type.startswith('text'):
            return
        with open(f, 'rb') as fromfh:
            buf = fromfh.read()
        buf = buf.decode('UTF-8')
        return buf
    except UnicodeDecodeError:
        # bad encoding, try different encoding
        try:
            enc = None
            enc = chardet.detect(buf)
            if enc['encoding'] is None:
                return
            buf = buf.decode(enc['encoding'])
            return buf
        except UnicodeDecodeError:
            return
    except KeyboardInterrupt:
        sys.exit()
    except FileNotFoundError as e:
        # bad symlink
        import os.path
        if not os.path.islink(f):
            # something went horribly wrong!
            pass

# def filter_criteria(files):
#     filtered_files = []
#     for f in files:
#         size = os.path.getsize(f)
#         if '.git' not in f and f[0] is not '.' and \
#             'LICENSE' not in f and 'node_modules' not in f and \
#             '.min.' not in f and f.split('.')[-1] not in bad_extensions and \
#             f.split('.')[-1] in lang_exts and size

def detect_licenses(repodir):
    result = subprocess.run(f"licensee detect --json {repodir}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        license_data = json.loads(result.stdout)
        if result.stderr is not None and result.stderr.strip():
            print(result.stderr.strip())
        return [d['meta']['title'] for d in license_data['licenses']]
    except Exception as e:
        print(e, file=sys.stderr)
        return []

def remove_prefix(string, prefix):
    assert string.startswith(prefix)
    return string[len(prefix):]

def _process_repo(repo_data, repodir, license_filter, repo_type, extra_tags=None, remove_repo=True):
    if extra_tags is None:
        extra_tags = {}
    out = None
    # get metadata
    meta = repo_data.copy()
    # for backward compatibility
    meta['repo_name'] = meta['name']
    name = meta['name']

    if repo_type == 'git':
        exclude_list = ['.git']
    elif repo_type == 'svn':
        exclude_list = ['.svn', 'tags', 'branches']
    elif repo_type == 'hg':
        exclude_list = [".hg"]
    else:
        raise NotImplementedError(f"repo_type {repo_type}")

    #meta = {'repo_name': name, 'stars': stars, 'repo_language': lang, 'license': licen}
    try:
        if 'license' not in meta:
            licenses = [LICENSE_STANDARDIZATION.get(license, license) for license in detect_licenses(repodir)]
            meta['detected_licenses'] = licenses
        else:
            meta['license'] = LICENSE_STANDARDIZATION.get(meta['license'], meta['license'])
            licenses = [meta['license']]
        if license_filter is not None:
            # ensure this repo has a license and all licenses are within the filter
            if not licenses or any(license not in license_filter for license in licenses):
                return None, meta
        for curdir, dirs, files in os.walk(repodir):
            # size = os.path.getsize('C:\\Python27\\Lib\\genericpath.py')
            dirs[:] = [d for d in dirs if d not in exclude_list]
            filenames = []
            extensions = []
            text_outputs = []
            for short_file_path in files:
                full_file_path = os.path.join(curdir, short_file_path)
                split_ext = os.path.splitext(short_file_path)
                if len(split_ext) < 2:
                    continue
                extension = split_ext[-1]
                if extension not in LANGUAGE_EXTENSIONS:
                    continue
                if '.git' in short_file_path or short_file_path[0] == '.' or 'LICENSE' in short_file_path or 'node_modules' in short_file_path or '.min.' in short_file_path:
                    continue
                try:
                    mime_type = MIME.from_file(full_file_path)
                except FileNotFoundError:
                    mime_type = "n/a"

                try:
                    text = get_content(full_file_path)
                except TimeoutError:
                    raise TimeoutError
                except Exception as e:
                    print(e)
                    text = None

                if text is not None and text.strip() and len(text) < MAX_TEXT_SIZE:
                    meta_updated = dict(file_path=full_file_path, file_name=short_file_path, mime_type=mime_type, **meta, **extra_tags)
                    if out is None:
                        out = [[text, meta_updated]]
                    else:
                        out.append([text, meta_updated])
        #shutil.rmtree(repodir, ignore_errors=True)
    except TimeoutError:
        print(f"Processing for {name} timed out")
    except Exception as e:
        print(e)
    return out, meta

def process_repo_list(args, repo_data, clone_timeout, processing_timeout, source='github', license_filter=None, historic_checkout=False, abort_on_errors=False, scratch_dir=None):
    # TODO: get rid of redundant arguments that are also in args

    assert scratch_dir is not None

    # make output dirs
    tmp_dir = os.path.join(scratch_dir, '.tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    out = None
    meta = repo_data
    repodir = None
    try:
        name = repo_data['name']
        is_git = True
        if source == 'github':
            base_url = f'https://github.com/{name}'
        elif source == 'gitlab':
            base_url = f'https://gitlab.com/{name}.git'
        elif source == 'bitbucket':
            base_url = f'https://bitbucket.org/{name}.git'
        elif source == 'google_code':
            is_git = False
            base_url = f"https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/{name}/source-archive.zip"
        else:
            raise ValueError(f"invalid source {source}")
        if is_git:
            # gitlab allows this to have more than 2
            username, projectname = name.split("/")[-2:]
            rootfolder = os.path.join(tmp_dir, username)
            repodir = os.path.join(rootfolder, projectname)

            if historic_checkout:
                command = f'GIT_TERMINAL_PROMPT=0 git clone --single-branch {base_url} {projectname}'
            else:
                # clones master branch of repos with depth 1 (most recent commit only), ignoring any terminal prompts
                command = f'GIT_TERMINAL_PROMPT=0 git clone --depth 1 --single-branch {base_url} {projectname}'

            repo_type = 'git'
        else:
            assert not historic_checkout
            assert not args.pr_comments
            rootfolder = os.path.join(tmp_dir, name)
            repodir = os.path.join(rootfolder, name)

            command = f'wget {base_url}; unzip source-archive.zip'

            repo_type = repo_data['repoType']
        os.makedirs(rootfolder, exist_ok=True)
        p = subprocess.Popen(
            command,
            shell=True,
            cwd=rootfolder,
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        )
        try:
            p.wait(clone_timeout)
        except subprocess.TimeoutExpired:
            print(f'download for {name} timed out ')
            p.kill()
        # if repo_type == 'git':
        #     # not strictly necessary since we'll skip these directories in the exclude list
        #     shutil.rmtree(f'{repodir}/.git', ignore_errors=True)
        # elif repo_type == 'hg':
        #     shutil.rmtree(f'{repodir}/.hg', ignore_errors=True)
        # extracts text files from repo and returns them as list : [[text, metadata], ... ]
        if is_git:
            commits = get_git_commits(repodir)
            if commits:
                this_commit = commits[0]
                extra_tags = {
                    'commit': commits[0],
                    'commit_date': get_git_date(repodir, this_commit),
                    'commits_in_past': 0,
                }
            else:
                extra_tags = {}
        else:
            extra_tags = {}
        out, meta = timeout(_process_repo, args=(repo_data, repodir, license_filter, repo_type, extra_tags), timeout_duration=processing_timeout)
        if is_git and out is not None and historic_checkout:
            commits_in_past = len(commits)//2
            if commits_in_past > 0:
                middle_commit = commits[commits_in_past]
                middle_extra_tags = {
                    'commit': middle_commit,
                    'commit_date': get_git_date(repodir, middle_commit),
                    'commits_in_past': commits_in_past,
                }
                proc = subprocess.run(
                    f'git checkout {middle_commit}', shell=True, 
                    cwd=repodir, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
                )
                if proc.returncode == 0:
                    new_out, new_meta = timeout(_process_repo, args=(repo_data, repodir, license_filter, repo_type, middle_extra_tags), timeout_duration=processing_timeout)
                    if new_out is not None:
                        #print(f"successful historical checkout of commit {middle_commit} for {name}")
                        out += new_out
                    else:
                        #print(f"could not do historical checkout of commit {middle_commit} for {name}")
                        out = None
            else:
                out = None

    except Exception as e:
        if ignore_errors:
            print(e)
            print(f"error dir: {repodir}")
        else:
            raise e
        # if verbose:
        #     print(e)
    if repodir is not None:
        try:
            shutil.rmtree(repodir, ignore_errors=True)
        except Exception as e:
            print(e)
    return out, meta


def process_args():
    parser = argparse.ArgumentParser(
        description='CLI for git SCM downloader - A tool for scraping repos as text from github/gitlab/bitbucket')
    parser.add_argument('output_dir')
    parser.add_argument('--scratch_dir', default="/scratch/dpf/code-crawl-scratch")
    parser.add_argument('--input_csvs', nargs="+", required=True)
    parser.add_argument('--n_threads', help='number of threads for parallel processing, -1 for cpu_count * 3',
                        default=10,
                        type=int)
    parser.add_argument('--n_stars', help='filter repos with less than n_stars stars',
                        default=-1,
                        type=int)
    parser.add_argument('--chunk_size', help='size of chunks to feed into each thread',
                        default=-1,
                        type=int)
    parser.add_argument('--clone_timeout', help='timeout for git clone command in seconds',
                        default=150,
                        type=int)
    parser.add_argument('--processing_timeout', help='timeout for processing repo to text files in seconds',
                        default=150,
                        type=int)
    parser.add_argument('--commit_freq', help='how often (in number of chunks) to commit the archive file',
                        default=10,
                        type=int)
    parser.add_argument('--source', choices=['github', 'gitlab', 'bitbucket', 'google_code'], default='github')
    parser.add_argument('--language_filter')
    parser.add_argument('--min_language_size', type=int)
    parser.add_argument('-v', '--verbose', help='if flag is present, print errors', action='store_true')
    parser.add_argument('--open_source_only', action='store_true')
    parser.add_argument('--historic_checkout', action='store_true')
    parser.add_argument('--pr_comments', action='store_true')
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

    if args.open_source_only:
        license_filter = OPEN_SOURCE_LICENSES
    else:
        license_filter = None

    repo_data = []

    to_process = set()

    for input_csv in args.input_csvs:
        with open(input_csv, 'r') as f:
            csv_reader = csv.DictReader(f)
            this_repo_data = list(csv_reader)

        # possibly filter out repos from the file in --already_scraped_input
        repo_data_filtered = []

        skipped = Counter()

        for t in this_repo_data:
            if args.source == 'gitlab':
                t['name'] = remove_prefix(t['url'], 'https://gitlab.com/')

            if args.source == 'bitbucket':
                t['main_language'] = t['language']
                t['name'] = t['full_name']

            if args.source == 'google_code':
                t['main_language'] = t['main_common_language'].lower()

            assert 'name' in t, f"csv keys {t.keys()} do not include 'name'"

            if t['name'] in already_processed: 
                skipped['processed'] += 1
                continue
            if t['name'] in to_process:
                skipped['duplicate'] += 1
                continue
            if 'license' in t and license_filter is not None and t['license'] not in license_filter:
                skipped['license'] += 1
                continue

            if args.language_filter is not None: 
                if args.min_language_size is not None:
                    total_sizes_by_language = {k.lower(): v for k, v in json.loads(t['total_sizes_by_language']).items()}
                    if args.language_filter not in total_sizes_by_language or total_sizes_by_language[args.language_filter] < args.min_language_size:
                        skipped['language'] += 1
                        continue
                else:
                    assert 'main_language' in t, f"csv keys {t.keys()} do not include 'main_language'"
                    if t['main_language'] != args.language_filter:
                        skipped['language'] += 1
                        continue
            to_process.add(t['name'])
            repo_data_filtered.append(t)
        print(f"{input_csv}:\tskipping {sum(skipped.values())} repos \t {skipped.most_common()}")
        repo_data.extend(repo_data_filtered)

    print(f"{len(repo_data)} repos to process")

    already_scraped_output_file = open(already_processed_file, 'a', buffering=1)

    print(args.output_dir)

    data_dir = os.path.join(args.output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # filter by number of stars
    if args.n_stars != -1:
        repo_data = filter_by_stars(repo_data, args.n_stars)
    repo_data.sort(key=lambda t: t['name'])

    random.seed(420)
    random.shuffle(repo_data)

    n_threads = cpu_count() * 3 if args.n_threads == -1 else args.n_threads
    chunk_size = n_threads * 3 if args.chunk_size == -1 else args.chunk_size

    assert n_threads != 0

    # do work
    repo_chunks = split_into_chunks(repo_data, chunk_size)
    ar = lmd.Archive(data_dir)
    processed_names = []

    pool = Pool(n_threads)
    pbar = tqdm(repo_chunks, total=len(repo_chunks), ncols=80)
    success_hist = []

    license_counter = Counter()

    os.chdir(scratch_dir)

    # TODO: also include language filter
    processing_function = functools.partial(
        process_repo_list, clone_timeout=args.clone_timeout, processing_timeout=args.processing_timeout,
        source=args.source, license_filter=license_filter, historic_checkout=args.historic_checkout,
        abort_on_errors=args.abort_on_errors, scratch_dir=scratch_dir, args=args,
    )

    for count, chunk in enumerate(pbar):
        not_none = 0
        none = 0
        repos_out = pool.map(processing_function, chunk)
        for processed_files, repo_data in repos_out: 
            if 'detected_licenses' in repo_data:
                licenses = repo_data['detected_licenses']
            elif 'license' in repo_data:
                licenses = [repo_data['license']]
            else:
                licenses = []
            if not licenses:
                licenses = ['no-license']
            license_counter.update(licenses)

            processed_names.append(repo_data['name'])
            if processed_files is not None:
                not_none += 1
                for f in processed_files:
                    try:
                        ar.add_data(f[0], meta=f[1])
                    except UnicodeEncodeError as e:
                        print(e)
                        continue
            else:
                none += 1

        # remove any leftover files
        subprocess.Popen("rm -rfv .tmp && mkdir .tmp", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, cwd=scratch_dir)
        if count % args.commit_freq == 0:
            ar.commit()
            for name in processed_names:
                already_scraped_output_file.write(name+"\n")
            already_scraped_output_file.flush()
            processed_names = []
            tqdm.write(str(license_counter.most_common(20)))
        this_success_rate = (not_none / len(repos_out)) * 100
        success_hist.append(this_success_rate)
        success_rate = sum(success_hist) / len(success_hist)
        pbar.set_postfix({'overall_sr': success_rate, 'this_sr': this_success_rate})
    ar.commit() # final commit
    for name in processed_names:
        already_scraped_output_file.write(name+"\n")
    already_scraped_output_file.close()
