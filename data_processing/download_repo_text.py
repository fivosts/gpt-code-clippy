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
from itertools import repeat
import functools

from hacky_linguist import COMMON_LANGUAGES, EXTENSION_TO_LANGUAGE

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

def build_language_extensions(filename='./Programming_Languages_Extensions_filtered.json'):
    lang_exts = []
    with open(filename) as f:
        for i in json.load(f):
            if "extensions" not in i:
                continue
            lang_exts.extend(i["extensions"])
    return lang_exts

# load programming language extensions from json file
LANGUAGE_EXTENSIONS = build_language_extensions()

MIME = magic.Magic(mime=True)

class TimeoutError(Exception):
    pass

def timeout(func, args=(), kwargs={}, timeout_duration=150, default=None):
    # wrap any function in this wrapper to raise a TimeoutError after timeout_duration secs
    import signal

    def handler(signum, frame):
        raise TimeoutError()

    # set the timeout handler
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout_duration)
    try:
        result = func(*args, **kwargs)
    except TimeoutError:
        result = default
    finally:
        signal.alarm(0)

    return result

def split_into_chunks(l, n):
    n = max(1, n)
    return [l[i:i + n] for i in range(0, len(l), n)]


def filter_by_stars(repo_data, n_stars):
    return [record for record in repo_data if int(repo['stargazers']) >= n_stars]


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

def _process_repo(repo_data, repodir, license_filter, license_counter=None):
    out = None
    # get metadata
    meta = repo_data.copy()
    # for backward compatibility
    meta['repo_name'] = meta['name']

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

                if text is not None and text.strip():
                    meta_updated = dict(file_name=short_file_path, mime_type=mime_type, **meta)
                    if out is None:
                        out = [[text, meta_updated]]
                    else:
                        out.append([text, meta_updated])
        shutil.rmtree(repodir, ignore_errors=True)
    except TimeoutError:
        print(f"Processing for {name} timed out")
    except Exception as e:
        print(e)
    return out, meta


def process_repo_list(repo_data, clone_timeout, processing_timeout, source='github', license_filter=None):
    out = None
    meta = repo_data
    try:
        name = repo_data['name']
        lang = repo_data['main_language']
        if source == 'github':
            base_url = f'https://github.com/{name}'
        elif source == 'gitlab':
            base_url = f'https://gitlab.com/{name}.git'
        elif source == 'bitbucket':
            base_url = f'https://bitbucket.org/{name}.git'
        else:
            raise ValueError(f"invalid source {source}")
        # gitlab allows this to have more than 2
        username, projectname = name.split("/")[-2:]
        rootfolder = os.path.join(".tmp", username)
        os.makedirs(rootfolder, exist_ok=True)
        repodir = os.path.join(rootfolder, projectname)
        # clones master branch of repos with depth 1 (most recent commit only), ignoring any terminal prompts
        git_command = f'GIT_TERMINAL_PROMPT=0 git clone --depth 1 --single-branch {base_url} {projectname}'
        p = subprocess.Popen(
            git_command,
            shell=True,
            cwd=os.path.join(os.getcwd(), rootfolder),
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        )
        try:
            p.wait(clone_timeout)
        except subprocess.TimeoutExpired:
            print(f'Git clone for {name} timed out ')
            p.kill()
        shutil.rmtree(f'{repodir}/.git', ignore_errors=True)
        # extracts text files from repo and returns them as list : [[text, metadata], ... ]
        out, meta = timeout(_process_repo, args=(repo_data, repodir, license_filter), timeout_duration=processing_timeout)
    except Exception as e:
        print(e)
        # if verbose:
        #     print(e)
    return out, meta


def process_args():
    parser = argparse.ArgumentParser(
        description='CLI for git SCM downloader - A tool for scraping repos as text from github/gitlab/bitbucket')
    parser.add_argument('output_dir')
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
    parser.add_argument('--source', choices=['github', 'gitlab', 'bitbucket'], default='github')
    parser.add_argument('--language_filter')
    parser.add_argument('-v', '--verbose', help='if flag is present, print errors', action='store_true')
    parser.add_argument('--open_source_only', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':

    args = process_args()  # parse args
    verbose = args.verbose

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

            assert 'name' in t, f"csv keys {t.keys()} do not include 'name'"
            assert 'main_language' in t, f"csv keys {t.keys()} do not include 'main_language'"

            if t['name'] in already_processed: 
                skipped['processed'] += 1
                continue
            if t['name'] in to_process:
                skipped['duplicate'] += 1
                continue
            if 'license' in t and license_filter is not None and t['license'] not in license_filter:
                skipped['license'] += 1
                continue
            if args.language_filter is not None and t['language'] != args.language_filter:
                skipped['language'] += 1
                continue
            to_process.add(t['name'])
            repo_data_filtered.append(t)
        print(f"{input_csv}:\tskipping {sum(skipped.values())} repos \t {skipped.most_common()}")
        repo_data.extend(repo_data_filtered)

    print(f"{len(repo_data)} repos to process")

    already_scraped_output_file = open(already_processed_file, 'a', buffering=1)

    print(args.output_dir)
    os.chdir(args.output_dir)

    # make output dirs
    if '.tmp' not in os.listdir():
        os.makedirs('.tmp')
    if 'data' not in os.listdir():
        os.makedirs('data')

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
    archive_name = 'data'
    ar = lmd.Archive(archive_name)
    processed_names = []

    pool = Pool(n_threads)
    pbar = tqdm(repo_chunks, total=len(repo_chunks), ncols=80)
    success_hist = []

    license_counter = Counter()

    # TODO: also include language filter
    processing_function = functools.partial(
        process_repo_list, clone_timeout=args.clone_timeout, processing_timeout=args.processing_timeout,
        source=args.source, license_filter=license_filter,
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
        subprocess.Popen("rm -rfv .tmp && mkdir .tmp", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
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
