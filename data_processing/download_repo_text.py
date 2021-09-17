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
from multiprocessing import cpu_count, Pool
from tqdm import tqdm
import argparse
import subprocess
from itertools import repeat

bad_extensions = [
    'app',
    'bin',
    'bmp',
    'bz2',
    'class',
    'csv',
    'dat',
    'db',
    'dll',
    'dylib',
    'egg',
    'eot',
    'exe',
    'gif',
    'gitignore',
    'glif',
    'gradle',
    'gz',
    'ico',
    'jar',
    'jpeg',
    'jpg',
    'lo',
    'lock',
    'log',
    'mp3',
    'mp4',
    'nar',
    'o',
    'ogg',
    'otf',
    'p',
    'pdf',
    'png',
    'pickle',
    'pkl',
    'pyc',
    'pyd',
    'pyo',
    'rkt',
    'so',
    'ss',
    'svg',
    'tar',
    'tsv',
    'ttf',
    'war',
    'webm',
    'woff',
    'woff2',
    'xz',
    'zip',
    'zst'
]
# load programming language extensions from json file
with open("./Programming_Languages_Extensions.json", "r") as f:
    data = json.load(f)

lang_exts = []
for i in data:
    if "extensions" not in i:
        continue
    lang_exts.extend(i["extensions"])

mime = magic.Magic(mime=True)


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


def is_digit(x):
    return x in "1234567890"


def keep(x):
    # simple filters to decide whether a file is worth keeping
    num_digits = len(list(filter(is_digit, x)))
    num_newlines = len(list(filter(lambda x: x == '\n', x)))
    if num_digits / len(x) > 0.8:
        return False

    # avg line length
    if len(x) / (num_newlines + .001) > 200:
        return False

    return True


def filter_by_stars(repo_data, n_stars):
    return [item for item in repo_data if int(item[1]) >= n_stars]


def get_content(f):
    # discerns filetype with mime and reads text from file if possible

    type = None
    try:
        enc = 'utf-8'
        type = mime.from_file(f)
        if not type.startswith('text'):
            return
        with open(f, 'rb') as fromfh:
            buf = fromfh.read()

        buf = buf.decode('UTF-8')
        if not keep(buf):
            return

        return buf
    except UnicodeDecodeError:
        # bad encoding, try different encoding
        try:
            enc = None
            enc = chardet.detect(buf)
            if enc['encoding'] is None:
                return
            buf = buf.decode(enc['encoding'])
            if not keep(buf):
                return
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
            ...

# def filter_criteria(files):
#     filtered_files = []
#     for f in files:
#         size = os.path.getsize(f)
#         if '.git' not in f and f[0] is not '.' and \
#             'LICENSE' not in f and 'node_modules' not in f and \
#             '.min.' not in f and f.split('.')[-1] not in bad_extensions and \
#             f.split('.')[-1] in lang_exts and size

def _process_repo(repo_data, repodir):
    out = None
    # get metadata
    name, stars, lang, licen = repo_data
    meta = {'repo_name': name, 'stars': stars, 'repo_language': lang, 'license': licen}
    try:
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
                if extension not in lang_exts or extension in bad_extensions:
                    continue
                if '.git' in short_file_path or short_file_path[0] == '.' or 'LICENSE' in short_file_path or 'node_modules' in short_file_path or '.min.' in short_file_path:
                    continue

                try:
                    mime_type = mime.from_file(full_file_path)
                except FileNotFoundError:
                    mime_type = "n/a"

                try:
                    text = get_content(full_file_path)
                except TimeoutError:
                    raise TimeoutError
                except:
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
    return out


def process_repo(repo_data, repodir, processing_timeout):
    return timeout(_process_repo, args=(repo_data, repodir), timeout_duration=processing_timeout)


def process_repo_list(repo_data, clone_timeout, processing_timeout):
    out = None
    try:
        name, stars, lang, license = repo_data
        username, projectname = name.split("/")
        rootfolder = os.path.join(".tmp", username)
        os.makedirs(rootfolder, exist_ok=True)
        repodir = os.path.join(rootfolder, projectname)
        # clones master branch of repos with depth 1 (most recent commit only), ignoring any terminal prompts
        git_command = f'GIT_TERMINAL_PROMPT=0 git clone --depth 1 --single-branch https://github.com/{name} {projectname}'
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
        out = process_repo(repo_data, repodir, processing_timeout=processing_timeout)
    except Exception as e:
        print(e)
        # if verbose:
        #     print(e)
    return out


def process_args():
    parser = argparse.ArgumentParser(
        description='CLI for github downloader - A tool for scraping repos as text from github')
    parser.add_argument('input_csv')
    parser.add_argument('output_dir')
    parser.add_argument('--n_threads', help='number of threads for parallel processing, -1 for cpu_count',
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
    parser.add_argument('-v', '--verbose', help='if flag is present, print errors', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':

    args = process_args()  # parse args
    verbose = args.verbose

    # read repo data to a tuple (reponame, n_stars, language, license)
    with open(args.input_csv, 'r') as f:
        csv_reader = csv.reader(f)
        row = next(csv_reader)
        assert tuple(row) == ('name', 'stargazers', 'main_language', 'license')
        repo_data = list(map(tuple, csv_reader))

    if not os.path.isdir(args.output_dir):
        raise Exception("output directory {args.output_dir} does not exist; exiting")

    print(args.output_dir)
    os.chdir(args.output_dir)

    # make output dirs
    if '.tmp' not in os.listdir():
        os.makedirs('.tmp')
    if 'github_data' not in os.listdir():
        os.makedirs('github_data')

    # filter by number of stars
    if args.n_stars != -1:
        repo_data = filter_by_stars(repo_data, args.n_stars)
    repo_data.sort()

    random.seed(420)
    random.shuffle(repo_data)

    n_threads = cpu_count() * 3 if args.n_threads == -1 else args.n_threads
    chunk_size = n_threads * 3 if args.chunk_size == -1 else args.chunk_size

    assert n_threads != 0

    # do work
    repo_chunks = split_into_chunks(repo_data, chunk_size)
    archive_name = 'github_data'
    ar = lmd.Archive(archive_name)
    pool = Pool(n_threads)
    pbar = tqdm(repo_chunks, total=len(repo_chunks), ncols=80)
    success_hist = []
    for count, chunk in enumerate(pbar):
        repos_out = pool.starmap(process_repo_list,
                                 zip(chunk, repeat(args.clone_timeout), repeat(args.processing_timeout)))
        not_none = 0
        none = 0
        for repo in repos_out:
            if repo is not None:
                not_none += 1
                for f in repo:
                    ar.add_data(f[0], meta=f[1])
            else:
                none += 1

        # remove any leftover files
        subprocess.Popen("rm -rfv .tmp && mkdir .tmp", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        if count % args.commit_freq == 0:
            ar.commit()
        success_hist.append((not_none / len(repos_out)) * 100)
        success_rate = sum(success_hist) / len(success_hist)
        pbar.set_postfix({"Success Rate": success_rate})
    ar.commit() # final commit
