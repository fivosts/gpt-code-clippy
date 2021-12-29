from collections import Counter, defaultdict
from os.path import splitext

from datasets import load_dataset
from transformers import GPT2Tokenizer

from multiprocessing import Pool

import tqdm

from guesslang import Guess

# HF: Whether or not to add an initial space to the input. This allows to treat
# the leading word just as any other word. (GPT2 tokenizer detect beginning of
# words by the preceding space).
ADD_PREFIX_SPACE = False

# data = load_dataset("code_clippy_dataset", data_dir="scrapes/out_forkless_open-source_10-1/")['train']

guess = Guess()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2", add_prefix_space=ADD_PREFIX_SPACE)

def tabulate_single(x):
    fname_toks = splitext(x['file_name'])
    if len(fname_toks) > 1:
        filename_ext = fname_toks[-1]
    else:
        filename_ext = None
    guessed_lang = guess.language_name(x['text'])

    tokens = tokenizer(x['text'])['input_ids']
    token_count = len(tokens)
    return {'repo_language': x['repo_language'],
            'guesslang': guessed_lang,
            'filename_extension': filename_ext,
            'token_count': token_count}

def foo(train):
    data = []
    for x in train:
        data.append(x)
        if len(data) >= 8:
            break
    return tabulate(data, n_threads=4)

def tabulate(data, n_threads=20):
    file_counts = defaultdict(Counter)
    token_counts = defaultdict(Counter)

    with Pool(n_threads) as p:
        all_tabulated = list(tqdm.tqdm(p.imap(tabulate_single, data), total=len(data), ncols=80))

    assert len(all_tabulated) == len(data)

    guesslang_mismatch = defaultdict(list)

    for datum, tabulated in zip(datum, all_tabulated):
        token_count = tabulated['token_count']
        for language_source in ['repo_language', 'guesslang', 'filename_extension']:
            language = tabulated[language_source]
            file_counts[language_source][language] += 1
            token_counts[language_source][language] += token_count
        if tabulated['filename_extension'] == '.py' and tabulated['guesslang'] != 'Python':
            guesslang_mismatch[tabulated['guesslang']].append((datum['repo_name'], datum['file_name']))

    return file_counts, token_counts, guesslang_mismatch
