import random
import itertools
import lxml.etree
import tqdm
from datasets import load_dataset
import sentencepiece as spm

MAX_DOC_LENGTH = 10000

NEWLINE_REP = "<|\n|>"

def stackexchange_reader(filename, yield_rate=None, parse_html=False):

    basename = filename.split('/')[-1]
    if basename == 'Comments.xml':
        text_field = 'Text'
        num_rows = 82_037_744 - 3
    elif basename == 'Posts.xml':
        text_field = 'Body'
        num_rows = 53_949_888 - 3
    else:
        raise ValueError(f"unrecognized basename {basename}")

    with open(filename, 'rb') as f:
        for event, element in tqdm.tqdm(
            lxml.etree.iterparse(f), ncols=80, total=num_rows, desc=basename
        ):
            if event == 'end' and element.tag == 'row':
                if yield_rate is not None and random.random() > yield_rate:
                    continue
                if text_field not in element.attrib:
                    continue
                text = element.attrib[text_field]
                if parse_html:
                    from bs4 import BeautifulSoup
                    parsed = BeautifulSoup(text, "html.parser")
                    yield parsed.get_text()
                else:
                    yield text

def dataset_reader(data_dir, source="github", yield_rate=None):
    # datasets are hashed based on arguments, which are sensitive to trailing dashes (even though loading is not)
    data_dir = data_dir.rstrip("/")
    data = load_dataset("code_clippy_dataset", source=source, data_dir=data_dir)['train']
    for x in tqdm.tqdm(data, ncols=80, desc=data_dir):
        if yield_rate is None or random.random() <= yield_rate:
            yield x['text']

def preprocess_text(text, max_len=MAX_DOC_LENGTH):
    i = 0
    while i < len(text):
        piece = text[i:i+max_len]
        piece = piece.replace("\n", NEWLINE_REP)
        if piece:
            yield piece
        i += max_len

if __name__ == "__main__":
    import os

    # directory and number of files to take
    data_dirs = [
        ("/private/home/dpf/data/github/python_forkless_open-source_2star+/data_dedup/", 50000),
        ("/private/home/dpf/data/github/javascript_forkless_open-source/data_dedup/", 50000),
    ]

    # file and yield (subsampling) rate
    stack_exchange = [
        ("/private/home/dpf/projects/stackexchange_dataset/dumps/stackoverflow/Posts.xml", 0.01),
        ("/private/home/dpf/projects/stackexchange_dataset/dumps/stackoverflow/Comments.xml", 0.01),
    ]
    
    model_prefix = "tokenizers/sp-unigram_github-py-js+stackoverflow"

    vocab_size = 50_000

    generators = []
    for data_dir, limit in data_dirs:
        generators.append(itertools.islice(dataset_reader(data_dir), limit))
    for path, yield_rate in stack_exchange:
        generators.append(stackexchange_reader(path, yield_rate=yield_rate, parse_html=False))

    limit = 10000
    if limit is not None:
        generators = [itertools.islice(generator, limit) for generator in generators]
        vocab_size = 10000
        model_prefix += "_small"

    generator = iter(
        piece for generator in generators
        for text in generator
        # since \n gets expanded, add in some buffer room by making pre-expansion chunks substantially smaller than MAX_DOC_LEN
        for piece in preprocess_text(text, max_len=MAX_DOC_LENGTH // 3)
    )

    user_defined_symbols = [NEWLINE_REP, '<|endoftext|>', '<|pad|>', '<|mask|>']

    spm.SentencePieceTrainer.train(
        sentence_iterator=generator, model_prefix=model_prefix, vocab_size=vocab_size,
        user_defined_symbols=user_defined_symbols,
        model_type='unigram',
        max_sentence_length=MAX_DOC_LENGTH,
        max_sentencepiece_length=128,
        split_by_unicode_script=False,
        split_by_number=False,
        split_by_whitespace=False,
        split_digits=False,
    )
