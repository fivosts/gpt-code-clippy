from collections import Counter, defaultdict
from os.path import splitext

from datasets import load_dataset
from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast
from tokenizers import ByteLevelBPETokenizer

from multiprocessing import Pool, Process, JoinableQueue

import numpy as np

import itertools

import tqdm
import pprint

from guesslang import Guess

# HF: Whether or not to add an initial space to the input. This allows to treat
# the leading word just as any other word. (GPT2 tokenizer detect beginning of
# words by the preceding space).
ADD_PREFIX_SPACE = False

class Worker(Process):
    def __init__(self, index, input_queue, output_queue, progress_bar=None, **kwargs):
        super().__init__()
        self.index = index
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.progress_bar = progress_bar

    def run(self):
        #print(f"worker {self.index} starting")
        guess = Guess()
        # guess = None
        gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", add_prefix_space=ADD_PREFIX_SPACE)
        codet5_tokenizer_model = ByteLevelBPETokenizer.from_file(
            "../CodeT5/tokenizer/salesforce/codet5-vocab.json",
            "../CodeT5/tokenizer/salesforce/codet5-merges.txt"
        )
        codet5_tokenizer_model.add_special_tokens([
                "<pad>",
                "<s>",
                "</s>",
                "<unk>",
                "<mask>"
        ])
        codet5_tokenizer = PreTrainedTokenizerFast(tokenizer_object=codet5_tokenizer_model)
        our_tokenizer_model = ByteLevelBPETokenizer.from_file(
            "scrapes/out_python_forkless_open-source_10-9/github_data_dedup/tokenizer_preserve_newlines/vocab.json",
            "scrapes/out_python_forkless_open-source_10-9/github_data_dedup/tokenizer_preserve_newlines/merges.txt",
        )
        our_tokenizer = PreTrainedTokenizerFast(tokenizer_object=our_tokenizer_model)
        tokenizers = {
            'gpt2': gpt2_tokenizer,
            'codet5': codet5_tokenizer,
            'ours': our_tokenizer,
        }

        while True:
            x = self.input_queue.get()
            if x is None:
                self.output_queue.put(None)
                self.input_queue.task_done()
                print("worker completing")
                break
            tabulated = self.tabulate_single(guess, tokenizers, x)
            if self.progress_bar:
                with self.progress_bar.get_lock():
                    self.progress_bar.update(1)
            self.output_queue.put((x, tabulated))
            self.input_queue.task_done()
        # print(f"worker {self.index} ending")

    @staticmethod
    def tabulate_single(guess, tokenizers, x):
        fname_toks = splitext(x['file_name'])
        if len(fname_toks) > 1:
            filename_ext = fname_toks[-1]
        else:
            filename_ext = None
        guessed_lang = guess.language_name(x['text'])
        #guessed_lang = None

        d = {'repo_language': x['repo_language'],
             'guesslang': guessed_lang,
             'filename_extension': filename_ext}

        for tokenizer_name, tokenizer in tokenizers.items():
            tokens = tokenizer(x['text'])['input_ids']
            token_count = len(tokens)
            d[f'{tokenizer_name}_token_count'] = token_count
            d[f'{tokenizer_name}_above_1024'] = 1.0 if token_count > 1024 else 0.0
            d[f'{tokenizer_name}_above_2048'] = 1.0 if token_count > 2048 else 0.0
            d[f'{tokenizer_name}_lost_1024'] = max(token_count - 1024, 0)
            d[f'{tokenizer_name}_lost_2048'] = max(token_count - 2048, 0)
        return d

def foo(train):
    data = []
    for x in train:
        data.append(x)
        if len(data) >= 100:
            break
    return tabulate(data, n_procs=4)

def agg_token_counts(token_counts, agg_fn=np.sum):
    return {tokenizer: {
        language_source: {
            language: agg_fn(values)
            for language, values in inner.items()
        }
        for language_source, inner in outer.items()
    }
        for tokenizer, outer in token_counts.items()
    }

def tabulate(data, n_procs=10):
    file_counts = defaultdict(Counter)

    # tokenizer, language source, language
    token_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    token_counts_above_1024 = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    token_counts_above_2048 = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    tokens_lost_1024 = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    tokens_lost_2048 = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    guesslang_mismatch = defaultdict(list)

    in_queue, out_queue = JoinableQueue(), JoinableQueue()

    workers = []

    running_count = 0

    for i in range(n_procs):
        # signal to stop
        worker = Worker(i, in_queue, out_queue)
        worker.start()
        running_count += 1
        workers.append(worker)

    for x in tqdm.tqdm(data, ncols=80, desc="loading queue"):
        in_queue.put(x)

    for i in range(n_procs):
        in_queue.put(None)

    file_count = 0
    print(f"running count: {running_count}")
    with tqdm.tqdm(total=len(data), ncols=80, desc="processing") as progress_bar:
        while running_count >= 1:
            r = out_queue.get()
            if r is None:
                running_count -= 1
                print(f"running count: {running_count}")
                out_queue.task_done()
                continue
            datum, tabulated = r
            
            for language_source in ['repo_language', 'guesslang', 'filename_extension']:
                language = tabulated[language_source]
                file_counts[language_source][language] += 1
                #for tokenizer in ['gpt2', 'codet5', 'ours']:
                for tokenizer in ['ours']:
                    token_counts[tokenizer][language_source][language].append(tabulated[f'{tokenizer}_token_count'])
                    token_counts_above_1024[tokenizer][language_source][language].append(tabulated[f'{tokenizer}_above_1024'])
                    token_counts_above_2048[tokenizer][language_source][language].append(tabulated[f'{tokenizer}_above_2048'])
                    tokens_lost_1024[tokenizer][language_source][language].append(tabulated[f'{tokenizer}_lost_1024'])
                    tokens_lost_2048[tokenizer][language_source][language].append(tabulated[f'{tokenizer}_lost_2048'])
            progress_bar.update(1)
            out_queue.task_done()
            file_count += 1
            if file_count % 100000 == 0:
                print(f"total files: {file_count}")
                pprint.pprint(file_counts)
                print("total token counts:")
                pprint.pprint(agg_token_counts(token_counts, np.sum))
                print("mean token counts:")
                pprint.pprint(agg_token_counts(token_counts, np.mean))
                print("percent files above 1024:")
                pprint.pprint(agg_token_counts(token_counts_above_1024, np.mean))
                print("percent files above 2048:")
                pprint.pprint(agg_token_counts(token_counts_above_2048, np.mean))
                print("tokens lost above 1024:")
                pprint.pprint(agg_token_counts(tokens_lost_1024, np.sum))
                print("tokens lost above 2048:")
                pprint.pprint(agg_token_counts(tokens_lost_2048, np.sum))
                print()

        [worker.join() for worker in workers]

    in_queue.join()
    out_queue.join()

        # if tabulated['filename_extension'] == '.py' and tabulated['guesslang'] != 'Python':
        #     guesslang_mismatch[tabulated['guesslang']].append((datum['repo_name'], datum['file_name']))

    return file_counts, token_counts, guesslang_mismatch, token_counts_above_1024, token_counts_above_2048, tokens_lost_1024, tokens_lost_2048

## before converting to not need train dir
#data = load_dataset("code_clippy_dataset", data_dir="scrapes/out_forkless_open-source_10-1/")['train']
#data = load_dataset("code_clippy_dataset", data_dir="scrapes/out_forkless_open-source_10-1_dedup")['train']
#data = load_dataset("code_clippy_dataset", data_dir="scrapes/out_forkless_10-1")['train']

## after converting to not need train dir
# data_dir = "scrapes/out_python_forkless_open-source_10-9/github_data_dedup"

# data_dir = "scrapes/out_python_forkless_10-9/github_data_1"
data_dir = "scrapes/out_python_forkless_10-9/github_data_dedup"
# datasets are hashed based on arguments, which are sensitive to trailing dashes (even though loading is not)
data_dir = data_dir.rstrip("/")
print(data_dir)
data = load_dataset("code_clippy_dataset", data_dir=data_dir)['train']

file_counts, token_counts, guesslang_mismatch, token_counts_above_1024, token_counts_above_2048, tokens_lost_1024, tokens_lost_2048 = tabulate(data, n_procs=10)
print("file counts:")
pprint.pprint(file_counts)
print("total token counts:")
pprint.pprint(agg_token_counts(token_counts, np.sum))
print("median token counts:")
pprint.pprint(agg_token_counts(token_counts, np.median))
print("mean token counts:")
pprint.pprint(agg_token_counts(token_counts, np.mean))
print("percent files above 1024:")
pprint.pprint(agg_token_counts(token_counts_above_1024, np.mean))
print("percent files above 2048:")
pprint.pprint(agg_token_counts(token_counts_above_2048, np.mean))
print("tokens lost above 1024:")
pprint.pprint(agg_token_counts(tokens_lost_1024, np.sum))
print("tokens lost above 2048:")
pprint.pprint(agg_token_counts(tokens_lost_2048, np.sum))
