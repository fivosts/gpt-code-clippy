from collections import Counter, defaultdict
from os.path import splitext

from datasets import load_dataset
from transformers import GPT2TokenizerFast

from multiprocessing import Pool, Process, JoinableQueue

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
        # guess = Guess()
        guess = None
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", add_prefix_space=ADD_PREFIX_SPACE)

        while True:
            x = self.input_queue.get()
            if x is None:
                self.output_queue.put(None)
                self.input_queue.task_done()
                break
            tabulated = self.tabulate_single(guess, tokenizer, x)
            if self.progress_bar:
                with self.progress_bar.get_lock():
                    self.progress_bar.update(1)
            self.output_queue.put((x, tabulated))
            self.input_queue.task_done()
        # print(f"worker {self.index} ending")

    @staticmethod
    def tabulate_single(guess, tokenizer, x):
        fname_toks = splitext(x['file_name'])
        if len(fname_toks) > 1:
            filename_ext = fname_toks[-1]
        else:
            filename_ext = None
        #guessed_lang = guess.language_name(x['text'])
        guessed_lang = None

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
        if len(data) >= 100:
            break
    return tabulate(data, n_procs=4)

def tabulate(data, n_procs=10):
    file_counts = defaultdict(Counter)
    token_counts = defaultdict(Counter)

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
    with tqdm.tqdm(total=len(data), ncols=80, desc="processing") as progress_bar:
        while running_count >= 1:
            r = out_queue.get()
            if r is None:
                running_count -= 1
                out_queue.task_done()
                continue
            datum, tabulated = r
            
            token_count = tabulated['token_count']
            for language_source in ['repo_language', 'guesslang', 'filename_extension']:
                language = tabulated[language_source]
                file_counts[language_source][language] += 1
                token_counts[language_source][language] += token_count
            progress_bar.update(1)
            out_queue.task_done()
            file_count += 1
            if file_count % 100000 == 0:
                print(f"total files: {file_count}")
                pprint.pprint(file_counts)
                pprint.pprint(token_counts)
                print()

        [worker.join() for worker in workers]

    in_queue.join()
    out_queue.join()

        # if tabulated['filename_extension'] == '.py' and tabulated['guesslang'] != 'Python':
        #     guesslang_mismatch[tabulated['guesslang']].append((datum['repo_name'], datum['file_name']))

    return file_counts, token_counts, guesslang_mismatch

data = load_dataset("code_clippy_dataset", data_dir="scrapes/out_forkless_open-source_10-1/")['train']

file_counts, token_counts, guesslang_mismatch = tabulate(data, n_procs=50)
print("file counts:")
pprint.pprint(file_counts)
print("token counts:")
pprint.pprint(token_counts)
