from transformers import PreTrainedTokenizerFast
from tokenizers import ByteLevelBPETokenizer

SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>"]

# tokenizer = ByteLevelBPETokenizer()
# tokenizer.train_from_iterator(["abc def", "abc\ndef", "abc\n\ndef"] * 100, vocab_size=257+1000, special_tokens=SPECIAL_TOKENS)

pretokenizer_split_newlines_only = False

tokenizer_model = ByteLevelBPETokenizer.from_file(
    "/private/home/dpf/data/github/out_python_forkless_open-source_10-9/github_data_dedup/tokenized_ours/tokenizer_preserve_newlines/vocab.json",
    "/private/home/dpf/data/github/out_python_forkless_open-source_10-9/github_data_dedup/tokenized_ours/tokenizer_preserve_newlines/merges.txt",
    pretokenizer_split_newlines_only=pretokenizer_split_newlines_only,
)
# tokenizer_model = ByteLevelBPETokenizer.from_file(
#     "/private/home/halilakin/src/gpt2_bpe/encoder.json",
#     "/private/home/halilakin/src/gpt2_bpe/vocab.bpe",
# )
tokenizer_model.add_special_tokens(SPECIAL_TOKENS)
tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_model)

with open('dataset_stats_par.py', 'r') as f:
    code_file = f.read()

def printable_tokens(tokens, show_whitespace_chars=True):
    agg = []
    for token in tokens:
        for subtok in token:
            if subtok == 'Ċ':
                agg.append('\n')
        if not show_whitespace_chars:
            token = token.replace('Ċ', '').replace('Ġ', '')
        agg.append(token)
    return agg

input_ids = tokenizer(code_file)['input_ids']
with open('/tmp/ids_sno-False.out', 'w') as f:
    f.write(' '.join(str(id_) for id_ in input_ids))

with open('/tmp/retokenized_sno-False.out', 'w') as f:
    f.write(' '.join(printable_tokens(tokenizer.convert_ids_to_tokens(input_ids), show_whitespace_chars=False)))

print(' '.join(printable_tokens(tokenizer.convert_ids_to_tokens(tokenizer(code_file)['input_ids']), show_whitespace_chars=False)))
