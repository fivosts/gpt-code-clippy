from transformers import PreTrainedTokenizerFast
from tokenizers import ByteLevelBPETokenizer

#tokenizer = ByteLevelBPETokenizer()
#tokenizer.train_from_iterator(["abc def", "abc\ndef", "abc\n\ndef"] * 100, vocab_size=257+1000)

tokenizer_model = ByteLevelBPETokenizer.from_file(
    "scrapes/out_python_forkless_open-source_10-9/github_data_dedup/tokenizer_preserve_newlines/vocab.json",
    "scrapes/out_python_forkless_open-source_10-9/github_data_dedup/tokenizer_preserve_newlines/merges.txt",
)
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

# with open('/tmp/retokenized.out', 'w') as f:
#     f.write(' '.join(printable_tokens(tokenizer.convert_ids_to_tokens(tokenizer(code_file)['input_ids']), show_whitespace_chars=False)))


# print(' '.join(printable_tokens(tokenizer.convert_ids_to_tokens(tokenizer(code_file)['input_ids']), show_whitespace_chars=False)))
