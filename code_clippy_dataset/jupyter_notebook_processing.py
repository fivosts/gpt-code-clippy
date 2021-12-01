from collections import namedtuple
import functools
from markdown import Markdown
from io import StringIO
import random

from utils import make_tagged

# https://stackoverflow.com/a/54923798/1319683
def unmark_element(element, stream=None):
    if stream is None:
        stream = StringIO()
    if element.text:
        stream.write(element.text)
    for sub in element:
        unmark_element(sub, stream)
    if element.tail:
        stream.write(element.tail)
    return stream.getvalue()

# patching Markdown
Markdown.output_formats["plain"] = unmark_element
__md = Markdown(output_format="plain")
__md.stripTopLevelTags = False

def unmark(text):
    return __md.convert(text)

JupyterOptions = namedtuple("JupyterOptions", [
    # Set[str]. include cells if they have a celltype in this set. Options: {'code', 'markdown', 'raw'}
    "cell_types", 
    # Set[str]. include outputs if they have a type in this set. Options: {'display_data', 'execute_result'}
    "include_output_types",
    # Set[str]. include data from outputs if they have a mime type in this set. Common options: {'text/plain', 'text/html', 'image/png', 'application/javascript'}. 
    # 'application/javascript' seem to be executed in the notebook and are often the output of magic functions or %matplotlib notebook
    "include_output_data_types",
    "strip_markdown",
    "attribute_drop_probability", 
    ])

Cell = namedtuple("Cell", ["cell_type", "source", "meta"])

Output = namedtuple("Output", ["output_type", "filtered_data"])

DEFAULT_JUPYTER_OPTIONS_NO_OUTPUTS = JupyterOptions(cell_types={'code', 'markdown'}, include_output_types={}, include_output_data_types={}, strip_markdown=True, attribute_drop_probability=0.01)
DEFAULT_JUPYTER_OPTIONS_WITH_OUTPUTS = JupyterOptions(cell_types={'code', 'markdown'}, include_output_types={'execute_result'}, include_output_data_types={'text/plain'}, strip_markdown=True, attribute_drop_probability=0.01)

language_standardization = {
    'python2': 'python',
    'python3': 'python',
}

def notebook_iterator(notebook_dictionary, options: JupyterOptions):
    language = notebook_dictionary['metadata']['kernelspec']['name']
    language = language_standardization.get(language, language)
    meta = {
        'language': language
    }
    for cell in notebook_dictionary['cells']:
        cell_type = cell['cell_type']
        if cell_type not in options.cell_types:
            continue
        # TODO: there is also cell metadata which we may want to include if it has more than just an 'id' field. right now all metadata is from the notebook
        yield Cell(cell_type, cell['source'], meta)
        for output in cell.get('outputs', []):
            output_type = output['output_type']
            if output_type not in options.include_output_types:
                continue
            data = output.get('data', {})
            filtered_data = {k: v for k, v in data.items() if k in options.include_output_data_types}
            if filtered_data:
                yield Output(output_type, filtered_data)

def notebook_to_text(notebook_dictionary, options: JupyterOptions):
    strings = []
    for item in notebook_iterator(notebook_dictionary, options):
        if isinstance(item, Cell):
            if item.cell_type == 'code':
                if 'language' in item.meta:
                    attributes = {'language': item.meta['language']}
                else:
                    attributes = {}
                strings.append(make_tagged('code', ''.join(item.source), attributes, insert_newlines=True, attribute_drop_probability=options.attribute_drop_probability))
            elif item.cell_type == 'markdown':
                text = ''.join(item.source)
                if options.strip_markdown:
                    text = unmark(text)
                strings.append(make_tagged('text', text, {}, insert_newlines=True, attribute_drop_probability=options.attribute_drop_probability))
            else:
                raise NotImplementedError(f"invalid cell_type {item.cell_type}")
        elif isinstance(item, Output):
            for data_type, data in item.filtered_data.items():
                strings.append(make_tagged('output', '\n'.join(data), {'type': data_type}, attribute_drop_probability=options.attribute_drop_probability))
    return '\n'.join(strings)


def process_file(filename, output_dir, jupyter_options):
    import zstandard as zstd
    import io
    import jsonlines
    import json
    import os
    dctx = zstd.ZstdDecompressor()

    basename = os.path.basename(filename)

    cctx = zstd.ZstdCompressor(level=3, threads=4)

    total_records = 0
    valid_records = 0

    out_path = os.path.join(output_dir, basename)
    assert out_path != filename, "out_path: {out_path}"
    
    with open(filename, "rb") as f_in, open(out_path, 'wb') as f_out:
        compressor = cctx.stream_writer(f_out)
        def write_record(record, flush=False):
            compressor.write(json.dumps(record).encode('UTF-8') + b'\n')
            if flush:
                compressor.flush(zstd.FLUSH_FRAME)
                f_out.flush()

        f = dctx.stream_reader(f_in)
        f = io.TextIOWrapper(f, encoding="utf-8")
        f = jsonlines.Reader(f)
        for record_ix, record in enumerate(f):
            meta = record["meta"]
            fname = meta["file_name"]
            print(fname)
            _, extension = os.path.splitext(fname)
            text = record["text"]
            valid = True
            if extension == '.ipynb':
                try:
                    notebook_dictionary = json.loads(text)
                    text = notebook_to_text(notebook_dictionary, jupyter_options)
                except Exception as e:
                    valid = False
                record['notebook_processed'] = True
                record["text"] = text
            if valid:
                write_record(record, flush=(record_ix+1)%1000==0)
                valid_records += 1
            total_records += 1
    print(f"{filename}: {valid_records} / {total_records} = {valid_records/total_records*100:.2f}%")
    return (filename, valid_records, total_records)

if __name__ == "__main__":
    import argparse
    import glob
    from multiprocessing import Pool
    import tqdm
    import os
    import sys
    import pprint

    print(' '.join(sys.argv))

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--num_procs", type=int, default=20)

    args = parser.parse_args()
    pprint.pprint(vars(args))

    input_files = glob.glob(os.path.join(args.input_dir, "*.jsonl.zst"))
    os.makedirs(args.output_dir, exist_ok=True)

    fn = functools.partial(process_file, output_dir=args.output_dir, jupyter_options=DEFAULT_JUPYTER_OPTIONS_NO_OUTPUTS)

    with Pool(args.num_procs) as p:
        processed = tqdm.tqdm(p.imap(fn, input_files), total=len(input_files), ncols=80)

    print("**complete**")

    for filename, valid_records, total_records in processed:
        print(f"{filename}: {valid_records} / {total_records} = {valid_records/total_records*100:.2f}%")