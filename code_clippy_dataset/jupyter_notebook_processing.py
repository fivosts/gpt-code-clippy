from collections import namedtuple
from markdown import Markdown
from io import StringIO
import random

from code_clippy_dataset.utils import make_tagged

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