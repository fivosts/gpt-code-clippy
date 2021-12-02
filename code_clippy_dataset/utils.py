import re
import numpy as np
import random
import datasets
import os
from datasets import load_from_disk, load_dataset

BASE_FEATURES = {
    "id": datasets.Value("int64"),
    "text": datasets.Value("string"),
    "repo_name": datasets.Value("string"),
    "file_name": datasets.Value("string"),
    "mime_type": datasets.Value("string"),
    "license": datasets.Value("string"),
    "repo_language": datasets.Value("string"),
}

EXTRA_FEATURES = {
    'github': [
        "stars",
    ],
    'google_code': [
        "ancestorRepo",
        "compressed_size",
        "contentLicense",
        "creationTime",
        "hasSource",
        "imageUrl",
        "labels",
        "logoName",
        "main_common_language",
        "movedTo",
        "percents_by_language",
        "repoType",
        "stars",
        "subrepos",
        "summary",
        "total_sizes_by_language",
        "uncompressed_size",
        "zip_file_size",
    ],
    'bitbucket': [
        "created_on",
        "full_name",
        "language",
        "size",
        "updated_on",
        "uuid",
    ],
    'gitlab': [
        "is_fork",
        "languages",
        "last_activity_at",
        "stars",
        "tags",
        "url",
    ]
}


TOKEN_RE = re.compile(r"\W+")

def strip_trailing_slash(path):
    while path[-1] == '/':
        path = path[:-1]
    return path

def infer_source_from_data_dir(data_dir):
    sources = []
    if 'bigquery' in data_dir:
        sources.append('bigquery')
    if 'github' in data_dir:
        sources.append('github')
    if 'google-code' in data_dir or 'google_code' in data_dir:
        sources.append('google_code')
    if 'bitbucket' in data_dir:
        sources.append('bitbucket')
    if 'gitlab' in data_dir:
        sources.append('gitlab')
    if len(sources) != 1:
        raise ValueError(f"could not infer source from path {data_dir}")
    return sources[0]

def load_dataset_infer(data_dir):
    if os.path.exists(os.path.join(data_dir, "dataset.arrow")):
        dataset = load_from_disk(data_dir)
    else:
        source = infer_source_from_data_dir(data_dir)
        if source == 'bigquery':
            dataset = datasets.load_dataset("code_clippy_dataset/bigquery_dataset.py", data_dir=data_dir, split="train")
        else:
            dataset = datasets.load_dataset("code_clippy_dataset", data_dir=data_dir, split="train", source=source)
    return dataset

def standardize(dataset, verbose=True):
    """ remove all source-specific columns, keeping only those that occur in all repo sources.
    also adds extra columns with default values """

    found = False
    for source, extra_features in EXTRA_FEATURES.items():
        if all(feat in dataset.features for feat in extra_features):
            found = True
            break
    assert found, f"unable to detect dataset type for features {dataset.features}"

    features_to_add_and_defaults = {'stars': '-1', 'source': source}
    features_to_keep = set(BASE_FEATURES.keys()) | set(features_to_add_and_defaults.keys())

    features_to_remove = [
        feature for feature in dataset.features.keys()
        if feature not in features_to_keep
    ]
    if verbose:
        print(f"removing features {features_to_remove}")
    dataset = dataset.remove_columns(features_to_remove)

    features_to_add = {
        k: v for k, v in features_to_add_and_defaults.items()
        if k not in dataset.features
    }
    if verbose:
        print(f"adding features with defaults: {features_to_add}")
    N = len(dataset)
    # could also do this with a map call but it's much slower
    for feat, value in features_to_add:
        values = np.full((N,), value)
        dataset = dataset.add_column(feat, values)
    if verbose:
        print(f"resulting dataset features: {dataset.features}")
    return dataset


def make_tagged(tag, inner, attributes={}, insert_newlines=True, attribute_drop_probability=None):
    if attributes:
        attr_strs = [f'{k}={v}' for k, v in attributes.items()]
        if attribute_drop_probability is not None:
            assert 0 <= attribute_drop_probability <= 1.0
            attr_strs = [x for x in attr_strs if random.random() > attribute_drop_probability]
    else:
        attr_strs = []
    if attr_strs:
        random.shuffle(attr_strs)
        attr_string = f" {' '.join(attr_strs)}"
    else:
        attr_string = ''
    if insert_newlines:
        return f'<{tag}{attr_string}>\n{inner}\n</{tag}>'
    else:
        return f'<{tag}{attr_string}>{inner}</{tag}>'