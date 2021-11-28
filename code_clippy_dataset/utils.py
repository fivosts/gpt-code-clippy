import re
import numpy as np
from code_clippy_dataset.code_clippy_dataset import BASE_FEATURES, EXTRA_FEATURES

TOKEN_RE = re.compile(r"\W+")

UNLIMITED_LINE_LENGTH_EXTENSIONS = {'.ipynb'}

def strip_trailing_slash(path):
    while path[-1] == '/':
        path = path[:-1]
    return path

def infer_source_from_data_dir(data_dir):
    sources = []
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
