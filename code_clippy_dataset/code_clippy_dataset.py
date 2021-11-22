# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the CodeClippy team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
CodeClippy dataset - opensource code from Github. Scrapped July 7 2021.
More to add here.
"""

import io
from typing import List
import jsonlines
import zstandard as zstd
from pathlib import Path
import os.path

import datasets

from hacky_linguist import LANGUAGE_EXTENSIONS


# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """"""

_DESCRIPTION = ""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here (once we have those)

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


class CodeClippyConfig(datasets.BuilderConfig):
    """BuilderConfig for CodeClippy."""

    def __init__(self, language_filter_type=None, licenses_filter=None, source='github', **kwargs):
        """BuilderConfig for CodeClippy.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(CodeClippyConfig, self).__init__(**kwargs)

        # TODO: implement this
        self.language_filter_type = language_filter_type
        # if self.language_filter_type not in (None, 'guesslang', 'repo_language', 'filename_extension'):
        #     raise NotImplementedError(f"invalid language_filter_type {self.language_filter_type}")

        self.licenses_filter = licenses_filter
        self.source = source

class CodeClippy(datasets.GeneratorBasedBuilder):
    """CodeClippy dataset - opensource code from Github. Scrapped July 7 2021."""

    VERSION = datasets.Version("0.1.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    BUILDER_CONFIG_CLASS = CodeClippyConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    # BUILDER_CONFIGS = [
    #     datasets.BuilderConfig(name="first_domain", version=VERSION, description="This part of my dataset covers a first domain"),
    #     datasets.BuilderConfig(name="second_domain", version=VERSION, description="This part of my dataset covers a second domain"),
    # ]

    # DEFAULT_CONFIG_NAME = "first_domain"


    def _info(self):
        features = BASE_FEATURES.copy()
        print(f"self.config.source: {self.config.source}")
        for feature in EXTRA_FEATURES[self.config.source]:
            features[feature] = datasets.Value("string")
        print(features.keys())
        features = datasets.Features(features)
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = self.config.data_dir
        return [datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepaths": sorted(
                        [
                            str(fp)
                            for fp in Path(f"{data_dir}/").glob("*.jsonl.zst")
                        ]
                    )
                },
            ),
        ]

    def _generate_examples(self, filepaths: List):
        """Yields examples as (key, example) tuples."""
        id_ = 0
        dctx = zstd.ZstdDecompressor()
        for filepath in filepaths:
            with open(filepath, "rb") as f:
                f = dctx.stream_reader(f)
                f = io.TextIOWrapper(f, encoding="utf-8")
                f = jsonlines.Reader(f)
                for line in f:
                    meta = line["meta"]
                    filename = meta["file_name"]
                    _, extension = os.path.splitext(filename)

                    # column renames
                    if "main_language" in meta and "repo_language" not in meta:
                        meta["repo_language"] = meta["main_language"]
                        del meta["main_language"]
                    if "stargazers" in meta:
                        meta["stars"] = meta["stargazers"]
                        del meta["stargazers"]

                    if "name" in meta:
                        if "repo_name" in meta["repo_name"]:
                            assert meta["repo_name"] == meta["name"]
                        else:
                            meta["repo_name"] = meta["name"]
                        del meta["name"]

                    if extension in LANGUAGE_EXTENSIONS:
                        if "detected_licenses" in meta:
                            # if multiple licenses, just concatenate them
                            # TODO: do something smarter if we want to do all/any filtering on particular license types
                            license = "+".join(meta["detected_licenses"])
                            del meta["detected_licenses"]
                            meta["license"] = license
                        yield id_, {"id": id_, "text": line["text"], **meta}
                        id_ += 1