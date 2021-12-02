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
from json.decoder import JSONDecodeError
from typing import List
import json
import jsonlines
import gzip
from pathlib import Path
import os.path

import datasets

from hacky_linguist import LANGUAGE_EXTENSIONS
from code_clippy_dataset.jupyter_notebook_processing import DEFAULT_JUPYTER_OPTIONS_NO_OUTPUTS, notebook_to_text

class BigQueryConfig(datasets.BuilderConfig):
    """BuilderConfig for CodeClippy."""

    def __init__(self, language_filter_type=None, licenses_filter=None, jupyter_options=DEFAULT_JUPYTER_OPTIONS_NO_OUTPUTS, **kwargs):
        """BuilderConfig for CodeClippy.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(BigQueryConfig, self).__init__(**kwargs)

        # TODO: implement this
        self.language_filter_type = language_filter_type
        # if self.language_filter_type not in (None, 'guesslang', 'repo_language', 'filename_extension'):
        #     raise NotImplementedError(f"invalid language_filter_type {self.language_filter_type}")

        self.licenses_filter = licenses_filter
        self.jupyter_options = jupyter_options

class BigQuery(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.0")

    BUILDER_CONFIG_CLASS = BigQueryConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    # BUILDER_CONFIGS = [
    #     datasets.BuilderConfig(name="first_domain", version=VERSION, description="This part of my dataset covers a first domain"),
    #     datasets.BuilderConfig(name="second_domain", version=VERSION, description="This part of my dataset covers a second domain"),
    # ]

    # DEFAULT_CONFIG_NAME = "first_domain"


    def _info(self):
        features = {
            # numeric id, from iteration
            "id": datasets.Value("int64"),
            # id from the BigQuery table
            "github_id": datasets.Value("string"),
            # "binary": datasets.Value("bool"),
            #"content": datasets.Value("string"),
            "text": datasets.Value("string"),
            # "copies": datasets.Value("int64"),
            "license": datasets.Value("string"),
            # "mode": datasets.Value("int64"),
            "path": datasets.Value("string"),
            "file_name": datasets.Value("string"),
            # "ref": datasets.Value("string"),
            "repo_name": datasets.Value("string"),
            "size": datasets.Value("int64"),
        }
        features = datasets.Features(features)
        return datasets.DatasetInfo(
            description=None,
            features=features,
            homepage=None,
            license=None,
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
                            for fp in Path(f"{data_dir}/").glob("*.jsonl.gz")
                        ]
                    )
                },
            ),
        ]

    def _generate_examples(self, filepaths: List):
        """Yields examples as (key, example) tuples."""
        id_ = 0
        num_key_errors = 0
        num_json_errors = 0
        num_missing_content = 0
        other_errors = []
        for filepath in filepaths:
            with gzip.open(filepath, "rb") as f:
                f = jsonlines.Reader(f)
                for record in f:
                    if record['binary']:
                        continue
                    path = record['path']
                    filename = os.path.basename(path)
                    _, extension = os.path.splitext(filename)

                    if 'content' not in record:
                        num_missing_content += 1
                        continue
                    text = record['content']

                    valid = True
                    if extension == '.ipynb':
                        try:
                            notebook_dictionary = json.loads(text)
                            text = notebook_to_text(notebook_dictionary, self.config.jupyter_options)
                        except KeyError as e:
                            valid = False
                            num_key_errors += 1
                            # print(f"KeyError({e})")
                        except JSONDecodeError as e:
                            valid = False
                            num_json_errors += 1
                        except Exception as e:
                            valid = False
                            other_errors.append(e)

                    if extension in LANGUAGE_EXTENSIONS and valid:
                        yield id_, {
                            "id": id_, 
                            "github_id": record["id"],
                            "text": text,
                            "license": record["license"],
                            "path": path,
                            "file_name": filename,
                            "repo_name": record["repo_name"],
                            "size": int(record["size"]),
                        }
                        id_ += 1
        print(f"{num_missing_content} records with missing content")
        print(f"{num_json_errors} json errors")
        print(f"{num_json_errors} key errors")
        print(f"{len(other_errors)} other errors")