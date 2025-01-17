# -*- coding: utf-8 -*-

import glob
import orjson
import os

import datasets
from itertools import islice

_DESCRIPTION = """
An open-source pretraining dataset containing 4690 billion tokens,
this bilingual dataset with both English and Chinese texts is used for training neo models.

This can split files when the number of files is small (the storage of each file is relatively large), 
adding artificial shards.
"""

_CITATION = """
@article{zhang2024mapneo,
    title   = {MAP-Neo: Highly Capable and Transparent Bilingual Large Language Model Series},
    author  = {
        Ge Zhang and
        Scott Qu and
        Jiaheng Liu and
        Chenchen Zhang and
        Chenghua Lin and
        Chou Leuang Yu and
        Danny Pan and
        Esther Cheng and
        Jie Liu and
        Qunshu Lin and
        Raven Yuan and
        Tuney Zheng and
        Wei Pang and
        Xinrun Du and
        Yiming Liang and
        Yinghao Ma and
        Yizhi Li and
        Ziyang Ma and
        Bill Lin and
        Emmanouil Benetos and
        Huan Yang and
        Junting Zhou and
        Kaijing Ma and
        Minghao Liu and
        Morry Niu and
        Noah Wang and
        Quehry Que and
        Ruibo Liu and
        Sine Liu and
        Shawn Guo and
        Soren Gao and
        Wangchunshu Zhou and
        Xinyue Zhang and
        Yizhi Zhou and
        Yubo Wang and
        Yuelin Bai and
        Yuhan Zhang and
        Yuxiang Zhang and
        Zenith Wang and
        Zhenzhu Yang and
        Zijian Zhao and
        Jiajun Zhang and
        Wanli Ouyang and
        Wenhao Huang and
        Wenhu Chen
    },
    year    = {2024},
    journal = {arXiv preprint arXiv: 2405.19327}
}
"""

_HOMEPAGE = "https://huggingface.co/datasets/m-a-p/Matrix"


class MatrixDataset(datasets.GeneratorBasedBuilder):
    """Custom dataset for JSON files with filtering capabilities."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "id": datasets.Value("string"),
                "text": datasets.Value("string"),
            }),
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        import random

        data_files = glob.glob("*/*.jsonl")
        data_shards = []
        for filepath in data_files:
            # max size of each shard is 1GB
            num_shards = -os.path.getsize(filepath) // -1024**3
            for i in range(num_shards):
                data_shards.append((filepath, i, num_shards))
        random.Random(42).shuffle(data_shards)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_shards": data_shards,
                },
            ),
        ]

    def _generate_examples(self, data_shards):
        for file, split, num_shards in data_shards:
            with open(file, "r") as f:
                for i, line in islice(enumerate(f), split, None, num_shards):
                    data = orjson.loads(line)
                    if 'id' not in data:
                        data['id'] = f"{file}_{i}"
                    if 'content' in data and 'text' not in data:
                        data['text'] = data.pop('content')
                    if data['text'] is not None:
                        yield data["id"], data