from pathlib import Path
from scipy import sparse
from tqdm.auto import tqdm
from transformers import BertTokenizerFast
import sys
import argparse
import json
import torch
import numpy
from collections import Counter


def knowledge_value_compute_fn():
    with open("vocab/idx_to_word.json", "r") as file:
        idx_to_word = json.loads(file.read())

    with open("vocab/word_to_idx.json", "r") as file:
        word_to_idx = json.loads(file.read())

    with open("vocab/word_freq.json", "r") as file:
        word_freq = json.loads(file.read())

    WCMS = sparse.load_npz("matrix/skip_pmi_matrix.npz")

    tokenizer = BertTokenizerFast.from_pretrained("knowledge_weighted")

    paths = [str(x) for x in Path("wikipedia").glob("**/*.txt")]
    paths.sort()

    segment_size = 10
    knowledge_value_sum = Counter()
    token_freq = Counter()

    for file_path in paths[:100]:

        with open(file_path, "r", encoding="utf-8") as infile:

            doc_list = infile.read().split("\n")

            for doc in tqdm(doc_list):

                sent_token_list = tokenizer.tokenize(doc)

                num_segment = len(sent_token_list) // segment_size

                for s in range(num_segment):

                    segment_token_list = sent_token_list[
                        s * segment_size : (s + 1) * segment_size
                    ]

                    sent_pmi_mat = numpy.zeros(
                        shape=(len(segment_token_list), len(segment_token_list))
                    )

                    for i in range(len(segment_token_list)):

                        for j in range(len(segment_token_list)):

                            if (
                                segment_token_list[i] in word_to_idx
                                and segment_token_list[j] in word_to_idx
                            ):
                                sent_pmi_mat[i, j] = WCMS[
                                    word_to_idx[segment_token_list[i]],
                                    word_to_idx[segment_token_list[j]],
                                ]

                    sent_mat_sum = numpy.sum(sent_pmi_mat, axis=0)

                    value_dict = {}

                    for i in range(len(segment_token_list)):

                        value_dict[segment_token_list[i]] = sent_mat_sum[i]

                    knowledge_value_sum.update(value_dict)
                    token_freq.update(segment_token_list)

    for key, value in knowledge_value_sum.items():

        knowledge_value_sum[key] = value / token_freq[key]

    with open("vocab/word_to_value.json", "w") as file:

        json.dump(dict(knowledge_value_sum), file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    knowledge_value_compute_fn()
