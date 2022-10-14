from transformers import BertTokenizerFast
import sys
import json
import torch
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt


def normalize_knowledge_value_fn():
    tokenizer = BertTokenizerFast.from_pretrained("knowledge_weighted")

    with open("vocab/word_to_value.json", "r") as file:

        word_to_value = json.load(file)

    value_list = [0] * len(tokenizer.vocab)

    for key, value in word_to_value.items():

        value_list[tokenizer.convert_tokens_to_ids(key)] = min(20, value)

    value_array = np.array(value_list).reshape(-1, 1)
    scaler = preprocessing.MinMaxScaler(feature_range=(1, 10))
    normalizedlist = scaler.fit_transform(value_array)

    print(normalizedlist.shape)

    with open("vocab/word_to_value_normalized.json", "w") as file:

        json.dump(normalizedlist.tolist(), file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    normalize_knowledge_value_fn()
