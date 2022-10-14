from audioop import reverse
from pathlib import Path
from collections import Counter, OrderedDict
import nltk
import scipy
import random
from tqdm import tqdm
import json
import functools
from nltk.util import skipgrams
from sklearn.feature_extraction.text import CountVectorizer


def build_pmi_matrix_fn():
    with open("vocab/idx_to_word.json", "r") as file:
        idx_to_word = json.loads(file.read())

    with open("vocab/word_to_idx.json", "r") as file:
        word_to_idx = json.loads(file.read())

    with open("vocab/word_freq.json", "r") as file:
        word_freq = json.loads(file.read())

    WCM = scipy.sparse.load_npz("matrix/word_matrix_10.npz")

    print(WCM.nnz)

    idxw1 = 23
    idxw2 = 44
    print(WCM[idxw1, idxw2])

    nwords = WCM.get_shape()[0]
    print(nwords)

    import math

    row_sum_skip = WCM.sum(axis=0)
    col_sum_skip = WCM.sum(axis=1)
    tot_sum_skip = WCM.sum()

    print(row_sum_skip.shape)
    print(col_sum_skip.shape)
    print(tot_sum_skip.shape)

    for i in tqdm(range(nwords)):
        for j in range(WCM.indptr[i], WCM.indptr[i + 1]):
            den = (row_sum_skip[0, i] * col_sum_skip[WCM.indices[j], 0]) / tot_sum_skip
            score = math.log(WCM.data[j] / den)
            WCM.data[j] = score

    print(WCM.nnz)

    scipy.sparse.save_npz("matrix/skip_pmi_matrix.npz", WCM.asformat(format="csr"))

    print(WCM[word_to_idx["lord"], word_to_idx["voldemort"]])
    print(WCM[word_to_idx["voldemort"], word_to_idx["lord"]])


if __name__ == "__main__":
    build_pmi_matrix_fn()
