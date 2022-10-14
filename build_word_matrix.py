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
from transformers import BertTokenizerFast
import ngram_calc
import os


def get_token_list(bert_token_list):
    token_list = []

    for bert_token in bert_token_list:
        if bert_token.startswith("##"):
            token_list[-1] = token_list[-1] + bert_token[2:]
        else:
            token_list.append(bert_token)

    return token_list


def build_word_matrix_fn():
    if not os.path.exists("matrix"):
        os.makedirs("matrix")

    paths = [str(x) for x in Path("wikipedia").glob("**/*.txt")]

    with open("vocab/idx_to_word.json", "r") as file:
        idx_to_word = json.loads(file.read())

    with open("vocab/word_to_idx.json", "r") as file:
        word_to_idx = json.loads(file.read())

    with open("vocab/word_freq.json", "r") as file:
        word_freq = json.loads(file.read())

    tokenizer = BertTokenizerFast.from_pretrained("knowledge_weighted")

    nwords = len(word_to_idx.keys())
    print(nwords)

    skip_distance = 10
    WCMF = scipy.sparse.csr_matrix((nwords, nwords))

    block_count = 0
    WCM = scipy.sparse.lil_matrix((nwords, nwords))

    glob_loop_dict = Counter()

    iter_count = 0

    for file_path in tqdm(paths):

        with open(file_path, "r", encoding="utf-8") as file:

            line_list = file.readlines()

            for line in line_list:

                bert_token_list = tokenizer.tokenize(line.strip())
                token_list = get_token_list(bert_token_list)
                loop_dict = ngram_calc.ngram_calc(
                    token_list, len(token_list), skip_distance
                )
                glob_loop_dict.update(loop_dict)

                if block_count % 100 == 0:

                    for key, value in glob_loop_dict.items():
                        (w1, w2) = key
                        if w1 in word_to_idx and w2 in word_to_idx:
                            WCM[word_to_idx[w1], word_to_idx[w2]] += value
                            WCM[word_to_idx[w2], word_to_idx[w1]] += value

                    WCMF += WCM.asformat(format="csr")
                    WCM = scipy.sparse.lil_matrix((nwords, nwords))
                    glob_loop_dict = Counter()

                block_count += 1

        if iter_count % 20 == 0:
            scipy.sparse.save_npz(
                "matrix/word_matrix_" + str(skip_distance) + ".npz", WCMF
            )

        iter_count += 1

    print(WCMF.nnz)

    scipy.sparse.save_npz(
        "matrix/word_matrix_" + str(skip_distance) + ".npz", WCMF.asformat(format="csr")
    )


if __name__ == "__main__":
    build_word_matrix_fn()
