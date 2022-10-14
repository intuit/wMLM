from audioop import reverse
from pathlib import Path
from collections import Counter, OrderedDict
import nltk
import scipy
import random
from tqdm import tqdm
from transformers import BertTokenizerFast
import functools
from nltk.util import skipgrams
from sklearn.feature_extraction.text import CountVectorizer


def get_token_list(bert_token_list):
    token_list = []

    for i, bert_token in enumerate(bert_token_list):
        if bert_token.startswith("##"):
            token_list[-1] = token_list[-1] + bert_token[2:]
        else:
            token_list.append(bert_token)

    return token_list


def prep_vocab_fn():
    word_to_idx = {}
    idx_to_word = {}

    word_freq = Counter()
    tokenizer = BertTokenizerFast.from_pretrained("knowledge_weighted")

    paths = [str(x) for x in Path("wikipedia").glob("**/*.txt")]

    for file_path in tqdm(paths):
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                text = line.strip()
                bert_token_list = tokenizer.tokenize(text)
                token_list = get_token_list(bert_token_list)
                word_freq.update(Counter(token_list))

    word_freq_list = word_freq.most_common(100000)
    word_list = [word[0] for word in word_freq_list]

    print(word_freq.most_common(10))

    word_to_idx = {}
    idx_to_word = {}

    for i, word in enumerate(word_list):
        word_to_idx[word] = i
        idx_to_word[i] = word

    nwords = len(word_list)

    idxw1 = 23
    idxw2 = 48
    print(idx_to_word[idxw1], idx_to_word[idxw2])
    print(idx_to_word[idxw1], idx_to_word[idxw2])

    import json
    import os

    if not os.path.exists("vocab"):
        os.makedirs("vocab")

    word_freq_new = {}

    for key, value in word_to_idx.items():
        word_freq_new[key] = word_freq[key]

    with open("vocab/idx_to_word.json", "w", encoding="utf-8") as file:
        json.dump(idx_to_word, file, indent=4, ensure_ascii=False)

    with open("vocab/word_to_idx.json", "w", encoding="utf-8") as file:
        json.dump(word_to_idx, file, indent=4, ensure_ascii=False)

    with open("vocab/word_freq.json", "w", encoding="utf-8") as file:
        json.dump(word_freq_new, file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    prep_vocab_fn()
