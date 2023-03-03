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

        value_list[tokenizer.convert_tokens_to_ids(key)] = min(25, value)

    value_array = np.array(value_list).reshape(-1, 1)
    scaler = preprocessing.MinMaxScaler(feature_range=(1, 5))
    normalizedlist = scaler.fit_transform(value_array)

    print(normalizedlist.shape)
    
    value_list=normalizedlist.tolist()
    penalty_list=[round(x[0],3) for x in value_list]
    mask_prob_list=[round((x[0]+0.5)/10,3) for x in value_list]
    
    with open("vocab/loss_weight_normalized.json",'w') as file:
        json.dump(penalty_list,file,indent=4,ensure_ascii=False)

    with open("vocab/mask_probability_normalized.json",'w') as file:
        json.dump(mask_prob_list,file,indent=4,ensure_ascii=False)

if __name__ == "__main__":
    normalize_knowledge_value_fn()
