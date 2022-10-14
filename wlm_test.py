import unittest
from get_dataset import loaddataset
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from build_tokenizer import get_tokenizer, get_tok_encoder


def wiki_dataset(dataset_name):
    dataset = loaddataset(dataset_name)
    dataset["train"][0]
    wiki_length = len(dataset["train"])
    first_record = len(dataset["train"][0])
    first_record_key = list(dataset["train"][0].keys())[0]
    return {
        "total_length": wiki_length,
        "first_record": first_record,
        "first_record_key": first_record_key,
    }


def tokenizer_bert(query):
    paths, tokenizer = get_tokenizer()
    tk1 = tokenizer.pre_tokenizer.pre_tokenize_str(query)
    length_tk1 = len(tk1)
    first_key = tk1[0][0]
    first_key_values = tk1[0][1]
    first_key_value = tk1[0][1][1]
    return {
        "total_length": length_tk1,
        "first_key": first_key,
        "first_key_values": first_key_values,
        "first_key_value": first_key_value,
    }


def encoder_bert(encoder_query_1, encoder_query_2):
    paths, tokenizer = get_tokenizer()
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.WordPieceTrainer(vocab_size=10000, special_tokens=special_tokens)
    tokenizer = get_tok_encoder(paths[:100], tokenizer, trainer)
    encoding = tokenizer.encode(encoder_query_1, encoder_query_2)
    cls_token = encoding.tokens[0]
    sep_token = encoding.tokens[43]
    length_tokens = len(encoding.tokens)
    return {
        "cls_token": cls_token,
        "sep_token": sep_token,
        "length_tokens": length_tokens,
    }


class DataSet:
    def __init__(self):
        self.dataset_name = "wikipedia"
        self.query = "This is an example!"
        self.encoder_query_1 = "This is one sentence."
        self.encoder_query_2 = "With this one we have a pair."

    def data_check(self):
        self.dataset_name = "wikipedia-t"
        self.query = "##"
        self.encoder_query_1 = "This is one sentence."
        self.encoder_query_2 = "With this one we have a pair."


class wlm_unittest(unittest.TestCase):
    def setUp(self):
        self.data = DataSet()

    def test_get_dataset(self):
        actual = wiki_dataset(self.data.dataset_name)
        expected = {
            "total_length": 210000,
            "first_record": 1,
            "first_record_key": "text",
        }
        self.assertEqual(actual, expected)

    def test_tokenizer(self):
        actual = tokenizer_bert(self.data.query)
        expected = {
            "total_length": 5,
            "first_key": "This",
            "first_key_values": (0, 4),
            "first_key_value": 4,
        }
        self.assertEqual(actual, expected)

    def test_encoder(self):
        actual = encoder_bert(self.data.encoder_query_1, self.data.encoder_query_2)
        expected = {"cls_token": "[CLS]", "sep_token": "[SEP]", "length_tokens": 44}
        self.assertEqual(actual, expected)

    def test_get_dataset_exception(self):
        self.data.data_check()
        with self.assertRaises(ValueError) as exception_context:
            wiki_dataset(self.data.dataset_name)
        self.assertEqual(str(exception_context.exception), "unable to download dataset")
