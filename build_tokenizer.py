from pathlib import Path
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from transformers import BertTokenizerFast


def build_tokenizer_fn():
    paths, tokenizer = get_tokenizer()
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.WordPieceTrainer(
        vocab_size=100000, special_tokens=special_tokens
    )
    tokenizer = get_tok_encoder(paths, tokenizer, trainer)
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    new_tokenizer = BertTokenizerFast(tokenizer_object=tokenizer)
    new_tokenizer.save_pretrained("knowledge_weighted")


def get_tok_encoder(paths, tokenizer, trainer):
    tokenizer.train(files=paths, trainer=trainer)
    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]")
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", cls_token_id),
            ("[SEP]", sep_token_id),
        ],
    )
    return tokenizer


def get_tokenizer():
    tokenizer = Tokenizer(models.WordPiece(unl_token="[UNK]"))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
    paths = [str(x) for x in Path("wikipedia").glob("**/*.txt")]
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    return paths, tokenizer


if __name__ == "__main__":
    build_tokenizer_fn()
