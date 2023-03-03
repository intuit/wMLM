from pathlib import Path
from transformers import BertTokenizerFast
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from transformers import Trainer, TrainingArguments
from custom_dataset import Dataset
from custom_trainer import BaselineTrainer, WeightedLossTrainer
from transformers import TrainingArguments
import argparse


def train_lm_fn():
    parser = argparse.ArgumentParser(description="Knowledge Enhanced BERT training")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument(
        "--loss_type", type=str, default="weighted", choices=["regular", "weighted"]
    )
    parser.add_argument(
        "--mask_type", type=str, default="inform_mask",choices=["random_mask", "inform_mask"]
    )
    args = parser.parse_args()

    document_length = 128
    batch_size = 36
    num_layer = 12
    loss_type = args.loss_type

    tokenizer = BertTokenizerFast.from_pretrained("knowledge_weighted")

    train_data = Dataset("wikipedia",doc_len=document_length,masking_type=args.mask_type)

    config = RobertaConfig(
        vocab_size=100000,
        max_position_embeddings=document_length + 2,
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=num_layer,
        type_vocab_size=2,
    )

    model = RobertaForMaskedLM(config=config)

    print("Num param: ", model.num_parameters())
    model_path = "./models/" + loss_type
    training_args = TrainingArguments(
        output_dir=model_path,
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=batch_size,
        save_strategy="epoch",
        # save_steps=10000,
        # save_total_limit=1,
        prediction_loss_only=True,
        local_rank=args.local_rank,
        fp16=True,
    )
    if loss_type == "regular":

        trainer = BaselineTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
        )
    else:
        trainer = WeightedLossTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
        )
    trainer.train()
    #trainer.train(resume_from_checkpoint=True)


if __name__ == "__main__":
    train_lm_fn()
