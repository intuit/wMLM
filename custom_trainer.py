import torch
from torch.utils import data
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader, Dataset
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data.distributed import DistributedSampler
import json


class BaselineTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_data = self.train_dataset

        sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            shuffle=False,
            drop_last=True,
        )

        return torch.utils.data.DataLoader(
            train_data, batch_size=self.args.train_batch_size, sampler=sampler
        )

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:

        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        return torch.utils.data.DataLoader(
            eval_dataset, batch_size=self.args.eval_batch_size, shuffle=False
        )

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:

        if is_datasets_available() and isinstance(test_dataset, datasets.Dataset):
            test_dataset = self._remove_unused_columns(test_dataset, description="test")

        return torch.utils.data.DataLoader(
            test_dataset, batch_size=self.args.eval_batch_size, shuffle=False
        )


class WeightedLossTrainer(Trainer):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        with open("vocab/word_to_value_normalized.json", "r") as file:

            self.vocab_weight = json.load(file)

    def get_train_dataloader(self) -> DataLoader:

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_data = self.train_dataset

        sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            shuffle=False,
            drop_last=True,
        )

        return torch.utils.data.DataLoader(
            train_data, batch_size=self.args.train_batch_size, sampler=sampler
        )

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:

        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        return torch.utils.data.DataLoader(
            eval_dataset, batch_size=self.args.eval_batch_size, shuffle=False
        )

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:

        if is_datasets_available() and isinstance(test_dataset, datasets.Dataset):
            test_dataset = self._remove_unused_columns(test_dataset, description="test")

        return torch.utils.data.DataLoader(
            test_dataset, batch_size=self.args.eval_batch_size, shuffle=False
        )

    def compute_loss(self, model, inputs, return_outputs=False):

        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(self.vocab_weight, device=self.args.device).flatten()
        )
        loss = loss_fct(logits.view(-1, self.model.config.vocab_size), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
