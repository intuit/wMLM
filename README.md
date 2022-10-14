# Knowledge Optimized Pre-training

This repository contains the pre-training code for a knowledge enhanced language model. 

## Installation

Clone the repository
```
git clone https://github.intuit.com/data-mlplatform/kelm.git
cd kelm
```

Install the required libraries using pip within your virtual environment

```
pip install numpy scipy scikit-learn tqdm datasets torch transformers tensorboard nltk matplotlib
```

Build cython script
```
python setup_cython.py build_ext --inplace
```

## Dataset Download
We will perform pre-training using Wikipedia dataset available via datasets library. Download itself requires 37GB of storage and default download location may not have that much space. We use a custom download location called 'dataset_cache' within the project root directory.

```
mkdir dataset_cache
python get_dataset.py
```

## Preprocessing
We need to create all the resources required for our proposed pre-training approach. The steps include building a custom tokenizer, preparing vocbulary files for informative words, PMI matrix preparation and token specific knowledge value computation.

```
python build_tokenizer.py
python prepare_vocab.py
python build_word_matrix.py
python build_pmi_matrix.py	
python knowledge_value_compute.py
python normalize_knowledge_value.py
```

## Training 

We performed pre-training using AWS p3.8xlarge instance with 4 Nvidia V100 GPUs. The training require around 2 hours 30 minutes for each epoch. You can launch distributed training using the following command.

```
python -m torch.distributed.launch --nproc_per_node 4 train_lm.py
```

## Unit test

```
python -m unittest wlm_test.py
```