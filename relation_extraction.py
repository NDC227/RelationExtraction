import math
import numpy as np
# import scipy
# import sklearn
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from datasets import load_from_disk

import data_processing
import model_training

model_folder = "trained_model"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# print(device)
# print(torch.__version__)
# print(torch.version.cuda)

def import_data(dataset_name):
    dataset = load_dataset(dataset_name)
    train, test = dataset["train"], dataset["test"]

    train = train.rename_column("sentence", "text") # combine these?
    train = train.rename_column("relation", "labels")
    test = test.rename_column("sentence", "text")
    test = test.rename_column("relation", "labels")
    print(train)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, return_tensors="pt").to(device)

    train_dataset = train.map(tokenize_function, batched=True)
    test_dataset = test.map(tokenize_function, batched=True)

    print(train_dataset[0])
    train_dataset.save_to_disk("datasets/train.hf")
    test_dataset.save_to_disk("datasets/test.hf")

def load_data(train_link, test_link):
    train_dataset = load_from_disk(train_link)
    test_dataset = load_from_disk(test_link)
    train_dataset.set_format(type="torch")
    test_dataset.set_format(type="torch")
    train_dataset = train_dataset.shuffle(seed=0).select(range(1000))
    test_dataset = test_dataset.shuffle(seed=0).select(range(50))
    return train_dataset, test_dataset

# Flags/Constants for manipulating data
IMPORT_DATA = False
LIMIT_CLASSES = True
NUM_PER_CLASS = 10
RESAMPLE_CLASSES = False
AUGMENT_DATA = False

if IMPORT_DATA:
    import_data("sem_eval_2010_task_8")

train_dataset, test_dataset = load_data("datasets/train.hf", "datasets/test.hf")
print(train_dataset)
print(test_dataset)

if LIMIT_CLASSES:
    train_dataset = data_processing.make_low_resource(train_dataset, NUM_PER_CLASS)
# print(train_dataset)
# print(train_dataset[0])
# quit(0)
    
if RESAMPLE_CLASSES:
    train_dataset = data_processing.balance_classes(train_dataset)
    print(len(train_dataset))

if AUGMENT_DATA:
    print(len(train_dataset))
    train_dataset = data_processing.augment_data_by_class(train_dataset, 5, by_class=True)
    print(len(train_dataset))

# quit(0)

from transformers import AutoModelForSequenceClassification

TRAIN = False
if TRAIN:
    model_training.train_model(model_folder, device, hf_implementation=False)

model = AutoModelForSequenceClassification.from_pretrained("./" + model_folder).to(device)
untrained_model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=19)
model = untrained_model
model.to(device)

model_acc = model_training.test(model, device, test_dataset)
print(model_acc)

# untrained_model_acc = test(untrained_model, test_dataset)
# print(untrained_model_acc)