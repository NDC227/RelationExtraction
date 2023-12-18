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

IMPORT_DATA = False
if IMPORT_DATA:
    import_data("sem_eval_2010_task_8")

train_dataset, test_dataset = load_data("datasets/train.hf", "datasets/test.hf")
print(train_dataset)
print(test_dataset)

from collections import defaultdict
from datasets import Dataset

def get_data_by_label(dataset):
    data_by_label = defaultdict(list)
    for datum in dataset:
        label = datum["labels"].item()
        data_by_label[label].append(datum)
    print([(label, len(data_by_label[label])) for label in sorted(data_by_label)])
    return data_by_label

def make_low_resource(dataset, num_per_class):
    # Re-process data for low-resource training
    # Note: SemEval 2010 Task 8 only has 1 training data point for label 7. All others have at least 50, if not many hundreds.
    data_by_label = get_data_by_label(dataset)

    new_train_dataset = []
    for label in data_by_label:
        # print(min(len(data_by_label[label]), NUM_PER_CLASS))
        new_train_dataset += list(np.random.choice(data_by_label[label], min(len(data_by_label[label]), num_per_class), replace=False))
    print("new_dataset_length", len(new_train_dataset))

    new_train_dataset = Dataset.from_list(new_train_dataset)
    new_train_dataset.set_format("torch")
    return new_train_dataset

LIMIT_CLASSES = True
NUM_PER_CLASS = 10
if LIMIT_CLASSES:
    train_dataset = make_low_resource(train_dataset, NUM_PER_CLASS)
# print(train_dataset)
# print(train_dataset[0])
# quit(0)
    
def balance_classes(dataset):
    data_by_label = get_data_by_label(dataset)
    label_counts = [len(data_by_label[key]) for key in data_by_label]
    print(label_counts)

    percentile = 0.25
    label_counts = sorted(label_counts)
    min_count_accepted = label_counts[math.ceil(len(label_counts) * percentile)]
    amt_to_augment = {label: max(0, min_count_accepted - len(data_by_label[label])) for label in data_by_label}
    print(amt_to_augment)

    #TODO: This could probably be more concise, but I want to update data_by_label for the debug
    new_train_dataset = []
    for key in data_by_label:
        augmentation = list(np.random.choice(data_by_label[key], amt_to_augment[key], replace=True))
        data_by_label[key] += augmentation
        new_train_dataset += data_by_label[key]
    
    print([len(data_by_label[key]) for key in data_by_label])
    # print(sorted([data["text"] for data in data_by_label[12]]))
    new_train_dataset = Dataset.from_list(new_train_dataset)
    new_train_dataset.set_format("torch")
    return new_train_dataset

RESAMPLE_CLASSES = False
if RESAMPLE_CLASSES:
    train_dataset = balance_classes(train_dataset)
    print(len(train_dataset))

# Create a new data point based on an existing data point and matching label, by changing a word
# @param data should be the standard list of features: text, input_ids, token_type_ids, attention_mask, and a label
# @return a data point in the same format as above

from collections import Counter
from transformers import BertForMaskedLM
masked_LM_bert = BertForMaskedLM.from_pretrained("bert-base-uncased")
def augment_data_point(data, method="synonym"):
    text = data["text"]
    if method == "synonym":
        # print(data["attention_mask"].tolist())

        text_counter = Counter(data["attention_mask"].tolist())
        # print(text_counter)
        text_length = text_counter[1]
        # print(text_length)
        mask_token_index = np.random.randint(text_length)
        data["input_ids"][mask_token_index] = tokenizer.mask_token_id

        with torch.no_grad():
            logits = masked_LM_bert(data["input_ids"].unsqueeze(0), 
                                    data["token_type_ids"].unsqueeze(0), 
                                    data["attention_mask"].unsqueeze(0))

        # print(data["input_ids"])
        # print(logits)
        # print(logits.logits)
        # print(logits.logits[0])
        # print(logits.logits[0][mask_token_index])
        predicted_token_id = logits.logits[0][mask_token_index].argmax()
        synonym = tokenizer.decode(predicted_token_id)
        masked_text = tokenizer.decode(data["input_ids"][:text_length])
        print(predicted_token_id)
        print(masked_text)
        print(synonym)

        augmented_text = text
    elif method == "insert":
        augmented_text = text
    elif method == "delete":
        augmented_text = text
    else:
        augmented_text = text
    tokenized = tokenizer(augmented_text)
    labels = data["labels"]
    return {"text": augmented_text, "input_ids": tokenized["input_ids"], "token_type_ids": tokenized["token_type_ids"],
            "attention_mask": tokenized["attention_mask"], "labels": labels}

def augment_data_by_class(dataset, num_to_augment, by_class=True):
    if by_class:
        data_by_label = get_data_by_label(dataset)
        label_counts = [len(data_by_label[key]) for key in data_by_label]
        print(label_counts)
        
        for label in data_by_label:
            augmentation = []
            for i in range(num_to_augment):
                model_data = np.random.choice(data_by_label[label])
                new_data = augment_data_point(model_data)
                augmentation.append(new_data)
            data_by_label[label] += augmentation

        new_train_dataset = []
        for label in data_by_label:
            new_train_dataset += data_by_label[label]

        new_train_dataset = Dataset.from_list(new_train_dataset)
        new_train_dataset.set_format("torch")
        return new_train_dataset
    else:
        augmentation = []
        for i in range(num_to_augment):
            model_data = np.random.choice(dataset)
            new_data = augment_data_point(model_data)
            augmentation.append(new_data)
        new_train_dataset = list(dataset) + augmentation
        new_train_dataset = Dataset.from_list(new_train_dataset)
        new_train_dataset.set_format("torch")
        return new_train_dataset 

# print(augment_data_point(train_dataset[0]))
print(len(train_dataset))
train_dataset = augment_data_by_class(train_dataset, 5, by_class=False)
print(len(train_dataset))

quit(0)

from transformers import AutoModelForSequenceClassification
TRAIN = False
if TRAIN:
    HF_TRAINER = False
    FREEZE_LAYERS = []

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=19)
    model.to(device)

    # Freeze the BERT encoder for training
    for name, param in model.named_parameters():
        if sum([name.startswith(layer) for layer in FREEZE_LAYERS]) > 0:
            param.requires_grad = False
        # print(name, param.requires_grad)

    import evaluate

    # Train using HuggingFace Trainer
    if HF_TRAINER:
        from transformers import TrainingArguments
        training_args = TrainingArguments(per_device_train_batch_size=8,
                                        per_device_eval_batch_size=8,
                                        learning_rate=5e-5,
                                        weight_decay=0,
                                        num_train_epochs=3,
                                        save_strategy="steps",
                                        save_steps=100,
                                        save_total_limit=3,
                                        load_best_model_at_end=True,
                                        optim="adamw_torch",
                                        output_dir="trainer_ckpts", 
                                        evaluation_strategy="steps",
                                        eval_steps=100)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        from transformers import Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        trainer.save_model(model_folder)

    else:
        BATCH_SIZE = 8
        train_dataset = train_dataset.remove_columns(["text"])
        test_dataset = test_dataset.remove_columns(["text"])

        from torch.optim import AdamW
        optimizer = AdamW(model.parameters(), lr=5e-5)

        from transformers import get_scheduler
        num_epochs = 3
        num_training_steps = num_epochs * math.ceil(len(train_dataset) / BATCH_SIZE)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )

        progress_bar = tqdm(range(num_training_steps))

        def get_batches(dataset, batch_size, max_batches=math.inf):
            batches = []
            n = len(dataset)
            idxs = np.arange(n)
            np.random.shuffle(idxs)
            for i in range(min(math.ceil(n / batch_size), max_batches)):
                batch = {}
                batch["labels"] = torch.stack([dataset[int(idx)]["labels"] for idx in idxs[i*batch_size:i*batch_size+batch_size]])
                batch["input_ids"] = torch.stack([dataset[int(idx)]["input_ids"] for idx in idxs[i*batch_size:i*batch_size+batch_size]])
                batch["token_type_ids"] = torch.stack([dataset[int(idx)]["token_type_ids"] for idx in idxs[i*batch_size:i*batch_size+batch_size]])
                batch["attention_mask"] = torch.stack([dataset[int(idx)]["attention_mask"] for idx in idxs[i*batch_size:i*batch_size+batch_size]])
                batches.append(batch)
            return batches

        def train_model(model):
            model.train()
            for epoch in range(num_epochs):
                batches = get_batches(train_dataset, BATCH_SIZE)
                
                for i in range(len(batches)):
                    batch = batches[i]
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    loss = outputs.loss
                    loss.backward()

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
        train_model(model)
        
        def eval_model(model):
            metric = evaluate.load("accuracy")
            model.eval()
            batches = get_batches(test_dataset, BATCH_SIZE)
            for i in tqdm(range(len(batches))):
                batch = batches[i]
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)

                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=predictions, references=batch["labels"])

            return metric.compute()
    
        model_acc = eval_model(model)
        print(model_acc)

        model.save_pretrained(model_folder, from_pt=True)

model = AutoModelForSequenceClassification.from_pretrained("./" + model_folder).to(device)
untrained_model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=19)
model = untrained_model
model.to(device)

def inference(model, input):
    # print(input)
    # print(next(model.parameters()).device)
    model.eval()
    with torch.no_grad():
        pred = model(input["input_ids"], input["token_type_ids"], input["attention_mask"])
    logits = pred.logits[0]
    # print(logits)
    class_id = torch.argmax(logits)
    return class_id

def test(model, data):
    model.eval()
    labels = []
    predictions = []
    count = 0
    for datum in tqdm(data):
        datum["input_ids"] = datum["input_ids"].unsqueeze(0).to(device)
        datum["token_type_ids"] = datum["token_type_ids"].unsqueeze(0).to(device)
        datum["attention_mask"] = datum["attention_mask"].unsqueeze(0).to(device)
        # print(datum)
        pred = inference(model, datum)
        labels.append(datum["labels"].item())
        predictions.append(pred.item())
        if pred.item() == datum["labels"]:
            count += 1

    accuracy = count / len(data)
    from collections import Counter
    print(Counter(predictions))

    from sklearn.metrics import f1_score
    micro_f1 = f1_score(labels, predictions, average="micro")
    macro_f1 = f1_score(labels, predictions, average="macro")
    return accuracy, micro_f1, macro_f1

model_acc = test(model, test_dataset)
print(model_acc)

# untrained_model_acc = test(untrained_model, test_dataset)
# print(untrained_model_acc)