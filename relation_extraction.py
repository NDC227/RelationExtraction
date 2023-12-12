import math
import numpy as np
import scipy
import sklearn
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from datasets import load_from_disk

model_folder = "trainer_ckpts/checkpoint-2200"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
print(torch.__version__)
print(torch.version.cuda)

def load_data():
    dataset = load_dataset("sem_eval_2010_task_8")
    train, test = dataset["train"], dataset["test"]

    train = train.rename_column("sentence", "text") # combine these?
    train = train.rename_column("relation", "labels")
    test = test.rename_column("sentence", "text")
    test = test.rename_column("relation", "labels")

    print(train)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, return_tensors="pt").to(device)

    train_dataset = train.map(tokenize_function, batched=True)
    test_dataset = test.map(tokenize_function, batched=True)

    print(train_dataset[0])
    train_dataset.save_to_disk("datasets/train.hf")
    test_dataset.save_to_disk("datasets/test.hf")

LOAD_DATA = False
if LOAD_DATA:
    load_data()
train_dataset = load_from_disk("datasets/train.hf")
test_dataset = load_from_disk("datasets/test.hf")
train_dataset.set_format(type="torch")
test_dataset.set_format(type="torch")
train_dataset = train_dataset.shuffle(seed=0).select(range(1001))
test_dataset = test_dataset.shuffle(seed=0).select(range(500))
train_dataset
test_dataset
print(train_dataset)
print(test_dataset)

from transformers import AutoModelForSequenceClassification
TRAIN = True
if TRAIN:
    HF_TRAINER = False
    FREEZE_LAYERS = ["bert.encoder"]

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
            print(idxs)
            for i in range(min(math.ceil(n / batch_size), max_batches)):
                batch = {}
                batch["labels"] = torch.stack([dataset[int(idx)]["labels"] for idx in idxs[i*batch_size:i*batch_size+batch_size]])
                batch["input_ids"] = torch.stack([dataset[int(idx)]["input_ids"] for idx in idxs[i*batch_size:i*batch_size+batch_size]])
                batch["token_type_ids"] = torch.stack([dataset[int(idx)]["token_type_ids"] for idx in idxs[i*batch_size:i*batch_size+batch_size]])
                batch["attention_mask"] = torch.stack([dataset[int(idx)]["attention_mask"] for idx in idxs[i*batch_size:i*batch_size+batch_size]])
                batches.append(batch)
            return batches

        model.train()
        for epoch in range(num_epochs):
            batches = get_batches(train_dataset, BATCH_SIZE)
            # print(batches)
            
            for i in range(len(batches)):
                batch = batches[i]
                # print(batch)
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
        
        metric = evaluate.load("accuracy")
        model.eval()
        batches = get_batches(test_dataset, BATCH_SIZE)
        for i in range(len(batches)):
            batch = batches[i]
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        metric.compute()

model = AutoModelForSequenceClassification.from_pretrained("./" + model_folder).to(device)
untrained_model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=19)
model.to(device)

def inference(model, input):
    # print(input)
    # print(next(model.parameters()).device)
    pred = model(input["input_ids"], input["token_type_ids"], input["attention_mask"])
    logits = pred.logits[0]
    # print(logits)
    class_id = torch.argmax(logits)
    return class_id

def test(model, data):
    count = 0
    for datum in tqdm(data):
        datum["input_ids"] = datum["input_ids"].unsqueeze(0).to(device)
        datum["token_type_ids"] = datum["token_type_ids"].unsqueeze(0).to(device)
        datum["attention_mask"] = datum["attention_mask"].unsqueeze(0).to(device)
        # print(datum)
        pred = inference(model, datum)
        if pred.item() == datum["labels"]:
            count += 1
    return count / len(data)

model_acc = test(model, test_dataset)
print(model_acc)

# untrained_model_acc = test(untrained_model, test_dataset)
# print(untrained_model_acc)