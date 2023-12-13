import numpy as np
import torch
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("./trained_model")

untrained_model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=19)

input = "The <e1>company</e1> fabricates plastic <e2>chairs</e2>."
input = tokenizer(input, padding="max_length", truncation=True, return_tensors="pt")
print(input)
# quit(0)

def inference(model, input):
    pred = model(**input)
    logits = pred.logits[0]
    print(logits)
    class_id = torch.argmax(logits)
    return class_id

print(inference(model, input))
print(inference(untrained_model, input))

from datasets import load_dataset
dataset = load_dataset("sem_eval_2010_task_8")
train, test = dataset["train"], dataset["test"]

test = test.rename_column("sentence", "text")
test = test.rename_column("relation", "labels")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, return_tensors="pt").to(device)

test_dataset = test.map(tokenize_function, batched=True)

def test(model, data):
    count = 0
    for datum in data:
        pred = inference(model, datum)
        if pred.item() == datum["labels"]:
            count += 1
    return count / len(data)

model_acc = test(model, test_dataset)
print(model_acc)