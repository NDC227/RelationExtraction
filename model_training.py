import numpy as np
import math
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

def train_model(output_dir, device, hf_implementation=False):
    import evaluate

    FREEZE_LAYERS = []

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=19)
    model.to(device)

    # Freeze the BERT encoder for training
    for name, param in model.named_parameters():
        if sum([name.startswith(layer) for layer in FREEZE_LAYERS]) > 0:
            param.requires_grad = False
        # print(name, param.requires_grad)

    # Train using HuggingFace Trainer
    if hf_implementation:
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
        trainer.save_model(output_dir)

    else:
        from torch.optim import AdamW
        from transformers import get_scheduler

        BATCH_SIZE = 8
        train_dataset = train_dataset.remove_columns(["text"])
        test_dataset = test_dataset.remove_columns(["text"])
        
        optimizer = AdamW(model.parameters(), lr=5e-5)

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

        model.save_pretrained(output_dir, from_pt=True)

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

def test(model, device, data):
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