from transformers import BertForMaskedLM
from transformers import pipeline
import numpy as np
from collections import Counter
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

#  Create a new data point based on an existing data point and matching label, by changing a word
# @param data should be the standard list of features: text, input_ids, token_type_ids, attention_mask, and a label
# @return a data point in the same format as above
def replace_synonym(text):
    fill_mask_pipeline = pipeline("fill-mask", model="bert-base-cased")

    tokens = tokenizer(text)["input_ids"]
    mask_token_index = np.random.randint(1, len(tokens) - 1)
    while tokens[mask_token_index] in [133, 120, 174, 1477, 1475, 135]:
        mask_token_index = np.random.randint(1, len(tokens) - 1)
    masked_tokens = tokens.copy()
    print(tokenizer.mask_token_id)
    print(mask_token_index)
    masked_tokens[mask_token_index] = tokenizer.mask_token_id
    print(masked_tokens)
    masked_text = tokenizer.decode(masked_tokens)
    
    print(masked_text)

    preds = fill_mask_pipeline(masked_text)
    print(preds)
    for pred in preds:
        print(pred["token"], tokens[mask_token_index])
    augments = [pred["sequence"] for pred in preds if pred["token"] != tokens[mask_token_index]]
    # print(augments)
    best_augment = augments[0]

    return best_augment

def insert_augment(text):
    fill_mask_pipeline = pipeline("fill-mask", model="bert-base-cased")

    tokens = tokenizer(text)["input_ids"]
    mask_token_index = np.random.randint(1, len(tokens) - 1)
    while tokens[mask_token_index] in [120, 174, 1477, 1475, 135]:
        mask_token_index = np.random.randint(1, len(tokens) - 1)
    masked_tokens = tokens.copy()
    print(tokenizer.mask_token_id)
    print(mask_token_index)
    masked_tokens.insert(mask_token_index, tokenizer.mask_token_id)
    print(masked_tokens)
    masked_text = tokenizer.decode(masked_tokens)
    
    print(masked_text)

    preds = fill_mask_pipeline(masked_text)
    print(preds)
    for pred in preds:
        print(tokenizer.decode(pred["token"]))
    augments = [pred["sequence"] for pred in preds]
    # print(augments)
    best_augment = augments[0]

    return best_augment

def augment_data_point(data, method="synonym"):
    text = data["text"]
    if method == "synonym":
        augmented_text = replace_synonym(text)
    elif method == "insert":
        augmented_text = insert_augment(text)
    elif method == "delete":
        augmented_text = text
    else:
        augmented_text = text
    tokenized = tokenizer(augmented_text)
    labels = data["labels"]
    return {"text": augmented_text, "input_ids": tokenized["input_ids"], "token_type_ids": tokenized["token_type_ids"],
            "attention_mask": tokenized["attention_mask"], "labels": labels}

text = "<e1>Trauma</e1> to the face and nasal area causes <e2>nosebleeds</e2>, such as getting punched or violently slapped."
# data = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")
augmented_data = insert_augment(text)
print(augmented_data)