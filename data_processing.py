import numpy as np
import math
from collections import defaultdict
from datasets import Dataset

import data_augmentation

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

def augment_data_by_class(dataset, num_to_augment, by_class=True):
    if by_class:
        data_by_label = get_data_by_label(dataset)
        label_counts = [len(data_by_label[key]) for key in data_by_label]
        # print(label_counts)
        
        for label in data_by_label:
            augmentation = []
            for i in range(num_to_augment):
                model_data = np.random.choice(data_by_label[label])
                new_data = data_augmentation.augment_data_point(model_data)
                augmentation.append(new_data)
            data_by_label[label] += augmentation

        new_train_dataset = []
        for label in data_by_label:
            new_train_dataset += data_by_label[label]

        new_train_dataset = Dataset.from_list(new_train_dataset)
        new_train_dataset.set_format("torch")
        # print(sorted([data["text"] for data in data_by_label[0]]))
        return new_train_dataset
    else:
        augmentation = []
        for i in range(num_to_augment):
            model_data = np.random.choice(dataset)
            new_data = data_augmentation.augment_data_point(model_data)
            augmentation.append(new_data)
        new_train_dataset = list(dataset) + augmentation
        new_train_dataset = Dataset.from_list(new_train_dataset)
        new_train_dataset.set_format("torch")
        return new_train_dataset 