from src.utils import get_distinct_value_indexes
import numpy as np
from random import choice, shuffle

def stratified_train_test_split(dataset: np.array, classification_column_index: int, training_set_proportion: float):
    classification_column = dataset[:, classification_column_index]
    class_indexes = list(get_distinct_value_indexes(classification_column).values())

    class_amounts = []
    for index_list in class_indexes:
        class_amounts.append(len(index_list))
    class_amounts_training_set = []
    for amount_in_class in class_amounts:
        class_amounts_training_set.append(round(amount_in_class*training_set_proportion))

    training_set = []
    for i in range(len(class_amounts_training_set)):
        for _ in range(class_amounts_training_set[i]):
            random_class_index_location = choice(range(len(class_indexes[i])))
            record_index = class_indexes[i][random_class_index_location]
            class_indexes[i].pop(random_class_index_location)
            training_set.append(dataset[record_index])

    testing_set = []
    for index_list in class_indexes:
        for index in index_list:
            testing_set.append(dataset[index])

    return np.array(training_set), np.array(testing_set)

def stratified_k_fold_split(dataset, classification_column_index, no_of_folds):
    # calculate sublist sizes based on number of folds
    fold_sizes = calculate_sublist_sizes(no_of_folds, len(dataset))
    folds = [[] for _ in range(no_of_folds)]
    # stratified splitting of records
    classification_column = dataset[:, classification_column_index]
    class_indexes = list(get_distinct_value_indexes(classification_column).values())
    class_lists = [[dataset[class_index] for class_index in distinct_class] for distinct_class in class_indexes]
    for class_list in class_lists:
        shuffle(class_list)
        while len(class_list) > 0:
            for i in range(len(folds)):
                if len(folds[i]) < fold_sizes[i]:
                    folds[i].append(class_list[0])
                    class_list.pop(0)
    return folds


def get_random_sublist(data, size:int):
    sublist = []
    for i in range(0, size):
        random_index = choice(range(0, len(data)))
        sublist.append(data[random_index])
        data.pop(random_index)
    return sublist


def split_list_values(sublist_sizes, data):
    sublists = []
    for i, bin_size in enumerate(sublist_sizes):
        sublists.append([])
        for j in range(bin_size):
            sublists[i].append(data[0])
            data.pop(0)
    return sublists

def split_list_values_random(sublist_sizes, data):
    sublists = []
    for size in sublist_sizes:
        sublists.append(get_random_sublist(data, size))
    return sublists

def calculate_sublist_sizes(no_of_sublists, no_of_items):
    sublist_size = no_of_items // no_of_sublists
    sublist_sizes = [sublist_size] * no_of_sublists
    bin_size_remainder = no_of_items % no_of_sublists
    for i in range(bin_size_remainder):  # remainder size added to the outer sublists
        index_to_increment = i // 2
        if i % 2 == 1:
            index_to_increment = (index_to_increment + 1) * -1
        sublist_sizes[index_to_increment] += 1
    return sublist_sizes
