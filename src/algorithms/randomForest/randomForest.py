from src.algorithms.ID3.ID3 import train_decision_tree, output_decision_tree, classify_record_id3
from src.datasetSplitting import calculate_sublist_sizes, split_list_values_random
from src.utils import find_most_common_value
import numpy as np

def construct_random_forest(training_data:np.array, column_names, classes, classification_column_index, confidence_threshold=1, residual_data_threshold=1, trees:int=5):
    forest = []
    sublist_sizes = calculate_sublist_sizes(trees, len(training_data))
    training_data_separated = split_list_values_random(sublist_sizes, list(training_data.copy()))
    for i in range(trees):
        forest.append(train_decision_tree(np.array(training_data_separated[i]), column_names, classes, classification_column_index, f"{i}",confidence_threshold, residual_data_threshold))
    return forest

def test_random_forest(forest, test_data, column_names, classes, classification_column_index):
    confusion_matrix = np.array([[0 for _ in range(len(classes))] for _ in range(len(classes))])
    class_confusion_matrix_indexes = {classes[i]: i for i in range(len(classes))}
    incorrectly_classified = []
    for i, record in enumerate(test_data):
        record_classification = classify_record_random_forest(forest, record, column_names)
        confusion_matrix[class_confusion_matrix_indexes[record[classification_column_index]]][
            class_confusion_matrix_indexes[record_classification]] += 1
        if record[classification_column_index] != record_classification:
            incorrectly_classified.append(i)
    return confusion_matrix, incorrectly_classified

def classify_record_random_forest(forest, record, column_names):
    classifications = []
    for tree in forest:
        classifications.append(classify_record_id3(tree, record, column_names))
    classification, classification_amount = find_most_common_value(classifications)
    return classification

def output_random_forest(forest):
    for tree in forest:
        output_decision_tree(tree)
