from collections import defaultdict


def count_value_instances(values: list):
    value_amounts = defaultdict(int)
    for value in values:
        value_amounts[value] += 1
    return value_amounts

def find_most_common_value(values: list):
    value_amounts = count_value_instances(values)
    most_common_value = None
    most_common_value_amount = 0
    for key in value_amounts.keys():
        if value_amounts[key] > most_common_value_amount:
            most_common_value = key
            most_common_value_amount = value_amounts[key]
    return most_common_value, most_common_value_amount

def get_distinct_value_indexes(column: list):
    distinct_value_indexes = defaultdict(list)
    for i, value in enumerate(column):
        distinct_value_indexes[value].append(i)
    return distinct_value_indexes


def calculate_f1_score(precision, recall):
    return 2 * ((precision * recall) / (precision + recall))


def calculate_recall(true_positive, false_negative):
    return true_positive / (true_positive + false_negative)


def calculate_precision(true_positive, false_positive):
    return true_positive / (true_positive + false_positive)


def calculate_accuracy(true_positive, true_negative, false_positive, false_negative):
    return (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)


def interpret_confusion_matrix(confusion_matrix, class_index):
    # calculate TP row == column of class, TN row == column not of class, FP column of class row != column, FN row of class row != column
    true_positive = confusion_matrix[class_index][class_index]
    true_negative = sum([confusion_matrix[i][i] for i in range(len(confusion_matrix)) if i != class_index])
    false_positive = sum([confusion_matrix[i][class_index] for i in range(len(confusion_matrix[class_index])) if i != class_index])
    false_negative = sum([confusion_matrix[class_index][i] for i in range(len(confusion_matrix[class_index])) if i != class_index])
    return true_positive, true_negative, false_positive, false_negative


def calculate_performance_metrics(confusion_matrix, class_matrix):
    tp, tn, fp, fn = interpret_confusion_matrix(confusion_matrix, class_matrix)
    accuracy = calculate_accuracy(tp, tn, fp, fn)
    precision = calculate_precision(tp, fp)
    recall = calculate_recall(tp, fn)
    f1_score = calculate_f1_score(precision, recall)
    return accuracy, precision, recall, f1_score

