import numpy as np

from src.datasetSplitting import get_random_sublist
from src.algorithms.ID3.decisionTree import DecisionTreeNode, BranchNode, LeafNode
from src.utils import count_value_instances, find_most_common_value, get_distinct_value_indexes

def train_decision_tree(training_data: np.array, column_names: list, classes, classification_column_index: int, root_value="root", confidence_threshold=1, residual_data_threshold=1):
    # partition training data
    window_size = calculate_sublist_size_proportional(training_data, 0.3)
    window = get_random_sublist(list(training_data.copy()), window_size)
    window = np.array(window)
    decision_tree = construct_decision_tree(window, column_names, classification_column_index)
    # train the decision tree on incorrectly classified records until all objects in the training data set are classified correctly
    training_data_classified = False
    while not training_data_classified:
        _, incorrectly_classified = test_decision_tree(decision_tree, training_data, column_names, classes, classification_column_index)
        incorrectly_classified = sorted(incorrectly_classified, reverse=True)
        if len(incorrectly_classified) != 0:
            window = list(window)
            training_data = list(training_data)
            for incorrectly_classified_index in incorrectly_classified:
                window.append(training_data[incorrectly_classified_index])
                training_data.pop(incorrectly_classified_index)
            training_data = np.array(training_data)
            window = np.array(window)

            decision_tree = construct_decision_tree(window, column_names, classification_column_index, root_value, confidence_threshold, residual_data_threshold)
        else:
            training_data_classified = True
    return decision_tree


def construct_decision_tree(training_data: np.array, column_names: list, classification_column_index: int, value: str= "root", confidence_threshold:float=1, residual_data_threshold:int=1):
    classification_column = training_data[:, classification_column_index]
    class_amounts = list(count_value_instances(classification_column).values())
    classification_column_entropy = calculate_entropy(class_amounts)


    if classification_column_entropy == 0:
        return LeafNode(value, classification_column[0], 1)

    information_gain_of_attributes = calculate_information_gain_of_attributes(training_data,
                                                                              classification_column_index,
                                                                              classification_column_entropy)
    no_positive_gain = all_values_non_positive(information_gain_of_attributes)
    most_common_value, most_common_value_amount = find_most_common_value(classification_column)
    confidence = most_common_value_amount/len(classification_column)
    if no_positive_gain or confidence >= confidence_threshold or len(classification_column) <= residual_data_threshold:
        return LeafNode(value, most_common_value, confidence)

    highest_information_gain = max(information_gain_of_attributes)
    criterion_attribute_index = information_gain_of_attributes.index(highest_information_gain)

    decision_node = BranchNode(column_names[criterion_attribute_index], value, most_common_value)
    child_column_names = column_names.copy()
    child_column_names.pop(criterion_attribute_index)
    child_classification_column_index = classification_column_index if criterion_attribute_index > classification_column_index else classification_column_index-1
    for value in get_distinct_values(training_data[:, criterion_attribute_index]):
        child_records = np.array([record for record in training_data if record[criterion_attribute_index] == value])
        child_records = np.delete(child_records, criterion_attribute_index, 1)
        decision_node.add_child_node(construct_decision_tree(child_records, child_column_names, child_classification_column_index, value, confidence_threshold, residual_data_threshold))
    return decision_node

def calculate_information_gain_of_attributes(training_data: np.array, classification_column_index: int, classification_column_entropy):
    information_gain_of_attributes = [0] * len(training_data[0])
    for i in range(0, len(training_data[0])):
        if i == classification_column_index:
            continue
        column = training_data[:, i]
        subset_indexes = get_distinct_value_indexes(column)

        entropy_of_subsets = []
        subset_lengths = []
        for subset in subset_indexes.values():
            subset_lengths.append(len(subset))
            subset_classes = []
            for index in subset:
                subset_classes.append(training_data[index][classification_column_index])

            subset_class_amounts = list(count_value_instances(subset_classes).values())
            entropy_of_subsets.append(calculate_entropy(subset_class_amounts))
        attribute_entropy = calculate_total_attribute_entropy(entropy_of_subsets, subset_lengths, len(training_data))
        information_gain_of_attributes[i] = classification_column_entropy - attribute_entropy
    return information_gain_of_attributes

def test_decision_tree(decision_tree: DecisionTreeNode, test_data: np.array, column_names: list, classes, classification_column_index: int):
    # get distinct values in the classification column
    # create a multidimensional array with the rows index being the actual classification and the column index being the predicted
    confusion_matrix = np.array([[0 for _ in range(len(classes))] for _ in range(len(classes))])
    class_confusion_matrix_indexes = {classes[i]: i for i in range(len(classes))}
    incorrectly_classified = []

    for i, record in enumerate(test_data):
        record_classification = classify_record_id3(decision_tree, record, column_names)
        confusion_matrix[class_confusion_matrix_indexes[record[classification_column_index]]][class_confusion_matrix_indexes[record_classification]] += 1
        if record[classification_column_index] != record_classification:
            incorrectly_classified.append(i)
    return confusion_matrix, incorrectly_classified


def classify_record_id3(decision_tree: DecisionTreeNode, record: list, column_names: list):
    current_node = decision_tree
    classification = None
    classified = False

    while not classified:
        if isinstance(current_node, LeafNode):
            classification = current_node.classification
            classified = True
        else:
            assert isinstance(current_node, BranchNode)
            criterion_index = column_names.index(current_node.criterion)
            changed_current_node = False
            for child_node in current_node.child_nodes:
                assert isinstance(child_node, DecisionTreeNode)
                if child_node.value == record[criterion_index]:
                    current_node = child_node
                    changed_current_node = True
            if not changed_current_node:
                classification = current_node.most_common_classification
                classified = True
    return classification

def output_decision_tree(current_node: DecisionTreeNode, depth = 0):
    if isinstance(current_node, LeafNode):
        print(f"{current_node.value}: {current_node.classification}")
        return
    else:
        assert isinstance(current_node, BranchNode)
        print(f"{current_node.value}:")
        for child_node in current_node.child_nodes:
            print(f"{"|  "*depth}{current_node.criterion} = ", end="")
            output_decision_tree(child_node, depth+1)
        return


def calculate_entropy(value_amounts: np.array) -> float:
    total_values = np.sum(value_amounts)
    value_proportions = [value / total_values for value in value_amounts]
    entropy = -sum([(proportion*(np.log2(proportion) if proportion != 0 else 0)) for proportion in value_proportions])
    return entropy


def calculate_total_attribute_entropy(entropy_of_subsets: list, subset_lengths: list, no_of_records: int):
    attribute_entropy = 0
    for i, subset_entropy in enumerate(entropy_of_subsets):
        attribute_entropy += subset_lengths[i]/no_of_records * subset_entropy
    return attribute_entropy


def all_values_non_positive(values: list):
    no_positive_values = True
    for value in values:
        if value > 0:
            no_positive_values = False
            break
    return no_positive_values


def get_distinct_values(values):
    return list(set(values))


def calculate_sublist_size_proportional(values: list, proportion: float, minimum: int = 1, maximum: int = None):
    if proportion < 0 or proportion > 1:
        raise Exception("proportion of subset must be between 0 and 1")

    sublist_size = int(len(values)*proportion)
    if sublist_size < minimum:
        sublist_size = minimum
    elif maximum is not None and sublist_size > maximum:
        sublist_size = maximum
    return sublist_size
