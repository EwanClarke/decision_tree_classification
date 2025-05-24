
class DecisionTreeNode:
    def __init__(self, value):
        self.value = value


class BranchNode(DecisionTreeNode):
    def __init__(self, criterion, value, most_common_classification):
        super().__init__(value)
        self.criterion = criterion
        self.most_common_classification = most_common_classification
        self.child_nodes: [DecisionTreeNode] = []

    def add_child_node(self, child_node: DecisionTreeNode):
        self.child_nodes.append(child_node)

    def get_children(self):
        return self.child_nodes.copy()


class LeafNode(DecisionTreeNode):
    def __init__(self, value, classification, certainty):
        super().__init__(value)
        self.classification = classification
        self.certainty = certainty