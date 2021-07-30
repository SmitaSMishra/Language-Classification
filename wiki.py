import math
import pickle
import sys
import random
from tree import Node

ANSWERS = ["nl", "it"]

def is_consonant(alphabet):
    a = alphabet.lower()
    return (a != 'a' and a != 'e' and a != 'i' and a != 'o' and a != 'u')

def get_avg_max_consonent(word_list):
    max_count = 0
    temp_count = 0
    sum = 0
    for word in word_list:
        for alpha in word:
            if is_consonant(alpha):
                temp_count += 1
            else:
                max_count = max(max_count,temp_count)
                temp_count = 0
        sum += max(max_count,temp_count)
    return sum / len(word_list)

def build_features(Sentences,train_test):
    feature_dataset = []
    not_it_alpha = ['j', 'k', 'w', 'x', 'y']
    for row in Sentences:
        feature_row = []
        if train_test == 'train':
            lang = row.split('|')[1]
        else:
            lang = row
        words = lang.split()

        if get_avg_max_consonent(words) > 4.0:
            feature_row.append(True)
        else:
            feature_row.append(False)

        avg_word_len  = sum(len(word) for word in words) / len(words)
        if avg_word_len > 5.0:
            feature_row.append(True)
        else:
            feature_row.append(False)

        contains_appos = lang.find("'") != -1
        if contains_appos:
            feature_row.append(True)
        else:
            feature_row.append(False)

        contains_aplhas = [alpha for alpha in not_it_alpha if (alpha in lang)]
        if contains_aplhas:
            feature_row.append(True)
        else:
            feature_row.append(False)


        e_count = lang.count('e')
        if e_count > 4:
            feature_row.append(True)
        else:
            feature_row.append(False)

        if train_test == 'train':
            feature_row.append(row.split('|')[0])
        feature_dataset.append(feature_row)
    return feature_dataset


def entropy(du,it):
    if du == it ==0:
        p = 0
    else:
        p = du/(du+it)
    q = 1-p
    # print("p: ",p)
    if p == 1 or q == 1:
        return 0
    else:
        return -(p * math.log(p,2) + q * math.log(q,2))

def entropy_node(node):
    entrophy = entropy(node.dutchCount,node.italianCount)
    return entrophy

def entropy_data(examples):
    du,it = find_truthvalues(examples)
    entrophy = entropy(len(du),len(it))
    return entrophy

def subset_ei(Examples, ai):
    subset = []
    for e in Examples:
        if e[len(e)-1] is ai:
            subset.append(ai)
    return subset

def find_truthvalues(Examples):
    du = []
    it = []
    for ei in Examples:
        if ei[len(ei)-1] == ANSWERS[0].lower():
            du.append(ei)
        else:
            it.append(ei)
    return du, it

def split_as_attribute(Examples,attr):
    left_child = []
    right_child = []
    for ei in Examples:
        if ei[attr] == True:
            left_child.append(ei)
        else:
            right_child.append(ei)
    return left_child,right_child

def find_child_nodes(Examples, attr):
    left_child, right_child = split_as_attribute(Examples,attr)

    # if len(left_child) > 0:
    ldu,lit = find_truthvalues(left_child)
    left_node = Node(None,len(ldu),len(lit))
    # if len(right_child) > 0:
    rdu, rit = find_truthvalues(right_child)
    right_node = Node(None,len(rdu),len(rit))
    return left_node,right_node

def make_node(attribute, Examples):
    du,it = find_truthvalues(Examples)
    new_node = Node(attribute,len(du),len(it))
    return new_node

def calculate_gain(node):
    total_examples = node.dutchCount + node.italianCount
    left_weight = node.left.dutchCount + node.left.italianCount
    right_weight = node.right.dutchCount + node.right.italianCount
    gain = entropy_node(node) - \
           (left_weight/total_examples * entropy_node(node.left) +
            right_weight/total_examples * entropy_node(node.right))
    return gain

def best_attribute(Examples,Attributes):
    best_gain = 0
    best_node = None
    for attr in Attributes:
        root_node = make_node(attr, Examples)
        left_node,right_node = find_child_nodes(Examples, attr)
        root_node.left = left_node
        root_node.right = right_node
        gain = calculate_gain(root_node)
        if gain > best_gain or best_node is None:
            best_gain = gain
            best_node = root_node
    return best_node

def get_majority_output(Examples):
    du,it = find_truthvalues(Examples)
    if len(du) >= len(it):
        prediction = ANSWERS[0]
    else:
        prediction = ANSWERS[1]
    return Node("",len(du),len(it),prediction)


def d_tree(Examples, Attributes):
    if entropy_data(Examples) < 0.001:
        return get_majority_output(Examples)
    if not Attributes:
        return get_majority_output(Examples)
    best_root_node = best_attribute(Examples,Attributes)
    # print(Attributes)
    # print(Examples)
    left_split, right_split = split_as_attribute(Examples,best_root_node.attribute)
    remaining_attributes = Attributes[:len(Attributes)-1]
    if len(left_split) > 0:
        left_split = [j[:best_root_node.attribute]+j[best_root_node.attribute+1:] for j in left_split]
        best_root_node.left = d_tree(left_split, remaining_attributes)
    else:
        best_root_node.left = Node(best_root_node.attribute,0,0,get_majority_output(Examples))

    if len(right_split) > 0:
        right_split = [j[:best_root_node.attribute]+j[best_root_node.attribute+1:] for j in right_split]
        best_root_node.right = d_tree(right_split, remaining_attributes)
    else:
        best_root_node.right = Node(best_root_node.attribute,0,0,get_majority_output(Examples).prediction)

    return best_root_node

def decision_stump_learning(feature_examples, weights):
    weight_thresholds = []
    weight = 0
    for w in weights:
        weight += w
        weight_thresholds.append(weight)
    new_dataset = []
    for i in range(len(feature_examples)):
        random_weight = random.uniform(0,1)
        for j in range(len(feature_examples)):
            if random_weight < weight_thresholds[j]:
                new_dataset.append(feature_examples[j])
                break
    best_node = best_attribute(new_dataset,[i for i in range(len(new_dataset[0])-1)])
    return new_dataset,best_node.attribute

def normalize(weights):
    total_weight = sum(weights)
    normalized_weights = []
    for weight in weights:
        normalized_weights.append(weight/total_weight)
    return normalized_weights

def ada_boost(feature_examples):
    feature_length = len(feature_examples[0])-1
    result = [feature_examples[i][-1] == ANSWERS[0] for i in range(len(feature_examples))]
    w = [(1/len(feature_examples)) for i in range(len(feature_examples))]
    h = [None for _ in range(feature_length)]
    z = [0 for _ in range(feature_length)]
    # epsilon = 0.1
    for k in range(feature_length):
        new_dataset,h[k] = decision_stump_learning(feature_examples, w)
        error = 0
        for n in range(len(new_dataset)):
            if new_dataset[n][k] is not result[n]:
                error += w[n]
        # if error > 1/2:
        #     break;
        # error = min(error,1-epsilon)
        for n in range(len(new_dataset)):
            if new_dataset[n][k] is result[n]:
                w[n] = w[n] * (error/(1-error))
        w = normalize(w)
        if error == 0:
            z[k] = float('inf')
        elif error == 1:
            z[k] = 0
        else:
            z[k] = math.log((1-error)/error)
        # z[k] = math.log((1-error)/error)
    predicted_hypothesis = []
    for k in range(feature_length):
        predicted_hypothesis.append((h[k],z[k]))
    return predicted_hypothesis



def read_file(file_name):
    data = []
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            data.append(line.replace("\n",""))
    return data

def predict_ada(tuple, ada_trees):
    nl_prediction = 0
    it_prediction = 0
    for tree in ada_trees:
        if tuple[tree[0]]:
            nl_prediction += tree[1]
        else:
            it_prediction += tree[1]
    if it_prediction < nl_prediction:
        return ANSWERS[1]
    else:
        return ANSWERS[0]


def predict(data,node):
    # if node.attribute ==
    if node.prediction != None:
        return node.prediction
    if data[node.attribute] == True:
        return predict(data, node.left)
    else:
        return predict(data, node.right)

def clasify_predict(data, node):
    if isinstance(node, list):
        return predict_ada(data,node)
    else:
        return predict(data,node)

def main():
    train_predict = sys.argv[1]
    data_read = read_file('it.txt')
    feature_examples = build_features(data_read, 'train')
    attributes = [i for i in range(len(feature_examples[0])-1)]
    evaluate = True
    if train_predict == 'train':
        tree = d_tree(feature_examples[:4900], attributes)
        print("Decision Tree: ",tree)
        pickle.dump(tree, open('trained_model.txt', 'wb'))
        ada_tree = ada_boost(feature_examples[:4900])
        print("Adaboost Training Result: ", ada_tree)
        pickle.dump(ada_tree, open('ada_learned.txt','wb'))
        print('Training completed')

    else:
        # input_file = read_file('it.txt')
        if sys.argv[2] == 'tree' or sys.argv[2] == 'best':
            node = pickle.load(open('trained_model.txt', "rb"))
        elif sys.argv[2] == 'stumps':
            node = pickle.load(open('ada_learned.txt', "rb"))

        test_matrix = []
        if sys.argv[3] != 'evaluate':
            input_file = read_file(sys.argv[3])
            test_matrix = build_features(input_file,train_predict)
        else:
            test_matrix1 = feature_examples[4900:]
            for i in test_matrix1:
                t = []
                for j in i[:-2]:
                    t.append(j)
                test_matrix.append(t)

        # test_matrix = test_matrix1[]

        print('Predicted as:')
        counter = 0
        for data in range(len(test_matrix)):
            output_prediction = clasify_predict(test_matrix[data], node)
            if sys.argv[3] == 'evaluate':
                if output_prediction == test_matrix1[data][-1]:
                    counter += 1
            print("prediction: ",output_prediction)
        print(counter)

main()