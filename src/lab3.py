import json
import re
import sys
from fileinput import filename
from sys import argv
import math
from collections import Counter

def ada(data,features, num_stumps = 1):
    n = len(data)
    weights = [1/n] * n
    model = []
    used_feats = Counter()
    for feature in features:
        used_feats[feature] = 0

    for round in range(num_stumps):
        #weighted data
        new_weights = [(data[i][0], data[i][1],weights[i]) for i in range(n)]
        #introduce stump trained on the weights
        stump = train_stump(new_weights, features, used_feats)
        #Predict using the stump
        predictions = [dt_predict(stump, ex[0]) for ex in data]
        actual = [ex[1] for ex in data]
        #Weighted error
        error = sum(weights[i] for i in range(n) if predictions[i] != actual[i])

        if error == 0: #Prevents dividing by zero
            alpha = 10 # very confident prediction
        elif error >= .5: # very poor prediction
            if isinstance(stump, dict):
                used_feats[stump['feature']] += 1
            if num_stumps == 1:
                model.append({'tree': stump, 'alpha': 0})
            continue
        else:
            alpha = .5 * math.log((1-error)/error)

        for i in range(n): #updating weights
            correct = predictions[i] == actual[i]
            weights[i] *= math.exp(-alpha if correct else alpha)

        #Normalizing Weights
        total = sum(weights)
        weights = [w/total for w in weights]
        # print(
            # f"Round {round + 1}: error={error:.4f}, alpha={alpha:.4f}, feature={stump['feature'] if isinstance(stump, dict) else stump}")
        model.append({'tree': stump, 'alpha': alpha})
        if isinstance(stump, dict):
            used_feats[stump['feature']] += 1
    # for round in model:
    #     if isinstance(round['tree'], dict):
    #         used_feats[round['tree']['feature']] += 1
    # print("Features used across stumps:", used_feats.most_common())
    return model



def train_stump(data,features,used_features):
    best_gain = 0
    best_feature = None
    best_threshold = None
    best_split = None

    base_entropy = entropy_weighted(data)

    for feat in features:
        values = [ex[0][feat] for ex in data]
        gain = -1
        best_local_threshold = None
        best_local_split = None

        if isinstance(values[0],bool):
            left = [ex for ex in data if ex[0][feat]]
            right = [ex for ex in data if not ex[0][feat]]
            gain = base_entropy - weighted_avg_entropy(left, right)
            best_local_split = (left, right)
        else:
            thresholds = sorted(set(values))
            for i in range(len(thresholds) - 1):
                t = (thresholds[i] + thresholds[i+1]) / 2
                left = [ex for ex in data if ex[0][feat] <= t]
                right = [ex for ex in data if ex[0][feat] > t]
                local_gain = base_entropy - weighted_avg_entropy(left, right)
                if local_gain > gain:
                    gain = local_gain
                    best_feature = feat
                    best_local_threshold = t
                    best_local_split = (left, right)
        penalty = used_features[feat]*.3
        gain -= penalty

        if gain > best_gain:
            best_gain = gain
            best_feature = feat
            best_threshold = best_local_threshold
            best_split = best_local_split

    # print(f"Chosen feature: {best_feature}, penalty: {used_features[best_feature]*.3}, final gain: {best_gain}")

    if best_gain == 0 or not best_split:
        majority = Counter(label for _, label, _ in data).most_common(1)[0][0]
        return majority

    return {
        'feature': best_feature,
        'threshold': best_threshold,
        'left': majority_vote(best_split[0]),
        'right': majority_vote(best_split[1])
    }



def entropy_weighted(data):
    total_weight = sum(weight for _, _, weight in data)
    lang_weights = Counter()
    for _, lang, weight in data:
        lang_weights[lang] += weight
    return -sum((lang_weights[lang]/total_weight)* math.log2(lang_weights[lang]/total_weight) for lang in lang_weights)

def weighted_avg_entropy(left,right):
    total = sum(weight for _,_, weight in left+right)
    left_weight = sum(weight for _,_, weight in left)
    right_weight = sum(weight for _,_, weight in right)
    return (left_weight/total) * entropy_weighted(left) + (right_weight/total) * entropy_weighted(right)

def majority_vote(data):
    counts = Counter()
    for _,lang,weight in data:
        counts[lang] += weight
    return counts.most_common(1)[0][0]


# The main method for creating a decision tree of depth 4.
# dt_data | dict(dict,str) = train_examples with the language label provided
# features.txt | set(str) = features.txt to train the AI on
# depth | int = current depth of the tree
# max_depth | int = maximum depth the tree can build, set to 4.
def dt(dt_data,features,depth = 0,max_depth = 4):
    base_entropy = entropy(dt_data)
    # print(f"Building tree at depth {depth} with {len(dt_data)} train_examples")
    labels = [label for _, label in dt_data]

    if all(l == labels[0] for l in labels): #all train_examples are the same language
        # print(f"Pure leaf reached: {labels[0]}")
        return labels[0]
    if depth == max_depth or not features:
        majority = Counter(labels).most_common(1)[0][0]
        # print(f"No info gain at this node → Predict: {majority}")
        return majority

    best_gain = 0
    best_feature = None
    best_threshold = None
    best_split = None

    for feat in features:
        #check value for each example in example set
        values = [ex[0][feat] for ex in dt_data]

        if isinstance(values[0],bool):
            left, right = split_boolean(dt_data, feat)
            gain = info_gain(dt_data, left, right)
            if gain > best_gain:
                best_gain = gain
                best_feature = feat
                best_threshold = None
                best_split = (left, right)
        else:
            #obtain every unique value seen in example set
            unique_vals = sorted(set(values))
            for i in range(len(unique_vals) - 1):
                curr_threshold = (unique_vals[i] + unique_vals[i+1]) / 2
                left, right = split_numeric(dt_data, feat, curr_threshold)
                gain = info_gain(dt_data, left, right)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat
                    best_threshold = curr_threshold
                    best_split = (left, right)

    if best_gain == 0:
        return Counter(labels).most_common(1)[0][0]

    left_branch = dt(best_split[0], features, depth+1, max_depth)
    right_branch = dt(best_split[1], features, depth+1, max_depth)

    return {
        'feature': best_feature,
        'threshold': best_threshold,
        'left': left_branch,
        'right': right_branch
    }

def dt_predict(tree,features):
    if isinstance(tree,str):
        return tree
    value = features[tree['feature']]
    if tree['threshold'] is None: #boolean feature
        branch = tree['left'] if value else tree['right']
    else:
        branch = tree['left'] if value <= tree['threshold'] else tree['right']
    return dt_predict(branch,features)

def ada_predict(model,features):
    score = 0
    for entry in model:
        pred = dt_predict(entry['tree'],features)
        score += entry['alpha'] if pred == 'en' else -entry['alpha']
    return 'en' if score > 0 else 'nl'


def entropy(data):
    lang_counts = Counter(lang for _, lang in data)
    total = len(data)
    return -sum((count / total) * math.log2(count / total) for count in lang_counts.values())

def split_numeric(data, feature, threshold):
    left = [d for d in data if d[0][feature] <= threshold]
    right = [d for d in data if d[0][feature] > threshold]
    return left, right

def split_boolean(data, feature):
    left = [d for d in data if d[0][feature] == True]
    right = [d for d in data if d[0][feature] == False]
    return left, right

def info_gain(data,left, right):
    total = len(data)
    return entropy(data) - (len(left) / total) * entropy(left) - (len(right) / total) * entropy(right)


def get_word_features(text,keyword_list):
    vowels = 'aeiouAEIOU'
    words = text.split()
    total_letter_length = 0
    for word in words:
        total_letter_length += len(word)
    consonant_count = len(re.findall('[%s]' % re.escape(vowels), text))
    vowel_count = total_letter_length - consonant_count

    features = {
        'avg_word_length': total_letter_length / len(words) if words else 0,
        'vowel_ratio': vowel_count / total_letter_length if total_letter_length else 0,
        'consonant_count': consonant_count,
        'word_count': len(words),
        'char_count': total_letter_length,
        'is_question': text.strip().endswith('?'),
        'is_long_text': total_letter_length > 80
    }
    feature_list.add('vowel_ratio')
    feature_list.add('consonant_count')
    feature_list.add('avg_word_length')
    for word in keyword_list:
        features[f'contains_{word}'] = word in text
        feature_list.add('contains_' + word)

    return features


def get_examples(filename:str, predict:bool):
    file = open(filename, 'r', encoding='utf-8')
    if not predict:
        examples = {}
        for line in file:
            ex = line.strip().split("|")
            examples[ex[1]] = ex[0]
        file.close()
        return examples
    else:
        examples = []
        for line in file:
            ex = line.strip()
            examples.append(ex)
        return examples

def get_features(filename:str):
    file = open(filename)
    for line in file:
        keywords.add(line.strip())
    file.close()

def generate_tree(root):
    if isinstance(root,str):
        return root
    else:
        return {
            'feature': root['feature'],
            'threshold': root['threshold'],
            'left': generate_tree(root['left']),
            'right': generate_tree(root['right'])
        }


if __name__ == '__main__':
    args = argv[1:]
    if len(args) >= 1:
        operation = args[0].lower()
        if operation == 'predict':
            if len(args) != 3 and len(args) != 4:
                print("lab3.py predict <predict_examples> <features.txt> <hypothesis>")
            else:
                ex_file = args[1]
                if len(args) == 4:
                    hypo_file = args[3]
                feature_list = set()
                keywords = set()
                input_data = []
                try:
                    feat_file = args[2]
                    get_features(feat_file)
                    example_list = get_examples(ex_file, True)
                    if len(args) == 4:
                        model = json.load(open(hypo_file))
                    else:
                        model = json.load('best_hypothesis.json')
                except FileNotFoundError as e:
                    print(e)
                    sys.exit(1)
                for ex in example_list:
                    features = get_word_features(ex,keywords)
                    if isinstance(model, list) and 'alpha' in model[0]:
                        prediction = ada_predict(model,features)
                    else:
                        prediction = dt_predict(model, features)
                    print(prediction)


        elif operation == 'train':
            if len(args) != 5:
                print("lab3.py train <train_examples> <features.txt> <hypothesisOut> <learning-type>")
            else:
                data_type = args[4]
                if data_type not in ['dt','ada']:
                    print("Invalid data type. Please choose from dt (Decision Tree) or ada (Adaboost).")
                    print("lab3.py train <examples> <features.txt> <hypothesis> <learning-type>")
                    sys.exit(1)
                ex_file = args[1]
                feature_list = set()
                keywords = set()
                input_data = []
                try:
                    feat_file = args[2]
                    get_features(feat_file)
                    example_list = get_examples(ex_file,False)
                    hypo_file = args[3]
                    # open(hypo_file, 'w', encoding='utf-8').write("test")
                except FileNotFoundError as e:
                    print(e)
                    sys.exit(1)

                if data_type == 'dt':
                    for ex in example_list:
                        input_data.append((get_word_features(ex, keywords), example_list[ex]))
                    best_feat = dt(input_data,feature_list)
                    dt = generate_tree(best_feat)
                    # print(dt)
                    json.dump(dt, open(hypo_file, 'w'))
                elif data_type == 'ada':
                    for ex in example_list:
                        input_data.append((get_word_features(ex, keywords), example_list[ex]))
                    json.dump(ada(input_data,feature_list), open(hypo_file, 'w'))

        else:
            print('Invalid operation')
            sys.exit()
    else:
        print('bruh')