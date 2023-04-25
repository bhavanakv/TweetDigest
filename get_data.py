import tensorflow as tf
import transformers as ts
import logging
from treelib import Tree,Node
import subprocess

cache_dir = 'cache/'
logger = logging.getLogger("service")
logging.basicConfig(level=logging.INFO)
logger.setLevel('INFO')

classifier = ts.pipeline('zero-shot-classification', cache_dir=cache_dir)

# Labels that tweets will be classified into
labels = ['Technology', 'Politics', 'Entertainment', 'Sports', 'Science']
tree = {
    'Technology': ['Hardware', 'Software', 'AI'],
    'Politics': ['Domestic', 'International'],
    'Entertainment': 
        {
            'Movies': ['Drama', 'Comedy', 'Action'], 
            'Music': ['Pop', 'Rock', 'Hip Hop'], 
            'TV Shows': ['Reality', 'Comedy', 'Thriller']
        },
    'Sports': 
        {
            'Team sports':['Football', 'Basketball', 'Tennis'],
            'Individual': ['Chess', 'Golf']
        },
    'Science': ['Physics', 'Chemistry', 'Biology']
}

# Calling function to fetch tweets
tweets_data = 'US Election is going to be in 2024. The election is held accross the united states. The votes are collected and then counted and the next president is revealed. There would be lots of music and dancing once the winner is revealed.' + 'A famous pop star will be seen. All the fampus songs wil be played for everyone to watch'
logging.info("***Fetching tweets***")
# Classifying the tweets received and returning classifier response
response = classifier(tweets_data, labels)
classified_labels = {}
print(response)
for i in range(len(response['scores'])):
    if response['scores'][i] > 0.2:
        predicted_label = response['labels'][i]
        if predicted_label == 'Entertainment' or predicted_label == 'Sports':
            classified_labels[predicted_label] = {}
        else: 
            classified_labels[predicted_label] = []
print(classified_labels)
for label in classified_labels:
    if isinstance(tree[label],dict):
        labels = list(tree[label].keys())
    else:
        labels = tree[label]
    print(labels)
    response = classifier(tweets_data, labels)
    second_level_labels = []
    for i in range(len(response['scores'])):
        if response['scores'][i] > 0.4:
            if label == 'Entertainment' or label == 'Sports':
                print(label, response['labels'][i])
                classified_labels[label][response['labels'][i]] = []
                inner_labels = tree[label][response['labels'][i]]
                inner_response = classifier(tweets_data, inner_labels)
                for j in range(len(response['scores'])):
                    if inner_response['scores'][j] > 0.4:
                        classified_labels[label][response['labels'][i]].append(inner_response['labels'][j])
            else:
                classified_labels[label].append(response['labels'][i])

print(classified_labels)

# Create a new tree
tree = Tree()

# Add the root node
tree.create_node("Root", "root")

# Recursively add nodes to the tree
def add_nodes(parent_node_id, node_dict):
    for key, value in node_dict.items():
        if isinstance(value, dict):
            # Add a new node for the current key
            node_id = f"{parent_node_id}_{key}"
            tree.create_node(key, node_id, parent=parent_node_id)
            # Recursively add child nodes
            add_nodes(node_id, value)
        else:
            # Add a new leaf node for the current value
            leaf_id = f"{parent_node_id}_{value}"
            tree.create_node(value, leaf_id, parent=parent_node_id)

# Call the add_nodes function with the root node and dictionary
add_nodes("root", classified_labels)

tree.show()
# Render the tree as an HTML file
tree.to_graphviz(filename="tree.dot")
subprocess.call(["dot", "-Tpng", "tree.dot", "-o", "graph1.png"])
