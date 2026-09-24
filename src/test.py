import matplotlib.pyplot as plt
import pickle
from argparse import ArgumentParser

from utils import read_trees_json, load_config_file
from tree import Node
from symbol_library import generate_symbol_library

parser = ArgumentParser(prog='Data set generation', description='Generate a set of trees')
parser.add_argument("-config", default="../configs/config.json")
args = parser.parse_args()

config = load_config_file(args.config)
data_config = config["data_definition"]
ds_config = config["data_set_generation"]
dataset = config["dataset"]

symbols = data_config["expr_symbols"] if dataset == "expr" else data_config["neuro_symbols"]
sy_lib = generate_symbol_library(data_config["num_variables"], symbols, dataset, data_config["has_constants"])
Node.add_symbols(sy_lib)

# Example dummy trees list (replace with your actual dataset `t`)
trees = read_trees_json("../data/40k_expr.json")
with open("40k_expr_list.pkl", 'wb') as file:
    pickle.dump([t.to_list('expr') for t in trees], file)
file.close()
input("done")


# Example dummy node counting function (replace with your actual function `f`)
def f(tree):
    # f returns node count of a tree
    pass

# 1. Define the range bins and labels
bins = [(0, 15), (16, 20), (21, 25), (26, 30), (31, 35)]
labels = ['0-15', '16-20', '21-25', '26-30', '31-35']
counts = {label: 0 for label in labels}

# 2. Count nodes and categorize each tree
for tree in t:
    node_count = len(tree)
    for (low, high), label in zip(bins, labels):
        if low <= node_count <= high:
            counts[label] += 1
            break

# 3. Calculate percentages
total_trees = len(t)
percentages = {label: (count / total_trees) * 100 if total_trees > 0 else 0
               for label, count in counts.items()}

# Print results
print("Node Count Range Percentages:")
for label, pct in percentages.items():
    print(f"  {label} nodes: {pct:.2f}%")

# 4. Plot the results as a bar chart
plt.figure(figsize=(8, 5))
bars = plt.bar(percentages.keys(), percentages.values(), color='skyblue', edgecolor='black')

plt.title('Percentage of Trees by Node Count Range')
plt.xlabel('Node Count Ranges')
plt.ylabel('Percentage (%)')
plt.ylim(0, 100)

# Add percentage values above each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()