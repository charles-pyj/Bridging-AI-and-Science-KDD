from node2vec import Node2Vec
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

# Load top words for clusters
with open("../results/instruction_embedding/cluster_labels/top_words_sci_GPT_fixed.json", "r") as f:
    top_words_sci = json.load(f)
with open("../results/instruction_embedding/cluster_labels/top_words_ai_GPT_fixed.json", "r") as f:
    top_words_ai = json.load(f)

# Function to parse node names
def parse_name(name):
    idx = int(name.split("_")[-1])
    if "Sci" in name:
        if len(top_words_sci[idx]) > 0:
            return top_words_sci[idx]
        else:
            return []
    else:
        if len(top_words_ai[idx]) > 0:
            return top_words_ai[idx]
        else:
            return []

def parse_target_set(link_gt):
    set_pool = {}
    for edge in link_gt:
        start, end = edge
        if start not in set_pool.keys():
            set_pool[start] = []
        set_pool[start].append(end)
    for key in set_pool.keys():
        set_pool[key] = np.unique(set_pool[key])
    return set_pool
def parse_source_ratio(link_gt,node):
    return len([i for i in link_gt if i[0] == node]) / len(link_gt)

# Step 1: Load your edge list with weights
edges_all = pd.read_csv("../results/instruction_embedding/edges/kdd_sci_ai_train.csv")
start = edges_all['start'].tolist()
end = edges_all['end'].tolist()
weights = edges_all['weight'].tolist()
edges = [(start[i], end[i], weights[i]) for i in range(len(start))]

# Step 2: Create a Directed Bipartite Graph
G = nx.Graph()  # Use DiGraph for directed graphs

# Step 3: Add directed weighted edges from the edge list
G.add_weighted_edges_from(edges)

# Define the two bipartite sets (optional for analysis, not strictly needed for Node2Vec)
U = set([start for start, _, _ in edges])  # First set of nodes (Sci)
V = set([end for _, end, _ in edges])      # Second set of nodes (AI)

# Convert sets to lists
all_nodes = np.unique(edges_all['start'].tolist() + edges_all['end'].tolist())
U_list = [i for i in all_nodes if "Sci" in i]
V_list = [i for i in all_nodes if "AI" in i]
print(len(U_list))
print(len(V_list))
# Step 4: Initialize Node2Vec on the directed bipartite graph with directed=True
node2vec = Node2Vec(G, dimensions=256, walk_length=20, num_walks=500, weight_key='weight', workers=4)

# Step 5: Train the Node2Vec model to generate node embeddings
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# Step 6: Get the embeddings as a dictionary
embeddings = {node: model.wv[node] for node in G.nodes()}

# Step 7: Create a matrix of embeddings for AI methods (right-side nodes in bipartite graph)
embeddings_AI_methods = np.vstack([embeddings[i] for i in V_list])
embedding_sci = np.vstack([embeddings[i] for i in U_list])
print(embeddings_AI_methods.shape)
def pairwise_cosine_similarity(matrix):
    # Normalize the matrix rows
    norm_matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    
    # Compute pairwise cosine similarity
    cosine_similarity = np.dot(norm_matrix, norm_matrix.T)
    
    return cosine_similarity
# Step 8: Define the function to predict top links based on cosine similarity
def csv_to_list(edges):
    start = edges['start'].tolist()
    end = edges['end'].tolist()
    return [(start[i], end[i]) for i in range(len(start))]

# Function to predict top links starting from a given node
def predict_top_links_sci(node, k=1):
    # Get the embedding for the specified node
    embedding_node = embeddings[node]
    # Calculate cosine similarity with all embeddings in the AI set
    similarity = cosine_similarity(embedding_node.reshape(1, -1), embeddings_AI_methods).squeeze()
    # Get indices of top k similar nodes
    indices = np.argsort(similarity)[::-1]
    #print(np.sort(similarity)[::-1][:5])
    return [V_list[idx] for idx in indices[:k]]

def predict_top_links_ai(node, k=1):
    # Get the embedding for the specified node
    embedding_node = embeddings[node]
    # Calculate cosine similarity with all embeddings in the AI set
    similarity = cosine_similarity(embedding_node.reshape(1, -1), embedding_sci).squeeze()
    # Get indices of top k similar nodes
    indices = np.argsort(similarity)[::-1]
    return [U_list[idx] for idx in indices[:k]]

def filter_indices(links,indices):
    def parse_idx(string):
        return int(string.split("_")[-1])
    return [l for l in links if parse_idx(l[0]) in indices]

def sci_ai_under_well(links_gt_sci_ai,indices,name="None"):
    print(len(links_gt_sci_ai))
    links_gt_sci_ai = filter_indices(links_gt_sci_ai,indices)
    print(len(links_gt_sci_ai))
    gt_pool_sci_ai = parse_target_set(links_gt_sci_ai)
    for k in [1, 3, 5, 10]:
        print(f"K = {k}")
        precision_sci_ai = 0
        recall_sci_ai = 0
        ratio_sum = 0
        edges_dict_sci_ai = {}
        for i in U_list:
            # Predict top k links for each node in set U
            if i not in edges_dict_sci_ai.keys():
                edges_dict_sci_ai[i] = []
            edges_dict_sci_ai[i].extend(predict_top_links_sci(i, k=k))
        for start in edges_dict_sci_ai.keys():
            if start in gt_pool_sci_ai.keys():
                gt_pool = gt_pool_sci_ai[start]
                pred = edges_dict_sci_ai[start]
                assert len(pred) == k
                precision_sci_ai += (len([p for p in pred if p in gt_pool]) / len(pred)) * parse_source_ratio(links_gt_sci_ai,start)
                recall_sci_ai += (len([gt for gt in gt_pool if gt in pred]) / len(gt_pool)) * parse_source_ratio(links_gt_sci_ai,start)
                ratio_sum += parse_source_ratio(links_gt_sci_ai,start)
        print(f"Ratio sum: {ratio_sum}")
        print(f"Recall sci->ai {name}: {recall_sci_ai}")
        print(f"Precision sci->sci {name}: {precision_sci_ai}")
        print("="*100)

def ai_sci_under_well(links_gt_ai_sci,indices,name="None"):
    print(len(links_gt_ai_sci))
    links_gt_ai_sci = filter_indices(links_gt_ai_sci,indices)
    print(len(links_gt_ai_sci))
    gt_pool_ai_sci = parse_target_set(links_gt_ai_sci)
    for k in [1, 3, 5, 10]:
        print(f"K = {k}")
        precision_ai_sci = 0
        recall_ai_sci = 0
        ratio_sum = 0
        edges_dict_ai_sci = {}
        for i in V_list:
            if i not in edges_dict_ai_sci.keys():
                edges_dict_ai_sci[i] = []
            edges_dict_ai_sci[i].extend(predict_top_links_ai(i, k=k))
        for start in edges_dict_ai_sci.keys():
            if start in gt_pool_ai_sci.keys():
                gt_pool = gt_pool_ai_sci[start]
                pred = edges_dict_ai_sci[start]
                assert len(pred) == k
                precision_ai_sci += (len([p for p in pred if p in gt_pool]) / len(pred)) * parse_source_ratio(links_gt_ai_sci,start)
                recall_ai_sci += (len([gt for gt in gt_pool if gt in pred]) / len(gt_pool)) * parse_source_ratio(links_gt_ai_sci,start)
                ratio_sum += parse_source_ratio(links_gt_ai_sci,start)
        print(f"Ratio sum: {ratio_sum}")
        print(f"Recall ai->sci {name}: {recall_ai_sci}")
        print(f"Precision ai->sci {name}: {precision_ai_sci}")
        print("="*100)

link_gt_path = "../results/instruction_embedding/edges/kdd_sci_ai_test.json"
with open(link_gt_path,'r') as f:
    link_gt = json.load(f)
link_gt_path_ai_sci = "../results/instruction_embedding/edges/kdd_ai_sci_test.json"
with open(link_gt_path_ai_sci,'r') as f:
    link_gt_ai_sci = json.load(f)
with open("../results/instruction_embedding/cluster_labels/under_sci_kdd.json","r") as f:
    under_sci_indices = json.load(f)
with open("../results/instruction_embedding/cluster_labels/over_sci_kdd.json","r") as f:
    over_sci_indices = json.load(f)
with open("../results/instruction_embedding/cluster_labels/under_ai_kdd.json","r") as f:
    under_ai_indices = json.load(f)
with open("../results/instruction_embedding/cluster_labels/over_ai_kdd.json","r") as f:
    over_ai_indices = json.load(f)
# sci_ai_under_well(link_gt,over_sci_indices,"Well")
# ai_sci_under_well(link_gt_ai_sci,under_ai_indices,"Under")
# gt_pool_sci_ai = parse_target_set(link_gt)
# gt_pool_ai_sci = parse_target_set(link_gt_ai_sci)
# print(f"Length of sci-> AI edges: {len(link_gt)}")
# print(f"Length of AI-> sci edges: {len(link_gt_ai_sci)}")
def get_ref(data):
    start_ref = data['start'].tolist()
    end_ref = data['end'].tolist()
    train_edges = [[start_ref[i],end_ref[i]] for i in range(len(data))]
    return train_edges
train_edges = get_ref(pd.read_csv("../results/instruction_embedding/edges/kdd_sci_ai_train.csv"))
all_edges = get_ref(pd.read_csv("../results/instruction_embedding/edges/kdd_sci_ai_all.csv"))
t_edges = get_ref(pd.read_csv("../results/instruction_embedding/edges/kdd_sci_ai_test.csv"))
print(len(t_edges))
print(len([t for t in t_edges if t not in train_edges]))
#Step 9: Evaluate Precision and Recall at Different Values of k
# for k in [1, 3, 5, 10]:
#     with open(f"../results/GPT4o_mini/link_prediction_kdd/node2vec_sci_ai_{k}.json","r") as f:
#         predicted_links = json.load(f)
#     print(f"Predicted link length: {len(predicted_links)}")
#     print(f"Predicted in test: {len([l for l in predicted_links if l in train_edges])}")
#     print(f"Predicted new: {len([l for l in predicted_links if l not in train_edges])}")
# for k in [1, 3, 5, 10]:
#     print(f"K = {k}")
#     precision_sci_ai = 0
#     recall_sci_ai = 0
#     edges_dict_sci_ai = {}
#     edges_dict_ai_sci = {}
#     for i in U_list:
#         # Predict top k links for each node in set U
#         if i not in edges_dict_sci_ai.keys():
#             edges_dict_sci_ai[i] = []
#         edges_dict_sci_ai[i].extend(predict_top_links_sci(i, k=k))
#     for i in V_list:
#         if i not in edges_dict_ai_sci.keys():
#             edges_dict_ai_sci[i] = []
#         edges_dict_ai_sci[i].extend(predict_top_links_ai(i, k=k))
#     links_sci_ai_predicted = []
#     links_ai_sci_predicted = []
#     for start,v in edges_dict_sci_ai.items():
#         links_sci_ai_predicted.extend([[start,dest] for dest in v])
#     for start,v in edges_dict_ai_sci.items():
#         links_ai_sci_predicted.extend([[start,dest] for dest in v])
#     print(len(links_sci_ai_predicted))
#     print(len(links_ai_sci_predicted))
#     new_links_sci_ai = [l for l in links_sci_ai_predicted if l not in train_edges]
#     new_links_sci_ai_all = [l for l in links_sci_ai_predicted if l not in all_edges]
#     print(f"New Sci->AI ref train{len(new_links_sci_ai)}")
#     print(f"New Sci->AI ref all{len(new_links_sci_ai_all)}")
#     with open(f"../results/GPT4o_mini/link_prediction_kdd/node2vec_sci_ai_{k}.json","w") as f:
#             json.dump(links_sci_ai_predicted,f)
    # for start in edges_dict_sci_ai.keys():
    #     if start in gt_pool_sci_ai.keys():
    #         gt_pool = gt_pool_sci_ai[start]
    #         pred = edges_dict_sci_ai[start]
    #         assert len(pred) == k
    #         precision_sci_ai += (len([p for p in pred if p in gt_pool]) / len(pred)) * parse_source_ratio(link_gt,start)
    #         recall_sci_ai += (len([gt for gt in gt_pool if gt in pred]) / len(gt_pool)) * parse_source_ratio(link_gt,start)
    
    # precision_ai_sci = 0
    # recall_ai_sci = 0
    # ratio_sum = 0
    # for start in edges_dict_ai_sci.keys():
    #     if start in gt_pool_ai_sci.keys():
    #         gt_pool = gt_pool_ai_sci[start]
    #         pred = edges_dict_ai_sci[start]
    #         assert len(pred) == k
    #         precision_ai_sci += (len([p for p in pred if p in gt_pool]) / len(pred)) * parse_source_ratio(link_gt_ai_sci,start)
    #         recall_ai_sci += (len([gt for gt in gt_pool if gt in pred]) / len(gt_pool)) * parse_source_ratio(link_gt_ai_sci,start)
    #         ratio_sum += parse_source_ratio(link_gt_ai_sci,start)
    # print(f"Ratio sum: {ratio_sum}")
    # print(f"Recall sci->ai: {recall_sci_ai}")
    # print(f"Precision sci->sci: {precision_sci_ai}")
    # print(f"Recall ai->sci: {recall_ai_sci}")
    # print(f"Precision ai->sci: {precision_ai_sci}")
    # print("="*100)

def get_new_links():
    for k in [1]:
        print(f"K = {k}")
        edges_dict_sci_ai = {}
        edges_dict_ai_sci = {}
        for i in U_list:
            # Predict top k links for each node in set U
            if i not in edges_dict_sci_ai.keys():
                edges_dict_sci_ai[i] = []
            edges_dict_sci_ai[i].extend(predict_top_links_sci(i, k=k))
        for i in V_list:
            if i not in edges_dict_ai_sci.keys():
                edges_dict_ai_sci[i] = []
            edges_dict_ai_sci[i].extend(predict_top_links_ai(i, k=k))
        new_links_path_sci_ai = f"../results/GPT4o_mini/link_prediction/node2vec_sci_ai_{k}.json"
        new_links_path_ai_sci = f"../results/GPT4o_mini/link_prediction/node2vec_ai_sci_{k}.json"
        new_link_sci_ai = []
        new_link_ai_sci = []
        for start in edges_dict_sci_ai.keys():
            pred = edges_dict_sci_ai[start]
            new_link_sci_ai.extend([(start,p) for p in pred])
        for start in edges_dict_ai_sci.keys():
            pred = edges_dict_ai_sci[start]
            new_link_ai_sci.extend([(start,p) for p in pred])
        
        print(f"New link from sci to ai: {len(new_link_sci_ai)}")
        print(f"New link from ai to sci: {len(new_link_ai_sci)}")
        # with open(new_links_path_sci_ai,"w") as f:
        #     json.dump(new_link_sci_ai,f)
        # with open(new_links_path_ai_sci,"w") as f:
        #     json.dump(new_link_ai_sci,f)
        # print(f"Ratio sum: {ratio_sum}")
        # print(f"Recall sci->ai: {recall_sci_ai}")
        # print(f"Precision sci->sci: {precision_sci_ai}")
        # print(f"Recall ai->sci: {recall_ai_sci}")
        # print(f"Precision ai->sci: {precision_ai_sci}")
        # print("="*100)
        # Calculate precision and recall
#get_new_links()
