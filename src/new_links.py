
import json
from collections import Counter
import numpy as np
import pandas as pd
with open("../results/instruction_embedding/cluster_labels/top_words_sci_GPT_fixed.json", "r") as f:
    top_words_sci = json.load(f)
with open("../results/instruction_embedding/cluster_labels/top_words_ai_GPT_fixed.json", "r") as f:
    top_words_ai = json.load(f)
with open("../results/instruction_embedding/cluster_labels/science_problem_final.json", "r") as f:
    cluster_labels_sci = json.load(f)
with open("../results/instruction_embedding/cluster_labels/AI_method_final.json", "r") as f:
    cluster_labels_ai = json.load(f)
with open("../results/instruction_embedding/cluster_labels/over_ai_indices.json", "r") as f:
    well_ai = json.load(f)
with open("../results/instruction_embedding/cluster_labels/under_ai_indices.json", "r") as f:
    under_ai = json.load(f)
with open("../results/instruction_embedding/cluster_labels/over_sci_indices.json", "r") as f:
    well_sci = json.load(f)
with open("../results/instruction_embedding/cluster_labels/under_sci_indices.json", "r") as f:
    under_sci = json.load(f)

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
        
def parse_name_size(name):
    idx = int(name.split("_")[-1])
    if "Sci" in name:
        return len([i for i in cluster_labels_sci if i == idx])
    else:
        return len([i for i in cluster_labels_ai if i == idx])

def count_new_links(train_links,pred_links):
    train_link_set = set(train_links)
    pred_link_set = set(pred_links)
    diff = pred_link_set - train_link_set
    for i in diff:
        assert i in list(pred_link_set)
    return diff

def get_human_newlinks(train_links):
    new_links_path = f"../results/GPT4o_mini/link_prediction/test_set_sci_ai_gt.json"
    with open(new_links_path,'r') as f:
        link_pred = json.load(f)
    print(f"Human set has {len(link_pred)} links")
    link_pred = [(i[0],i[1]) for i in link_pred if "N/A" not in parse_name(i[0])]
    diff = count_new_links(train_links,link_pred)
    print(f"Human sci->ai: {len(diff)}")

def get_human_newlinks_merged(train_links):
    new_links_path = f"../results/GPT4o_mini/link_prediction/test_set_sci_ai_gt.json"
    new_links_path_ai_sci = f"../results/GPT4o_mini/link_prediction/test_set_ai_sci_gt.json"
    with open(new_links_path,'r') as f:
        link_pred = json.load(f)
    with open(new_links_path_ai_sci,'r') as f:
        link_pred_ai_sci = json.load(f)

    link_pred = [(i[0],i[1]) for i in link_pred if "N/A" not in parse_name(i[0]) and "N/A" not in parse_name(i[1])]
    link_pred_ai_sci = [(i[1],i[0]) for i in link_pred_ai_sci if "N/A" not in parse_name(i[0]) and "N/A" not in parse_name(i[1])]
    link_pred = link_pred + link_pred_ai_sci
    new_links = count_new_links(train_links,link_pred)
    print(f"Pred set has {len(link_pred)} links")
    print(f"LLM new links: {len(new_links)}")
    link_pred = [p for p in link_pred if p in new_links]
    print(f"New link pred set: {len(link_pred)}")
    counter_sci_ai = Counter(link_pred)
    most_common_pred = counter_sci_ai.most_common()
    rows = []
    for top_idx in range(len(most_common_pred)):
        start, end = most_common_pred[top_idx][0]
        frequency = most_common_pred[top_idx][1]
        if "N/A" not in start:
            rows.append({
            'start_name': parse_name(start),
            'end_name': parse_name(end),
            'frequency': frequency,
            'start_size': parse_name_size(start),
            'end_size': parse_name_size(end),
        })
    node2vec_sci_ai_df = pd.DataFrame(rows, columns=["start_name", "end_name", "frequency", "start_size", "end_size"])
    assert len(node2vec_sci_ai_df) == len(new_links)
    node2vec_sci_ai_df.to_csv(f"../results/GPT4o_mini/link_prediction/new_links_human.csv",index=False)

def get_human_newlinks_ai_sci(train_links):
    new_links_path = f"../results/GPT4o_mini/link_prediction/test_set_ai_sci_gt.json"
    with open(new_links_path,'r') as f:
        link_pred = json.load(f)
    print(f"Human set has {len(link_pred)} links")
    link_pred = [(i[0],i[1]) for i in link_pred if "N/A" not in parse_name(i[0])]
    diff = count_new_links(train_links,link_pred)
    print(f"Human ai->sci: {len(diff)}")

def get_LLM_newlinks_merged(train_links):
    for k in ["oneshot","threeshot","fiveshot","tenshot"]:
        print(f"Running experiment for {k}")
        new_links_path = f"../results/GPT4o_mini/link_prediction/LLM_sci_ai_{k}.json"
        new_links_path_ai_sci = f"../results/GPT4o_mini/link_prediction/LLM_ai_sci_{k}.json"
        with open(new_links_path,'r') as f:
            link_pred = json.load(f)
        with open(new_links_path_ai_sci,'r') as f:
            link_pred_ai_sci = json.load(f)

        link_pred = [(i[0],i[1]) for i in link_pred if "N/A" not in parse_name(i[0]) and "N/A" not in parse_name(i[1])]
        link_pred_ai_sci = [(i[1],i[0]) for i in link_pred_ai_sci if "N/A" not in parse_name(i[0]) and "N/A" not in parse_name(i[1])]
        link_pred = link_pred + link_pred_ai_sci
        new_links = count_new_links(train_links,link_pred)
        print(f"Pred set has {len(link_pred)} links")
        print(f"LLM new links: {len(new_links)}")
        link_pred = [p for p in link_pred if p in new_links]
        print(f"New link pred set: {len(link_pred)}")
        counter_sci_ai = Counter(link_pred)
        most_common_pred = counter_sci_ai.most_common()
        rows = []
        for top_idx in range(len(most_common_pred)):
            start, end = most_common_pred[top_idx][0]
            frequency = most_common_pred[top_idx][1]
            if "N/A" not in start:
                rows.append({
                'start_name': parse_name(start),
                'end_name': parse_name(end),
                'frequency': frequency,
                'start_size': parse_name_size(start),
                'end_size': parse_name_size(end),
            })
        node2vec_sci_ai_df = pd.DataFrame(rows, columns=["start_name", "end_name", "frequency", "start_size", "end_size"])
        assert len(node2vec_sci_ai_df) == len(new_links)
        node2vec_sci_ai_df.to_csv(f"../results/GPT4o_mini/link_prediction/new_links_LLM_{k}.csv",index=False)
        # for idx in range(len(link_gt)):
        #     pred = get_mapped_cluster(embedding,out[idx])
        #     gt_pool = set_pool_train[link_gt[idx][0]]
        #     new_links.extend([(link_gt_train[idx][0],p) for p in pred if p not in gt_pool])


def get_LLM_newlinks(train_links):
    for i in ["oneshot","threeshot","fiveshot","tenshot"]:
        print(f"Running experiment for {i}")
        new_links_path = f"../results/GPT4o_mini/link_prediction/LLM_sci_ai_{i}.json"
        with open(new_links_path,'r') as f:
            link_pred = json.load(f)
        links = []
        print(f"Pred set has {len(link_pred)} links")
        link_pred = [(i[0],i[1]) for i in link_pred if "N/A" not in parse_name(i[0])]
        new_links = count_new_links(train_links,link_pred)
        print(f"LLM sci->ai new links: {len(new_links)}")
        # for idx in range(len(link_gt)):
        #     pred = get_mapped_cluster(embedding,out[idx])
        #     gt_pool = set_pool_train[link_gt[idx][0]]
        #     new_links.extend([(link_gt_train[idx][0],p) for p in pred if p not in gt_pool])


def get_LLM_newlinks_ai_sci(train_links):
    for i in ["oneshot","threeshot","fiveshot","tenshot"]:
        print(f"Running experiment for {i}")
        new_links_path = f"../results/GPT4o_mini/link_prediction/LLM_ai_sci_{i}.json"
        with open(new_links_path,'r') as f:
            link_pred = json.load(f)
        links = []
        print(f"Pred set has {len(link_pred)} links")
        link_pred = [(i[0],i[1]) for i in link_pred if "N/A" not in parse_name(i[0])]
        new_links = count_new_links(train_links,link_pred)
        print(f"LLM ai->sci new links: {len(new_links)}")
        # for idx in range(len(link_gt)):
        #     pred = get_mapped_cluster(embedding,out[idx])
        #     gt_pool = set_pool_train[link_gt[idx][0]]
        #     new_links.extend([(link_gt_train[idx][0],p) for p in pred if p not in gt_pool])

def get_node2vec_newlinks(train_links):
    for k in [1,3,5,10]:
        new_links_path_sci_ai = f"../results/GPT4o_mini/link_prediction/node2vec_sci_ai_{k}.json"
        with open(new_links_path_sci_ai,'r') as f:
            link_pred = json.load(f)
        links = []
        print(f"Pred set has {len(link_pred)} links")
        link_pred = [(i[0],i[1]) for i in link_pred if "N/A" not in parse_name(i[0])]
        new_links = count_new_links(train_links,link_pred)
        print(f"Node2vec with k = {k} has {len(new_links)} new Sci -> AI links")
        node2vec_sci_ai = []
        for new_link in link_pred:
            start, end = new_link
            if "N/A" not in parse_name(start) and new_link in new_links:
                node2vec_sci_ai.append({
                    'start_name': parse_name(start),
                    'end_name': parse_name(end),
                    'start_size': parse_name_size(start),
                    'end_size': parse_name_size(end),
                })
        print(len(node2vec_sci_ai))

def get_node2vec_newlinks_ai_sci(train_links):
    for k in [1,3,5,10]:
        new_links_path_ai_sci = f"../results/GPT4o_mini/link_prediction/node2vec_ai_sci_{k}.json"
        with open(new_links_path_ai_sci,'r') as f:
            link_pred = json.load(f)
        links = []
        print(f"Pred set has {len(link_pred)} links")
        link_pred = [(i[0],i[1]) for i in link_pred if "N/A" not in parse_name(i[0])]
        new_links = count_new_links(train_links,link_pred)
        print(f"Node2vec with k = {k} has {len(new_links)} new AI -> Sci links")

def get_LLM_cluster_newlinks_ai_sci(train_links):
    for k in ["oneshot","threeshot","fiveshot","tenshot"]:
        new_links_path_ai_sci = f"../results/GPT4o_mini/link_prediction/LLM_cluster_ai_sci_examples_{k}_gpt4.json"
        with open(new_links_path_ai_sci,'r') as f:
            link_pred = json.load(f)
        links = []
        print(f"Pred set has {len(link_pred)} links")
        link_pred = [(i[0],i[1]) for i in link_pred if "N/A" not in parse_name(i[0])]
        new_links = count_new_links(train_links,link_pred)
        #print(new_links)
        print(f"LLM cluster with k = {k} has {len(new_links)} new AI -> Sci links")
        parse_link_well_under_sci_LLM(new_links)



def get_node2vec_links_merged(train_links):
    for k in [1,3,5,10]:
        print(f"Running experiment for {k}")
        new_links_path = f"../results/GPT4o_mini/link_prediction/node2vec_sci_ai_{k}.json"
        new_links_path_ai_sci = f"../results/GPT4o_mini/link_prediction/node2vec_ai_sci_{k}.json"
        with open(new_links_path,'r') as f:
            link_pred = json.load(f)
        with open(new_links_path_ai_sci,'r') as f:
            link_pred_ai_sci = json.load(f)

        link_pred = [(i[0],i[1]) for i in link_pred if "N/A" not in parse_name(i[0]) and "N/A" not in parse_name(i[1])]
        link_pred_ai_sci = [(i[1],i[0]) for i in link_pred_ai_sci if "N/A" not in parse_name(i[0]) and "N/A" not in parse_name(i[1])]
        link_pred = link_pred + link_pred_ai_sci
        new_links = count_new_links(train_links,link_pred)
        print(f"Pred set has {len(link_pred)} links")
        print(f"Node2Vec new links: {len(new_links)}")
        link_pred = [p for p in link_pred if p in new_links]
        print(f"New link pred set: {len(link_pred)}")
        counter_sci_ai = Counter(link_pred)
        most_common_pred = counter_sci_ai.most_common()
        rows = []
        for top_idx in range(len(most_common_pred)):
            start, end = most_common_pred[top_idx][0]
            if "N/A" not in start:
                rows.append({
                'start_name': parse_name(start),
                'end_name': parse_name(end),
                'start_size': parse_name_size(start),
                'end_size': parse_name_size(end),
            })
        node2vec_sci_ai_df = pd.DataFrame(rows, columns=["start_name", "end_name", "start_size", "end_size"])
        node2vec_sci_ai_df.to_csv(f"../results/GPT4o_mini/link_prediction/new_links_node2vec_{k}.csv",index=False)
        # for idx in range(len(link_gt)):
        #     pred = get_mapped_cluster(embedding,out[idx])
        #     gt_pool = set_pool_train[link_gt[idx][0]]
        #     new_links.extend([(link_gt_train[idx][0],p) for p in pred if p not in gt_pool])

def read(path):
    with open(path,'r') as f:
        link = json.load(f)
    link = [(i[0],i[1]) for i in link]
    return link

def parse_link_well_under_sci(links):
    well_clusters_sci = [top_words_sci[i] for i in well_sci]
    well_clusters_ai = [top_words_ai[i] for i in well_ai]
    under_clusters_sci = [top_words_sci[i] for i in under_sci]
    under_clusters_ai = [top_words_ai[i] for i in under_ai]
    start = links['start_name'].tolist()
    end = links['end_name'].tolist()
    links_list = [(start[i],end[i]) for i in range(len(start))]
    well_well_links = [link for link in links_list if (link[0] in well_clusters_sci or link[0] in well_clusters_ai) and (link[1] in well_clusters_sci or link[1] in well_clusters_ai)]
    under_under_links = [link for link in links_list if (link[0] in under_clusters_sci or link[0] in under_clusters_ai) and (link[1] in under_clusters_sci or link[1] in under_clusters_ai)]
    well_under_links = [link for link in links_list if (link[0] in well_clusters_sci or link[0] in well_clusters_ai) and (link[1] in under_clusters_sci or link[1] in under_clusters_ai)]
    under_well_links = [link for link in links_list if (link[0] in under_clusters_sci or link[0] in under_clusters_ai) and (link[1] in well_clusters_sci or link[1] in well_clusters_ai)]
    print(f"Total links: {len(links_list)}")
    print(f"Well-well links: {len(well_well_links)}")
    print(f"Under-under links: {len(under_under_links)}")
    print(f"Well-under links: {len(well_under_links)}")
    print(f"Under-well links: {len(under_well_links)}")


def parse_link_well_under_sci_LLM(links):
    well_clusters_sci = [top_words_sci[i] for i in well_sci]
    well_clusters_ai = [top_words_ai[i] for i in well_ai]
    under_clusters_sci = [top_words_sci[i] for i in under_sci]
    under_clusters_ai = [top_words_ai[i] for i in under_ai]
    def parse_name(string):
        idx = int(string.split("_")[-1])
        if "Sci" in string:
            return top_words_sci[idx]
        else:
            return top_words_ai[idx]
    links_list = [(parse_name(i[0]),parse_name(i[1])) for i in links]
    well_well_links = [link for link in links_list if (link[0] in well_clusters_sci or link[0] in well_clusters_ai) and (link[1] in well_clusters_sci or link[1] in well_clusters_ai)]
    under_under_links = [link for link in links_list if (link[0] in under_clusters_sci or link[0] in under_clusters_ai) and (link[1] in under_clusters_sci or link[1] in under_clusters_ai)]
    well_under_links = [link for link in links_list if (link[0] in well_clusters_sci or link[0] in well_clusters_ai) and (link[1] in under_clusters_sci or link[1] in under_clusters_ai)]
    under_well_links = [link for link in links_list if (link[0] in under_clusters_sci or link[0] in under_clusters_ai) and (link[1] in well_clusters_sci or link[1] in well_clusters_ai)]
    print(f"Total links: {len(links_list)}")
    print(f"Well-well links: {len(well_well_links)}")
    print(f"Under-under links: {len(under_under_links)}")
    print(f"Well-under links: {len(well_under_links)}")
    print(f"Under-well links: {len(under_well_links)}")

link_gt_train_path = "../results/GPT4o_mini/link_prediction/train_set_sci_ai.json"
with open(link_gt_train_path,'r') as f:
    link_gt_train = json.load(f)
print(f"Number of ground truth train links: {len(link_gt_train)}")
link_gt_train = [(i[0],i[1]) for i in link_gt_train]

link_gt_train_ai_sci_path = "../results/GPT4o_mini/link_prediction/train_set_ai_sci.json"

with open(link_gt_train_ai_sci_path,'r') as f:
    link_gt_train_ai_sci = json.load(f)

print(f"Number of ground truth train links: {len(link_gt_train_ai_sci)}")
link_gt_train_ai_sci = [(i[0],i[1]) for i in link_gt_train_ai_sci]

link_gt_test = read("../results/GPT4o_mini/link_prediction/test_set_sci_ai_gt.json")
# get_human_newlinks(link_gt_train)
# get_human_newlinks_ai_sci(link_gt_train_ai_sci)
# print(len(set(link_gt_train)))
# print(len(set(link_gt_train_ai_sci)))

# #get_LLM_newlinks_ai_sci(link_gt_train_ai_sci)
# get_node2vec_links_merged(link_gt_train)
# get_LLM_newlinks_merged(link_gt_train)
#get_human_newlinks_merged(link_gt_train)
# get_node2vec_newlinks(link_gt_train)
# get_node2vec_newlinks_ai_sci(link_gt_train_ai_sci)
# ks = [1,3,5,10]
# names = ["oneshot","threeshot","fiveshot","tenshot"]
# for k in range(len(ks)):
#     rows_sci_ai = []
#     rows_ai_sci = []
#     node2vec_sci_ai = []
#     node2vec_ai_sci = []
#     print(f"K = {ks[k]}")
#     new_links_path_sci_ai_node2vec = f"../results/GPT4o_mini/AI_recommendation/2023_after_fixed_node2vec_new_links_{ks[k]}.json"
#     new_links_path_ai_sci_node2vec = f"../results/GPT4o_mini/Science_recommendation/2023_after_fixed_node2vec_new_links_{ks[k]}.json"
#     new_links_path_sci_ai_LLM = f"../results/GPT4o_mini/AI_recommendation/2023_after_fixed_LLM_new_links_{names[k]}.json"
#     new_links_path_ai_sci_LLM = f"../results/GPT4o_mini/Science_recommendation/2023_after_fixed_LLM_new_links_{names[k]}.json"
#     with open(new_links_path_sci_ai_LLM,"r") as f:
#         new_links_sci_ai_LLM = json.load(f)
#     with open(new_links_path_ai_sci_LLM,"r") as f:
#         new_links_ai_sci_LLM = json.load(f)
#     with open(new_links_path_sci_ai_node2vec,"r") as f:
#         new_links_sci_ai_node2vec = json.load(f)
#     with open(new_links_path_ai_sci_node2vec,"r") as f:
#         new_links_ai_sci_node2vec = json.load(f)
#     print(len(np.unique(new_links_sci_ai_LLM)))
#     print(len(np.unique(new_links_ai_sci_LLM)))
#     new_links_sci_ai_LLM = [(i[0],i[1]) for i in new_links_sci_ai_LLM]
#     new_links_ai_sci_LLM = [(i[0],i[1]) for i in new_links_ai_sci_LLM]
#     counter_sci_ai = Counter(new_links_sci_ai_LLM)
#     counter_ai_sci = Counter(new_links_ai_sci_LLM)
#     print(f"Before: {len(new_links_sci_ai_LLM)}")
#     most_common_element_sci_ai = counter_sci_ai.most_common()  # Returns a list of tuples [(element, count)]
#     print(f"After: {len(counter_ai_sci)}")
#     most_common_element_ai_sci = counter_ai_sci.most_common()  # Returns a list of tuples [(element, count)]
#     print(f"Top Sci to AI links")
#     for new_link in new_links_sci_ai_node2vec:
#         start, end = new_link
#         if "N/A" not in start:
#             node2vec_sci_ai.append({
#                 'start_name': parse_name(start),
#                 'end_name': parse_name(end),
#                 'start_size': parse_name_size(start),
#                 'end_size': parse_name_size(end),
#             })
#     for new_link in new_links_ai_sci_node2vec:
#         start, end = new_link
#         if "N/A" not in start:
#             node2vec_ai_sci.append({
#                 'start_name': parse_name(start),
#                 'end_name': parse_name(end),
#                 'start_size': parse_name_size(start),
#                 'end_size': parse_name_size(end),
#             })
#     for top_idx in range(len(most_common_element_sci_ai)):
#         start, end = most_common_element_sci_ai[top_idx][0]
#         frequency = most_common_element_sci_ai[top_idx][1]
#         if "N/A" not in start:
#             rows_sci_ai.append({
#                 'start_name': parse_name(start),
#                 'end_name': parse_name(end),
#                 'frequency': frequency,
#                 'start_size': parse_name_size(start),
#                 'end_size': parse_name_size(end),
#             })
#         #print(f"{parse_name(start)} -> {parse_name(end)} frequency: {frequency} start size: {parse_name_size(start)} end size: {parse_name_size(end)}")
#     print(f"Top AI to Sci links")
#     for top_idx in range(len(most_common_element_ai_sci)):
#         start, end = most_common_element_ai_sci[top_idx][0]
#         frequency = most_common_element_sci_ai[top_idx][1]
#         if "N/A" not in start:
#             rows_ai_sci.append({
#                 'start_name': parse_name(start),
#                 'end_name': parse_name(end),
#                 'frequency': frequency,
#                 'start_size': parse_name_size(start),
#                 'end_size': parse_name_size(end),
#             })
#         #print(f"{parse_name(start)} -> {parse_name(end)} frequency: {frequency} start size: {parse_name_size(start)} end size: {parse_name_size(end)}")
#     df_sci_ai = pd.DataFrame(rows_sci_ai, columns=["start_name", "end_name", "frequency", "start_size", "end_size"])
#     df_ai_sci = pd.DataFrame(rows_ai_sci, columns=["start_name", "end_name", "frequency", "start_size", "end_size"])
#     node2vec_sci_ai_df = pd.DataFrame(node2vec_sci_ai, columns=["start_name", "end_name", "start_size", "end_size"])
#     node2vec_ai_sci_df = pd.DataFrame(node2vec_ai_sci, columns=["start_name", "end_name", "start_size", "end_size"])
#     df_ai_sci.to_excel(f"../results/GPT4o_mini/link_prediction/new_links_LLM_ai_sci_{ks[k]}.xlsx",index=False)
#     df_sci_ai.to_excel(f"../results/GPT4o_mini/link_prediction/new_links_LLM_sci_ai_{ks[k]}.xlsx",index=False)
#     node2vec_sci_ai_df.to_excel(f"../results/GPT4o_mini/link_prediction/new_links_node2vec_sci_ai_{ks[k]}.xlsx",index=False)
#     node2vec_ai_sci_df.to_excel(f"../results/GPT4o_mini/link_prediction/new_links_node2vec_ai_sci_{ks[k]}.xlsx",index=False)
#     df_sci_ai = df_sci_ai.drop_duplicates(subset=["start_name","end_name"])
#     df_ai_sci = df_ai_sci.drop_duplicates(subset=["start_name","end_name"])
#     node2vec_sci_ai_df = node2vec_sci_ai_df.drop_duplicates(subset=["start_name","end_name"])
#     node2vec_ai_sci_df = node2vec_ai_sci_df.drop_duplicates(subset=["start_name","end_name"])
#     print(f"LLM predicts {len(df_sci_ai)} sci to ai links")
#     print(f"Node2Vec predicts {len(node2vec_sci_ai_df)} sci to ai links")
#     print(f"LLM predicts {len(df_ai_sci)} ai to sci links")
#     print(f"Node2Vec predicts {len(node2vec_ai_sci_df)} ai to sci links")
#     print("="*100)
# for i in ["oneshot","threeshot","fiveshot","tenshot"]:
#     new_links = pd.read_csv(f"../results/GPT4o_mini/link_prediction/new_links_LLM_{i}.csv")
#     parse_link_well_under_sci(new_links)
#     print("="*100)
# for i in [1,3,5,10]:
#     new_links = pd.read_csv(f"../results/GPT4o_mini/link_prediction/new_links_node2vec_{i}.csv")
#     parse_link_well_under_sci(new_links)
#     print("="*100)

# for i in [1,3,5,10]:
#     new_links = pd.read_csv(f"../results/GPT4o_mini/link_prediction/new_links_node2vec_{i}.csv")
#     parse_link_well_under_sci(new_links)
#     print("="*100)

get_LLM_cluster_newlinks_ai_sci(link_gt_train_ai_sci)
    