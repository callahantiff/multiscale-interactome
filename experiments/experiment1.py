#!/usr/bin/env python
# -*- coding: utf-8 -*-


# import and load needed scripts
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pickle
import seaborn as sns

from experiments.utilities import *

# plot settings
plt.style.use('ggplot')


######## ENVIRONMENT SET-UP ########
# read in the needed data structures
dp_matrix = np.load("results/msi_diffusion_profile_matrix.npy")
dp_matrix_adj = np.load("results/msi_diffusion_profile_matrix_adjusted.npy")
node_idx_dict = pickle.load(open("results/msi_diffusion_profile_matrix_index_ids.npy", "rb"))
node_labels = pickle.load(open("results/msi_graph_node2name.pkl", "rb"))
node_types = pickle.load(open("results/msi_graph_node2type.pkl", "rb"))
msi_graph = pickle.load(open("results/msi_graph.pkl", "rb"))







# create histograms
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
y = sns.kdeplot(np.log10(cond_coverage_sets['cp_only']), color='#CC79A7', label='Concept Prevalence Only', shade=True, linewidth=2, alpha=0.2)
y = sns.kdeplot(np.log10(cond_coverage_sets['omop2obo_only']), color='#56B4E9', label='OMO2OBO Only', shade=True, linewidth=2, alpha=0.2)
y = sns.kdeplot(np.log10(cond_coverage_sets['overlap']), color='#E69f00', label='Overlap', shade=True, linewidth=2, alpha=0.2)

plt.title('OMOP2OBO - Concept Prevalence Coverage: Condition Concept Frequency\n', fontsize=18)
plt.suptitle('')
plt.xlabel('Concept Frequency ($log_{10}$)', fontsize=30, fontname="Arial", color='black')
plt.ylabel('Density\n', fontsize=28, fontname="Arial", color='black')
plt.ylim(0.0, 5.25)
plt.tick_params(labelsize=24)
plt.yticks(color='black', fontname="Arial", fontsize=30)
plt.xticks(color='black', fontname="Arial", fontsize=30)
plt.legend(title='', prop={"family":"Arial", "size": 30}, facecolor='white', edgecolor='darkgray', ncol=3,
           loc='lower center', bbox_to_anchor=(0.5,-0.40))
plt.show()


######## DESCRIPTIVE STATISTICS ########
### msi graph degree
degree_list = [x[1] for x in list(msi_graph.degree())]
print(gets_simple_statistics(degree_list, "msi graph", "degree"))

# create histograms
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
y = sns.kdeplot(degree_list, color='#56B4E9', label='Degree', shade=True, linewidth=2, alpha=0.2)
plt.title('MSI Graph - Node Degree\n', fontsize=16, fontname="Arial", color='black')
plt.suptitle('')
plt.xlabel('Degree', fontsize=14, fontname="Arial", color='black')
plt.ylabel('Density\n', fontsize=14, fontname="Arial", color='black')
plt.ylim(0.0, 0.014)
plt.tick_params(labelsize=14)
plt.yticks(color='black', fontname="Arial", fontsize=14)
plt.xticks(color='black', fontname="Arial", fontsize=14)
plt.show()

### diffusion profiles
## importance (including self-importance)
node_mean = dp_matrix.mean(1)
full_mean = np.mean(node_mean)
print(gets_simple_statistics(node_mean, "diffusion profile (w/self-importance)", "importance"))



#### METHODS TO SCORE NODES
# obtain diffusion profiles for specific nodes by index, which is stored in the node_idx_dict
# lisinopril (DB00722)
lis_concept_id = [k for k, v in node_labels.items() if v.lower() == "lisinopril"][0]  # DB00722
lis_index = [k for k, v in node_idx_dict.items() if v == lis_concept_id][0]
# myocardial infarction (C0027051)
mi_concept_id = [k for k, v in node_labels.items() if v.lower() == "myocardial infarction"][0]  # C0027051
mi_index = [k for k, v in node_idx_dict.items() if v == mi_concept_id][0]

# extract each node's diffusion profile from matrix
dp_imp_lis = dp_matrix[lis_index]
dp_imp_mi = dp_matrix[mi_index]


### Approach 1: Diffusion Profile Importance Scores
## task 1 - look at importance between two nodes
print(round(dp_imp_lis[mi_index], 9))
print(round(dp_imp_mi[lis_index], 9))
print(round(float(dp_imp_lis[mi_index] + dp_imp_mi[lis_index]) / 2.0, 9))

## task 2 - top-25 most important nodes
# lisinopril
idx = lis_index; node_data = dp_matrix[idx]; topn = 100
top_n_results = [[i, node_data[i]] for i in np.argsort(node_data)[-topn + 1:] if i != idx]
top_n_results.sort(key=lambda x: x[1], reverse=True)
results_formatter(top_n_results[0:10], node_idx_dict, node_labels, node_types)
results_formatter(top_n_results, node_idx_dict, node_labels, node_types, ['drug'])
results_formatter(top_n_results, node_idx_dict, node_labels, node_types, ['indication'])
results_formatter(top_n_results, node_idx_dict, node_labels, node_types, ['biological_function'])
results_formatter(top_n_results[0:20], node_idx_dict, node_labels, node_types, ['protein'])

# myocardial infarction
idx = mi_index; node_data = dp_matrix[idx]; topn = 500
top_n_results = [[i, node_data[i]] for i in np.argsort(node_data)[-topn + 1:] if i != idx]
top_n_results.sort(key=lambda x: x[1], reverse=True)
results_formatter(top_n_results[0:25], node_idx_dict, node_labels, node_types)
results_formatter(top_n_results, node_idx_dict, node_labels, node_types, ['drug'])
results_formatter(top_n_results, node_idx_dict, node_labels, node_types, ['indication'])
results_formatter(top_n_results, node_idx_dict, node_labels, node_types, ['biological_function'])
results_formatter(top_n_results[0:10], node_idx_dict, node_labels, node_types, ['protein'])



### Approach 2: Diffusion Profile Similarity
## task 1 - look at cosine similarity between two nodes
# 2A: including self-importance
n1 = dp_imp_lis; n2 = dp_imp_mi
print(round(dot(n1, n2)/(norm(n1)*norm(n2)), 10))

# 2B: excluding self-importance
lis_mod = np.array(list(dp_imp_lis)[0:lis_index] + list(dp_imp_lis)[lis_index + 1:])
mi_mod = np.array(list(dp_imp_mi)[0:mi_index] + list(dp_imp_mi)[mi_index + 1:])
n1 = lis_mod; n2 = mi_mod
print(round(dot(n1, n2)/(norm(n1)*norm(n2)), 10))

## task 2 - top-25 most similar nodes
# 2A: including self-importance
# lisinopril
sim_res_lis = similarity_search(dp_matrix, lis_index, 500)
sim_res_lis.sort(key=lambda x: x[1], reverse=True)
results_formatter(sim_res_lis[0:30], node_idx_dict, node_labels, node_types)
results_formatter(sim_res_lis[0:250], node_idx_dict, node_labels, node_types, ['drug'])
results_formatter(sim_res_lis[0:100], node_idx_dict, node_labels, node_types, ['indication'])
results_formatter(sim_res_lis[0:50], node_idx_dict, node_labels, node_types, ['biological_function'])
results_formatter(sim_res_lis[0:60], node_idx_dict, node_labels, node_types, ['protein'])
# myocardial infarction
sim_res_mi = similarity_search(dp_matrix, mi_index, 500)
sim_res_mi.sort(key=lambda x: x[1], reverse=True)
results_formatter(sim_res_mi[0:10], node_idx_dict, node_labels, node_types)
results_formatter(sim_res_mi[0:150], node_idx_dict, node_labels, node_types, ['drug'])
results_formatter(sim_res_mi[0:150], node_idx_dict, node_labels, node_types, ['indication'])
results_formatter(sim_res_mi[0:120], node_idx_dict, node_labels, node_types, ['biological_function'])
results_formatter(sim_res_mi[0:15], node_idx_dict, node_labels, node_types, ['protein'])

# 2B: excluding self-importance
dp_matrix_adj = remove_self_importance(dp_matrix, node_idx_dict)
# lisinopril
sim_res_lis_adj = similarity_search(dp_matrix_adj, lis_index, 500)
sim_res_lis_adj.sort(key=lambda x: x[1], reverse=True)
results_formatter(sim_res_lis_adj[0:11], node_idx_dict, node_labels, node_types)
results_formatter(sim_res_lis_adj[0:30], node_idx_dict, node_labels, node_types, ['drug'])
results_formatter(sim_res_lis_adj[0:50], node_idx_dict, node_labels, node_types, ['indication'])
results_formatter(sim_res_lis_adj[0:30], node_idx_dict, node_labels, node_types, ['biological_function'])
results_formatter(sim_res_lis_adj[0:60], node_idx_dict, node_labels, node_types, ['protein'])
# myocardial infarction
sim_res_mi_adj = similarity_search(dp_matrix_adj, mi_index, 5000)
sim_res_mi_adj.sort(key=lambda x: x[1], reverse=True)
results_formatter(sim_res_mi_adj[0:10], node_idx_dict, node_labels, node_types)
results_formatter(sim_res_mi_adj[0:2000], node_idx_dict, node_labels, node_types, ['drug'])
results_formatter(sim_res_mi_adj[0:15], node_idx_dict, node_labels, node_types, ['indication'])
results_formatter(sim_res_mi_adj[0:2000], node_idx_dict, node_labels, node_types, ['biological_function'])
results_formatter(sim_res_mi_adj[0:50], node_idx_dict, node_labels, node_types, ['protein'])


### Determining Relationship btw Coughing, Lisinopril, and Myocardial Infarction
# coughing (C0010200)
cu_concept_id = [k for k, v in node_labels.items() if v.lower() == "contact dermatitis"][0]  # C0010200
cu_index = [k for k, v in node_idx_dict.items() if v == cu_concept_id][0]

## importance overlap
lis_top_imp = [[i, dp_matrix[lis_index][i]] for i in np.argsort(dp_matrix[lis_index])[-1000 + 1:] if i != lis_index]
mi_top_imp = [[i, dp_matrix[mi_index][i]] for i in np.argsort(dp_matrix[mi_index])[-1000 + 1:] if i != mi_index]
cu_top_imp = [[i, dp_matrix[cu_index][i]] for i in np.argsort(dp_matrix[cu_index])[-1000 + 1:] if i != cu_index]

# find concepts that overlap across all 3 terms
# imp_common = set([x[0] for x in mi_top_imp]) & set([x[0] for x in lis_top_imp])
# imp_common = set([x[0] for x in mi_top_imp]) & set([x[0] for x in cu_top_imp])
imp_common = set([x[0] for x in lis_top_imp]) & set([x[0] for x in cu_top_imp])
# imp_common = set([x[0] for x in lis_top_imp]) & set([x[0] for x in mi_top_imp]) & set([x[0] for x in cu_top_imp])
imp_common_update = []
for i in imp_common:
    lis_score = [x[1] for x in lis_top_imp if i == x[0]][0]
    # mi_score = [x[1] for x in mi_top_imp if i == x[0]][0]
    cu_score = [x[1] for x in cu_top_imp if i == x[0]][0]
    # imp_common_update.append([i, max([mi_score, lis_score])])
    # imp_common_update.append([i, max([mi_score, cu_score])])
    imp_common_update.append([i, max([lis_score, cu_score])])
    # imp_common_update.append([i, max([lis_score, mi_score, cu_score])])
imp_common_update.sort(key=lambda x: x[1], reverse=True)
results_formatter(imp_common_update[0:10], node_idx_dict, node_labels, node_types)

## similarity with self-importance
lis_top_sim1 = similarity_search(dp_matrix, lis_index, 1000)
mi_top_sim1 = similarity_search(dp_matrix, mi_index, 1000)
cu_top_sim1 = similarity_search(dp_matrix, cu_index, 1000)
# find concepts that overlap across all 3 terms
sim1_common = set([x[0] for x in lis_top_sim1]) & set([x[0] for x in cu_top_sim1])
# sim1_common = set([x[0] for x in lis_top_sim1]) & set([x[0] for x in mi_top_sim1])
# sim1_common = set([x[0] for x in mi_top_sim1]) & set([x[0] for x in cu_top_sim1])
# sim1_common = set([x[0] for x in lis_top_sim1]) & set([x[0] for x in mi_top_sim1]) & set([x[0] for x in cu_top_sim1])
sim1_common_update = []
for i in sim1_common:
    lis_score = [x[1] for x in lis_top_sim1 if i == x[0]][0]
    # mi_score = [x[1] for x in mi_top_sim1 if i == x[0]][0]
    cu_score = [x[1] for x in cu_top_sim1 if i == x[0]][0]
    sim1_common_update.append([i, max([lis_score, cu_score])])
    # sim1_common_update.append([i, max([lis_score, mi_score])])
    # sim1_common_update.append([i, max([mi_score, cu_score])])
    # sim1_common_update.append([i, max([lis_score, mi_score, cu_score])])
sim1_common_update.sort(key=lambda x: x[1], reverse=True)
results_formatter(sim1_common_update[0:10], node_idx_dict, node_labels, node_types)


## similarity without self-importance
lis_top_sim2 = similarity_search(dp_matrix_adj, lis_index, 5000)
mi_top_sim2 = similarity_search(dp_matrix_adj, mi_index, 5000)
cu_top_sim2 = similarity_search(dp_matrix_adj, cu_index, 5000)
# find concepts that overlap across all 3 terms
# sim2_common = set([x[0] for x in lis_top_sim2]) & set([x[0] for x in mi_top_sim2])
sim2_common = set([x[0] for x in lis_top_sim2]) & set([x[0] for x in cu_top_sim2])
# sim2_common = set([x[0] for x in cu_top_sim2]) & set([x[0] for x in mi_top_sim2])
# sim2_common = set([x[0] for x in lis_top_sim2]) & set([x[0] for x in mi_top_sim2]) & set([x[0] for x in cu_top_sim2])
sim2_common_update = []
for i in sim2_common:
    lis_score = [x[1] for x in lis_top_sim2 if i == x[0]][0]
    # mi_score = [x[1] for x in mi_top_sim2 if i == x[0]][0]
    cu_score = [x[1] for x in cu_top_sim2 if i == x[0]][0]
    # sim2_common_update.append([i, max([lis_score, mi_score])])
    sim2_common_update.append([i, max([lis_score, cu_score])])
    # sim2_common_update.append([i, max([cu_score, mi_score])])
    # sim2_common_update.append([i, max([lis_score, mi_score, cu_score])])
sim2_common_update.sort(key=lambda x: x[1], reverse=True)
results_formatter(sim2_common_update[0:10], node_idx_dict, node_labels, node_types)


# importance
## ACE
ace_concept_id = [k for k, v in node_labels.items() if v == "ACE"][0]  # C0010200
ace_index = [k for k, v in node_idx_dict.items() if v == ace_concept_id][0]
res = [[i, dp_matrix[ace_index][i]] for i in np.argsort(dp_matrix[ace_index])[-50 + 1:] if i != ace_index]
results_formatter(res, node_idx_dict, node_labels, node_types)
ace_top_sim = similarity_search(dp_matrix_adj, ace_index, 50)
results_formatter(ace_top_sim, node_idx_dict, node_labels, node_types)



strng = "leth"
res = ['{} - {}'.format(k, v) for k, v in node_labels.items() if strng in v.lower() and k.startswith('C')]
print('\n'.join(res))