#!/usr/bin/env python
# -*- coding: utf-8 -*-


# import and load needed scripts
import numpy as np
import pandas as pd

from numpy import dot
from numpy.linalg import norm
from scipy.stats import gmean
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from tqdm import tqdm
from typing import Optional


def gets_simple_statistics(results_list: list, data_type: str, value_type: str) -> str:
    """Function takes a list of int or floats and calculates simple statistics including: min, max, mean, and median.

    Args:
        results_list: A list of ints or floats.
        data_type: A string indicating what the data structure is (e.g. "graph", "diffusion profile").
        value_type: A string indicating what the values were (e.g., "degree", "diffusion profile importance").

    Returns:
        formatted_string: A formatted string containing the statistics.
    """

    # calculate stats
    min_value = min(results_list)
    max_value = max(results_list)
    median_value = np.median(results_list)
    mean_value = np.mean(results_list)
    # format output
    print_string = "Statistics ({}, {}):\n  - min={}; max={}; median={}; mean={}"
    formatted_string = print_string.format(data_type, value_type, min_value, max_value, median_value, mean_value)

    print(formatted_string)


def similarity_search(matrix: np.array, index_node: int, top_n: int = 10) -> list:
    """ Function takes as input a numpy array (matrix), an integer representing a node index, and an integer
    representing the number of similar nodes to return. The function uses this information and calculates the cosine
    similarity between the index node and all other included nodes. The results are sorted and returned as a list of
    lists where each list contains a node index and the cosine similarity score of the top set of similar nodes.

    http://markhneedham.com/blog/2016/07/27/scitkit-learn-tfidf-and-cosine-similarity-for-computer-science-papers/

    Args:
        matrix: A numpy array where rows are nodes and columns are embedding values.
        index_node: an integer representing a node index.
        top_n: an integer representing the number of similar nodes to return (default=10).

    Returns:
        similar_nodes: a list of lists where each list contains a node index and the cosine similarity scores of the
            top set of similar nodes.
    """

    # calculate similarity
    # cosine_similarities = linear_kernel(matrix[index_node:index_node + 1], matrix).flatten()
    cosine_similarities = cosine_similarity(matrix[index_node:index_node + 1], matrix).flatten()
    rel_node_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index_node]
    similar_nodes = [(node, cosine_similarities[node]) for node in rel_node_indices][0:top_n]

    return similar_nodes


def results_formatter(results: list, node_idx: dict, labels: dict, types: dict, include: Optional[list] = None) -> None:
    """Function takes the results of a similarity search and several dictionaries containing node metadata. Using the
    metadata dictionaries, the results of the similarity search are formatted and printed.

    Args:
        results: a nested list containing node indices and the cosine similarity scores of similar nodes.
        node_idx: A dictionary where keys are matrix indices and values are node identifiers.
        labels: A dictionary where keys are node identifiers and values are node labels.
        types: A dictionary where keys are node identifiers and values are node types.
        include: A list of node types that should be included. Possible types include: 'drug', 'protein',
                 'biological_function', and 'indication'.

    Returns:
        None.
    """

    results.sort(key=lambda x: x[1], reverse=True)
    formatted_results = []
    for node, score in results:
        node_id = node_idx[node]; node_label, node_type = labels[node_id], types[node_id]
        score = round(score, 8)  # to make things that are more similar have a larger score
        if include is not None and node_type.lower() in include:
            r = "{} (id: {}, type: {}); Score: {}".format(node_label, node_id, node_type, score)
            formatted_results += [r]
        if include is None:
            r = "{} (id: {}, type: {}); Score: {}".format(node_label, node_id, node_type, score)
            formatted_results += [r]

    # print results
    if len(formatted_results) == 0: print("No results meet criteria, check exclude list")
    else: print('\n'.join(formatted_results))

    return None


def remove_self_importance(matrix: np.array, node_idx: dict) -> np.array:
    """Function creates a new version of the diffusion profile matrix, where the self-importance score of each node
    removed.

    Args:
        matrix: A Numpy array storing a matrix, where rows contain diffusion profiles for each node and column values
            contain importance scores between that node and all other nodes in the original graph.
        node_idx: A dictionary where keys are matrix indices and values are node identifiers.

    Returns:
         matrix_adj: A Numpy array that has been updated such that each row or node (i.e., diffusion profile) does
            not contain an importance score to itself.
    """

    adj_diff_profiles = []
    for node in tqdm(node_idx.keys()):
        # find node's array
        node_matrix = matrix[node]
        # remove self-importance score
        node_mod = list(node_matrix)[0:node] + list(node_matrix)[node + 1:]
        # append node diffusion profile to numpy matrix
        adj_diff_profiles.append(node_mod)

    # convert adjusted list of list matrix as numpy array
    matrix_adj = np.asarray(adj_diff_profiles)
    del adj_diff_profiles

    return matrix_adj


def get_node_pair_importance(matrix: np.array, node1_index: int, node2_index: int, n1_name: str, n2_name: str) -> list:
    """The function calculates the importance of two nodes by extracting their values from each other's diffusion
    profiles.

    Args:
        matrix: A Numpy array storing a matrix, where rows contain diffusion profiles for each node and column values
            contain importance scores between that node and all other nodes in the original graph.
        node1_index: An integer representing a node's location in the diffusion profile matrix.
        node2_index: An integer representing a node's location in the diffusion profile matrix.
        n1_name: A string containing the name of Node 1.
        n2_name: A string containing the name of Node 2.

    Returns:
        imp_list: A list of three values: (1) the importance of node2 within node1's diffusion profile; (2) the
            importance of node1 within node2's diffusion profile; and (3) the mean importance score derived from the
            prior two measures.
    """

    # extract each node's diffusion profile from matrix
    dp_imp_node1 = matrix[node1_index]
    dp_imp_node2 = matrix[node2_index]

    # get individual profile information



    # format strings
    str1 = "The importance of {} within {}'s Diffusion Profile: {}"
    str2 = "  - The minimum importance of both node's Diffusion Profiles: {}"
    str3 = "  - The maximum importance of both node's Diffusion Profiles: {}"
    str4 = "  - The average importance of both node's Diffusion Profiles: {}"

    # get importance between two nodes
    imp_list = [str1.format(n2_name, n1_name, dp_imp_node1[node2_index]),
                str1.format(n1_name, n2_name, dp_imp_node2[node1_index]),
                str2.format(min([dp_imp_node1[node2_index], dp_imp_node2[node1_index]])),
                str3.format(max([dp_imp_node1[node2_index], dp_imp_node2[node1_index]])),
                str4.format(np.mean([dp_imp_node1[node2_index], dp_imp_node2[node1_index]]))]

    return imp_list


def get_overlapping_concepts(n1_list: list, n2_list: list, node_index: dict, node_types: dict, node_labels: dict,
                             n1_label: str, n2_label: str) -> pd.DataFrame:
    """Function takes two lists of tuples, each containing the cosine similarity scores between a node of interest and
    some arbitrary set of nodes. It condenses the list into a Pandas DataFrame containing the original cosine
    similarity scores for each node as well as the min, max, mean, median, and geometric mean of the scores for each
    entity found in common in the two nodes cosine similarity results lists.

    Args:
        n1_list: A list of tuples containing the entity index and cosine similarity score for all nodes in the matrix to
            the node of interest.
        n2_list: A list of tuples containing the entity index and cosine similarity score for all nodes in the matrix to
            the node of interest.
        node_index: A dictionary where keys are matrix indices and values are node identifiers.
        node_types: A dictionary where keys are node identifiers and values are node types.
        node_labels: A dictionary where keys are node identifiers and values are node labels.
        n1_label: A string containing the label for the first node.
        n2_label: A string containing the label for the second node.

    Returns:
         df_sim_overlap: A Pandas DataFrame containing the results of the overlapping concept sets with original scores
            and aggregated metrics.
    """

    sim_common = set([x[0] for x in n1_list]) & set([x[0] for x in n2_list])
    results = []
    for i in sim_common:
        var_name, var_id, var_type = node_labels[node_index[i]], node_index[i], node_types[node_index[i]]
        n1_score = [x[1] for x in n1_list if i == x[0]][0]
        n2_score = [x[1] for x in n2_list if i == x[0]][0]
        # score metrics
        min_sim = min([n1_score, n2_score])
        max_sim = max([n1_score, n2_score])
        mean_sim = np.mean([n1_score, n2_score])
        median_sim = np.median([n1_score, n2_score])
        geomean_sim = gmean([n1_score, n2_score])
        results.append([var_name, var_id, var_type, n1_score, n2_score, min_sim, max_sim, mean_sim, median_sim, geomean_sim])
    # store results as a pandas DF
    df_sim_overlap = pd.DataFrame(data={
        'concept_name': [x[0] for x in results],
        'concept_id': [x[1] for x in results],
        'concept_type': [x[2] for x in results],
        '{}_org_score'.format(n1_label): [x[3] for x in results],
        '{}_org_score'.format(n2_label): [x[4] for x in results],
        'min_score': [x[5] for x in results],
        'max_score': [x[6] for x in results],
        'mean_score': [x[7] for x in results],
        'median_score': [x[8] for x in results],
        'geometric_mean_score': [x[9] for x in results]
    })

    return df_sim_overlap
