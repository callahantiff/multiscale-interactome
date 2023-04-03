#!/usr/bin/env python
# -*- coding: utf-8 -*-


# import and load needed scripts
import numpy as np

from numpy import dot
from numpy.linalg import norm
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from tqdm import tqdm
from typing import Optional


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


def results_formatter(results: list, node_idx: dict, labels: dict, types: dict, exclude: Optional[list] = None) -> None:
    """Function takes the results of a similarity search and several dictionaries containing node metadata. Using the
    metadata dictionaries, the results of the similarity search are formatted and printed.

    Args:
        results: a nested list containing node indices and the cosine similarity scores of similar nodes.
        node_idx: A dictionary where keys are matrix indices and values are node identifiers.
        labels: A dictionary where keys are node identifiers and values are node labels.
        types: A dictionary where keys are node identifiers and values are node types.
        exclude: A list of node types that should be ignored. Possible types include: 'drug', 'protein',
                 'biological_function', and 'indication'.

    Returns:
        None.
    """

    results.sort(key=lambda x: x[1], reverse=True)
    formatted_results = []
    for node, score in results:
        node_id = node_idx[node]; node_label, node_type = labels[node_id], types[node_id]
        score = round(score, 8)  # to make things that are more similar have a larger score
        if exclude is not None and node_type.lower() not in exclude:
            r = "{} (id: {}, type: {}); Score: {}".format(node_label, node_id, node_type, score)
            formatted_results += [r]
        if exclude is None:
            r = "{} (id: {}, type: {}); Score: {}".format(node_label, node_id, node_type, score)
            formatted_results += [r]

    # print results
    if len(formatted_results) == 0: print("No results meet criteria, check exclude list")
    else: print('\n'.join(formatted_results))

    return None


def remove_self_importance(matrix: np.array) -> np.array:
    """Function creates a new version of the diffusion profile matrix, where the self-importance score of each node
    removed.

    Args:
        matrix: A Numpy array storing a matrix, where rows contain diffusion profiles for each node and column values
            contain importance scores between that node and all other nodes in the original graph.

    Returns:
         matrix_adj: A Numpy array that has been updated such that each row or node (i.e., diffusion profile) does
            not contain an importance score to itself.
    """

    adj_diff_profiles = []
    for node in tqdm(node_idx_dict.keys()):
        # find node's array
        node_matrix = matrix[node]
        #  remove self-importance score
        node_mod = list(node_matrix)[0:node] + list(node_matrix)[node + 1:]
        # append node diffusion profile to numpy matrix
        adj_diff_profiles.append(node_mod)

    # convert adjusted list of list matrix as numpy array
    matrix_adj = np.asarray(adj_diff_profiles)
    del adj_diff_profiles

    return matrix_adj
