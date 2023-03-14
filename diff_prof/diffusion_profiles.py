#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

# import needed libraries
import copy
import os
import pickle
import math
import multiprocessing
import networkx as nx
import numpy as np
import scipy
import time

from datetime import datetime
from tqdm import tqdm
from typing import Generic, Optional, Tuple

WEIGHT: str = "weight"


class DiffusionProfiles(object):
    def __init__(self, alpha=None, max_iter=None, tol=None, weights=None, num_cores=None):
        self.alpha: Optional[float] = alpha
        self.max_iter: Optional[int] = max_iter
        self.tol: Optional[float] = tol
        self.weights: Optional[dict] = weights
        self.num_cores: Optional[int] = num_cores
        self.initial_m: Optional[scipy.sparse.csr.csr_matrix] = None
        self.diffusion_profile: Optional[dict] = None
        self.diffusion_profile_matrix: Optional[np.array] = None
        self.matrix_index_dict: Optional[dict] = None
        self.save_load_file_path: str = "results/"
        self.diffusion_profile_file_name: str = "_diffusion_profile.npy"

    def get_initial_m(self, msi):
        """This function is adapted from the NetworkX implementation of Personalized PageRank. Function takes a msi
            object and uses it to convert the msi graph into a numpy sparse matrix object.

            Args:
                msi: A MSI object containing a NetworkX graph and associated metadata.

            Returns:
                None.

            Raises:
                ValueError: if the process to convert the msi graph to a spase matrix fails.
            """

        m = nx.to_scipy_sparse_matrix(msi.graph, nodelist=msi.nodelist, weight=WEIGHT, dtype=float)
        if len(msi.graph) == 0:
            raise ValueError("Process to convert MSI graph to matrix failed, contains no nodes!")
        else:
            self.initial_m = m

    def convert_m_to_make_all_drugs_indications_sinks_except_selected(self, msi: Generic, entities: list) -> scipy.sparse.csr.csr_matrix:
        """If a node has no outgoing links to other pages, it is a sink. Sinks can be problematic for PageRank (i.e.,
        the walker can get stuck). To deal with this, the rank of sink nodes can be distributed among all nodes in the
        graph by adding outgoing edges from all sink nodes to all other nodes.

        This function is designed to convert all drugs and indications to sink nodes.

        Args:
            msi: A MSI object containing a NetworkX graph and associated metadata.
            entities: A nested list of drugs and indications.

        Returns:
            reconstructed_m: A sparse numpy matrix serving as the biased transition matrix.
        """

        reconstructed_m = copy.deepcopy(self.initial_m)

        # delete edges INTO selected drugs and indications
        for node in entities:
            if node in msi.drugs_in_graph + msi.indications_in_graph:
                proteins_to_drug_or_indication = msi.drug_or_indication2proteins[node]
                ids_to_remove_in_edges = [msi.node2idx[protein] for protein in proteins_to_drug_or_indication]
                reconstructed_m[ids_to_remove_in_edges, msi.node2idx[node]] = 0

        # delete edges OUT of unselected drugs and indications
        ids_to_remove_out_edges = []
        for node in msi.drugs_in_graph + msi.indications_in_graph:
            if not (node in entities):
                proteins_drug_or_indication_points_to = msi.drug_or_indication2proteins[node]
                for protein in proteins_drug_or_indication_points_to:
                    ids_to_remove_out_edges.append((msi.node2idx[node], msi.node2idx[protein]))
        i, j = zip(*ids_to_remove_out_edges)
        reconstructed_m[i, j] = 0.0
        return reconstructed_m

    @staticmethod
    def refine_m_s(m: scipy.sparse.csr.csr_matrix) -> Tuple:
        """This function is adapted from the NetworkX implementation of Personalized PageRank. The purpose of this
        function is to encode the graph and set of weights into a biased transition matrix, where each entry denotes
        the probability a random walker jumps from node_i to node_j when continuing its walk.

        Args:
            m: A sparse numpy matrix-based representation of the graph.

        Return:
            A list of two objects:
                m: A sparse numpy matrix-based representation of the graph.
                s: A restart vector which sets the probability the walker will jump to each node after a restart.
        """

        s = scipy.array(m.sum(axis=1)).flatten()
        s[s != 0] = 1.0 / s[s != 0]
        q = scipy.sparse.spdiags(s.T, 0, *m.shape, format="csr")
        m = q * m

        return m, s

    @staticmethod
    def get_personalization_dictionary(nodes_to_start_from: list, nodelist: list) -> dict:
        """Function takes a list of nodes to search and a list of a nodes in a batch (or the graph) and returns a
        dictionary which contains all nodes in the batch initialized with weight 1/ # nodes.

        Args:
            nodes_to_start_from: A list of node identifiers to search.
            nodelist: A list of all nodes in the batch or graph.

        Return:
             personalization_dict: A dictionary where keys are search nodes and values are floats representing the
                initialization value.
        """

        personalization_dict = dict.fromkeys(nodelist, 0)
        N = len(nodes_to_start_from)
        for node in nodes_to_start_from:
            personalization_dict[node] = 1. / N

        return personalization_dict

    def power_iteration(self, m: scipy.sparse.csr.csr_matrix, s: scipy.sparse.csr.csr_matrix, nodelist: list, per_dict: dict) -> np.array:
        """This function is adapted from the NetworkX implementation of Personalized PageRank.
        During power iteration, the walker can continue iterations until reaching convergence. At each step, the walker
        can restart its walk at the drug or disease node according to (1−α)s or continue its walk.
            - If the walker continues its walk from a node with out-edges, then it jumps to an adjacent node according
            to α(r(k)M).
            - If the walker continues its walk from a “sink” node, then it restarts its walk according to 𝛼(𝐬∑𝑗∈𝐽𝐫(𝑘)𝑗),
            where J is the set of “sink” nodes in the graph.

        Tolerance of ϵ = 1x 10−6; each node is started with a rank of 1/#vertices; convergence compares
        each walk’s rank against the prior ranks stops when the difference is less than a threshold.

        See the following slides for more information:
        https://cs.brown.edu/courses/cs016/static/files/assignments/projects/GraphHelpSession.pdf

        Args:
            m: A sparse numpy matrix-based representation of the graph.
            s: A restart vector which sets the probability the walker will jump to each node after a restart.
            nodelist: A list of nodes in the graph.
            per_dict: A dictionary where keys are search nodes and values are floats representing the
                initialization value.

        Returns:
            x: A numpy array representing a diffusion profile.

        Raises:
            NetworkXError: when a node is missing from the personalization dictionary or when power iteration failed to
                reach convergence within a specified number of max iterations.
        """

        n = len(nodelist)

        # personalization vector
        missing = set(nodelist) - set(per_dict)
        if missing:
            raise NetworkXError(
                "Personalization dictionary must have a value for every node. Missing nodes %s" % missing)
        p = scipy.array([per_dict[n] for n in nodelist], dtype=float)
        p = p / p.sum()

        # dangling nodes
        dangling_weights = p
        is_dangling = scipy.where(s == 0)[0]

        # power iteration: make up to max_iter iterations
        x = scipy.repeat(1.0 / n, n)  # Initialize; alternatively x = p
        for _ in range(self.max_iter):
            x_last = x
            x = self.alpha * (x * m + sum(x[is_dangling]) * dangling_weights) + (1 - self.alpha) * p
            # check convergence, l1 norm
            err = scipy.absolute(x - x_last).sum()
            if err < n * self.tol:
                return x

        raise NetworkXError("pagerank_scipy: power iteration failed to converge in %d iterations." % self.max_iter)

    @staticmethod
    def clean_file_name(file_name: str) -> str:
        """Cleans an input string meant to store a file name retaining only letters or numbers.

        Args:
            file_name: A string specifying a file name.

        Returns:
             A string containing the cleaned file name.
        """

        return "".join([c for c in file_name if c.isalpha() or c.isdigit() or c == " " or c == "_"]).rstrip()

    def save_diffusion_profile(self, diff_profile: np.array, node: Optional[str], f_name: Optional[str] = None) -> None:
        """Function saves a diffusion profile as a numpy array.

        Args:
            diff_profile: A numpy array containing the diffusion profile.
            node: A string containing the name of the entity.
            f_name: A string containing a file path (and file name) to use to save output to.

        Returns:
            None.
        """

        if f_name is None: file_name = self.clean_file_name(node) + self.diffusion_profile_file_name
        else: file_name = f_name
        np.save(os.path.join(self.save_load_file_path, file_name), diff_profile)

        return None

    def process_saved_diffusion_profiles(self) -> None:
        """Function reads in  saved numpy arrays for all nodes in the msi graph and appends them to a numpy matrix. The
        resulting matrix is indexed by the nodelist ordering. The node list and resulting numpy matrix are saved as
        separate files and pickled.

        Returns:
             None.
        """

        diffusion_profile_matrix = []
        for node in tqdm(msi.nodelist):
            # load node's numpy array
            f = os.path.join(self.save_load_file_path, self.clean_file_name(node) + self.diffusion_profile_file_name)
            node_diffusion_profile = np.load(f)
            # append node diffusion profile to numpy matrix
            diffusion_profile_matrix.append(node_diffusion_profile)
            os.remove(f)  # delete single diffusion profile files

        # create dictionary to store node list and diffusion profile matrix
        diffusion_profile_matrix = np.asarray(diffusion_profile_matrix)
        file_name1 = os.path.join(self.save_load_file_path, 'msi_diffusion_profile_matrix.npy')
        np.save(file_name1, diffusion_profile_matrix)
        # saving node index information
        file_name2 = os.path.join(self.save_load_file_path, 'msi_diffusion_profile_matrix_index_ids.npy')
        node_idx_dict = {x[0]: x[1] for x in enumerate(msi.nodelist)}
        pickle.dump(node_idx_dict, open(file_name2, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        print("Saving diffusion profile matrix ({}) and node index dictionary ({})".format(fiel_name1, file_name2))

        return None

    def calculate_diffusion_profile(self, msi: Generic, nodes: list) -> None:
        """Function takes a msi object and a list of entities in a batch, creates diffusion profiles, and writes them
        to disk.

        Args:
            msi: A MSI object containing a NetworkX graph and associated metadata.
            nodes: A list containing a node identifier.

        Returns:
            None.
        """

        assert (len(nodes) == 1)  # not enabling functionality to run from multiple
        selected_node = nodes[0]
        M = self.convert_m_to_make_all_drugs_indications_sinks_except_selected(msi, nodes)
        M, S = self.refine_m_s(M)
        per_dict = self.get_personalization_dictionary(nodes, msi.nodelist)
        diffusion_profile = self.power_iteration(M, S, msi.nodelist, per_dict)
        self.save_diffusion_profile(diffusion_profile, selected_node)

        return None

    # def calculate_diffusion_profile(self, msi: Generic, selected_drugs_and_indications: list) -> None:
    #     """Function takes a msi object and a list of entities in a batch, creates diffusion profiles, and writes them
    #     to disk.
    #
    #     Args:
    #         msi: A MSI object containing a NetworkX graph and associated metadata.
    #         selected_drugs_and_indications: A list of drug and indication identifiers.
    #
    #     Returns:
    #         None.
    #     """
    #
    #     assert (len(selected_drugs_and_indications) == 1)  # Not enabling functionality to run from multiple
    #     selected_drug_or_indication = selected_drugs_and_indications[0]
    #     M = self.convert_m_to_make_all_drugs_indications_sinks_except_selected(msi, selected_drugs_and_indications)
    #     M, S = self.refine_m_s(M)
    #     per_dict = self.get_personalization_dictionary(selected_drugs_and_indications, msi.nodelist)
    #     diffusion_profile = self.power_iteration(M, S, msi.nodelist, per_dict)
    #     self.save_diffusion_profile(diffusion_profile, selected_drug_or_indication)
    #
    #     return None

    def calculate_diffusion_profile_batch(self, msi: Generic, batch: list) -> None:
        """Function iterates over each item in each batch and calls the calculate_diffusion_profile function on each.

        Args:
            msi: A MSI object containing a NetworkX graph and associated metadata.
            batch: A list of drug and indication identifiers in a batch.

        Returns:
             None.
        """
        for selected_entities in tqdm(batch):
            self.calculate_diffusion_profile(msi, selected_entities)

        return None

    @staticmethod
    def batch_list(list_: list, batch_size: int = None, num_cores: int = None) -> list:
        """Function takes a list of drugs and indications and option arguments for batch size and the number of
        available cores and uses this information to create a batch list.

        Args:
            list_: A nested list of drug and indication identifiers.
            batch_size: An optional argument that can be an integer specifying the number of batches or None.
            num_cores: An integer specifying the number of available cores.

        Returns:
            batched_list: A list nested list where each inner list represents a batch.
        """

        if batch_size == float("inf"):
            batched_list = [list_]
        else:
            if batch_size is None:
                batch_size = math.ceil(len(list_) / (float(num_cores)))
            batched_list = []
            for i in range(0, len(list_), batch_size):
                batched_list.append(list_[i:i + batch_size])

        return batched_list

    def calculate_diffusion_profiles(self, msi: Generic) -> None:
        """Main function of class orchestrating the batch processing of nodes in the msi graph object and creating
        diffusion profiles for each entity in the batch.

        Args:
            msi: A MSI object containing a NetworkX graph and associated metadata.

        Returns:
             None.
        """

        print("\n\n" + "*" * 100 + "\nCalculating Diffusion Profiles\n" + "*" * 100)
        start_time = datetime.now()

        # STEP 1: weight graph
        print("---> Building and Weighting Adjacency Matrix")
        msi.weight_graph(self.weights)

        # STEP 2: prepare to run power iteration in parallel
        print("---> Preparing to Run Power Iteration")
        self.get_initial_m(msi)
        # computation_list = [[i] for i in msi.drugs_in_graph + msi.indications_in_graph]
        computation_list = [[i] for i in msi.nodelist]

        # STEP 3: run power iteration in parallel
        print("---> Running Power Iteration in Parallel")
        computation_list_batches = self.batch_list(computation_list, num_cores=self.num_cores)
        proc, procs = None, []
        for batch in computation_list_batches:
            # break
            while len([job for job in procs if job.is_alive()]) == self.num_cores:
                time.sleep(1)
            proc = multiprocessing.Process(target=self.calculate_diffusion_profile_batch, args=(msi, batch))
            procs.append(proc)
            proc.start()

        # STEP 4: wait until all processes done and then stop all jobs
        print("---> Running Power Iteration in Parallel")
        while len([job for job in procs if job.is_alive()]) > 0:
            time.sleep(1)

        # Stop all the jobs and close them
        for _ in procs:
            proc.join()
            proc.terminate()

        # STEP 5: process diffusion profiles to create a single matrix containing all diffusion profiles
        print("---> Converting Node's Diffusion Profiles into a Single Diffusion Profile Matrix")
        self.process_saved_diffusion_profiles()

        # complete process and output runtime
        end_time = datetime.now()
        print("\n" + "*" * 10 + " Diffusion Profile Runtime: {}".format(end_time - start_time) + "*" * 10)

        return None

    def load_diffusion_profiles(self) -> None:
        """ Function loads diffusion profiles dictionary object.

        Returns:
            diffusion_profile_matrix: A numpy array where rows are node identifiers and columns are diffusion
                profile values.
            node_idx_dict: A dictionary keyed by matrix index with node identifiers as values.

        Raises:
            FileNotFoundError: if diffusion profile dictionary cannot be found.
        """

        file_name1 = os.path.join(self.save_load_file_path, 'msi_diffusion_profile_matrix.npy')
        file_name2 = os.path.join(self.save_load_file_path, 'msi_diffusion_profile_matrix_index_ids.npy')
        if not os.path.exists(file_name1) and not os.path.exists(file_name2):
            raise FileNotFoundError("Cannot find file: {}".format(file_name))
        else:
            print("\n" + "*" * 100 + "\nLoading Diffusion Profiles\n" + "*" * 100)
            # loading diffusion profile data
            self.diffusion_profile_matrix = np.load(file_name1)
            # loading node index information
            self.matrix_index_dict = pickle.load(open(file_name2, "rb"))

            return None

    def diffusion_profile_lookup(self, node: str) -> np.array:
        """Function takes a string presenting a node identifier and returns its diffusion profile.

        Args:
            node: A string representing a node identifier.

        Returns:
            node_diffusion_profile: A numpy array containing a single node's diffusion profile.

        Raises:
            KeyError: If a node is not found within the diffusion profile matrix dictionary.
        """

        if node not in self.matrix_index_dict.values():
            raise KeyError("There is no diffusion profile for: {}")
        else:
            node_idx = [k for k, v in self.matrix_index_dict.items() if v == node]
            node_diffusion_profile = self.diffusion_profile_matrix[node_idx]

            return node_diffusion_profile
