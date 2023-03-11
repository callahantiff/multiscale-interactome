#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

# import needed libraries
import os
# import pickle
import multiprocessing
import networkx as nx
import math
import time
import copy
import scipy
import numpy as np

from datetime import datetime
from tqdm import tqdm
from typing import Generic, Optional, Tuple

WEIGHT: str = "weight"


class DiffusionProfiles(object):
    def __init__(self, alpha, max_iter, tol, weights, num_cores, save_load_file_path):
        self.alpha: float = alpha
        self.max_iter: int = max_iter
        self.tol: float = tol
        self.weights: dict = weights
        self.num_cores: int = num_cores
        self.save_load_file_path: str = save_load_file_path
        self.initial_m: Optional[scipy.sparse.csr.csr_matrix] = None
        self.drug_or_indication2diffusion_profile: dict = None

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
        for drug_or_indication in entities:
            proteins_pointing_to_selected_drug_or_indication = msi.drug_or_indication2proteins[drug_or_indication]
            ids_to_remove_in_edges = [msi.node2idx[protein] for protein in proteins_pointing_to_selected_drug_or_indication]
            reconstructed_m[ids_to_remove_in_edges, msi.node2idx[drug_or_indication]] = 0

        # delete edges OUT of unselected drugs and indications
        ids_to_remove_out_edges = []
        for drug_or_indication in msi.drugs_in_graph + msi.indications_in_graph:
            if not (drug_or_indication in entities):
                proteins_drug_or_indication_points_to = msi.drug_or_indication2proteins[drug_or_indication]
                for protein in proteins_drug_or_indication_points_to:
                    ids_to_remove_out_edges.append((msi.node2idx[drug_or_indication], msi.node2idx[protein]))
        i, j = zip(*ids_to_remove_out_edges)
        reconstructed_m[i, j] = 0.0
        return reconstructed_m

    @staticmethod
    def refine_m_s(m: scipy.sparse.csr.csr_matrix) -> Tuple:
        """This function is adapted from the NetworkX implementation of Personalized PageRank. The purpose of this
        function is to encode the graph and set of weights into a biased transition matrix, where each entry denotes the probability a random walker jumps from node_i to node_j when continuing its walk.

        Args:
            m: A sparse numpy matrix-based representation of the graph.

        Return:
            A list of two objects:
                m: A sparse numpy matrix-based representation of the graph.
                s: A restart vector which sets the probability the walker will jump to each node after a restart.
        """

        s = scipy.array(m.sum(axis=1)).flatten()
        s[s != 0] = 1.0 / s[s != 0]
        q = scipy.sparse.spdiags(s.T, 0, *m.shape, format='csr')
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
            - If the walker continues its walk from a node with out-edges, then it jumps to an adjacent node according to
            α(r(k)M).
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
            NetworkXError: when a node is missing from the personalization dictionary or when power iteration failed to reach convergence within a specified number of max iterations.
        """

        n = len(nodelist)

        # personalization vector
        missing = set(nodelist) - set(per_dict)
        if missing:
            raise NetworkXError(
                'Personalization dictionary must have a value for every node. Missing nodes %s' % missing)
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

        raise NetworkXError('pagerank_scipy: power iteration failed to converge in %d iterations.' % self.max_iter)

    @staticmethod
    def clean_file_name(file_name: str) -> str:
        """Cleans an input string meant to store a file name retaining only letters or numbers.

        Args:
            file_name: A string specifying a file name.

        Returns:
             A string containing the cleaned file name.
        """

        return "".join([c for c in file_name if c.isalpha() or c.isdigit() or c == ' ' or c == "_"]).rstrip()

    def save_diffusion_profile(self, diffusion_profile, selected_drug_or_indication) -> None:
        f = os.path.join(self.save_load_file_path,
                         self.clean_file_name(selected_drug_or_indication) + "_p_visit_array.npy")
        np.save(f, diffusion_profile)

        return None

    def calculate_diffusion_profile_batch(self, msi: Generic, batch: list) -> None:
        """Function iterates over each item in each batch and calls the calculate_diffusion_profile function on each.

        Args:
            msi: A MSI object containing a NetworkX graph and associated metadata.
            batch: A list of drug and indication identifiers in a batch.

        Returns:
             None.
        """
        for selected_drugs_and_indications in tqdm(batch):
            self.calculate_diffusion_profile(msi, selected_drugs_and_indications)

        return None

    def calculate_diffusion_profile(self, msi: Generic, selected_drugs_and_indications: list) -> None:
        """Function takes a msi object and a list of entities in a batch, creates diffusion profiles, and writes them
        to disk.

        Args:
            msi: A MSI object containing a NetworkX graph and associated metadata.
            selected_drugs_and_indications: A list of drug and indication identifiers.

        Returns:
            None.
        """

        assert (len(selected_drugs_and_indications) == 1)  # Not enabling functionality to run from multiple
        selected_drug_or_indication = selected_drugs_and_indications[0]
        M = self.convert_m_to_make_all_drugs_indications_sinks_except_selected(msi, selected_drugs_and_indications)
        M, S = self.refine_m_s(M)
        per_dict = self.get_personalization_dictionary(selected_drugs_and_indications, msi.nodelist)
        diffusion_profile = self.power_iteration(M, S, msi.nodelist, per_dict)
        self.save_diffusion_profile(diffusion_profile, selected_drug_or_indication)

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

        if batch_size == float('inf'):
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

        print("*" * 100 + "\nCalculating Diffusion Profiles\n" + "*" * 100)
        start_time = datetime.now()


        # STEP 1: save msi graph and node2idx
        print("---> Saving Multi-Interactome and Node Index Dictionary")
        msi.save_graph(self.save_load_file_path)
        msi.save_node2idx(self.save_load_file_path)

        #  STEP 2: weight graph
        print("---> Building and Weighting Adjacency Matrix")
        msi.weight_graph(self.weights)

        # STEP 3: prepare to run power iteration in parallel
        print("---> Preparing to Run Power Iteration")
        self.get_initial_m(msi)
        computation_list = [[i] for i in msi.drugs_in_graph + msi.indications_in_graph]

        # STEP 4: run power iteration in parallel
        print("---> Running Power Iteration in Parallel")
        computation_list_batches = self.batch_list(computation_list, num_cores=self.num_cores)
        proc, procs = None, []
        for batch in computation_list_batches:
            while len([job for job in procs if job.is_alive()]) == self.num_cores:
                time.sleep(1)
            proc = multiprocessing.Process(target=self.calculate_diffusion_profile_batch, args=(msi, batch))
            procs.append(proc)
            proc.start()

        # STEP 5: wait until all processes done and then stop all jobs
        print("---> Running Power Iteration in Parallel")
        while len([job for job in procs if job.is_alive()]) > 0:
            time.sleep(1)

        # Stop all the jobs and close them
        for _ in procs:
            proc.join()
            proc.terminate()

        end_time = datetime.now()
        print("Program took {} time to run".format(end_time - start_time))

        return None

    def load_diffusion_profiles(self, drugs_and_indications: list) -> None:
        """Function takes a list of drug and indication identifiers and for each identifier loads the corresponding
        diffusion profile. Function loads diffusion profiles into a dictionary where the keys are node identifiers and
        the values are diffusion profiles.

        Args:
            drugs_and_indications: A list of drug and indication identifiers.

        Returns:
            None.

        Raises:
            TypeError: if something other than a string is passed for the file path to load diffusion profiles.
            FileNotFoundError: if a diffusion profile for a drug or indication cannot be found.
        """

        if not isinstance(self.save_load_file_path, str):
            raise TypeError("Please provide a valid file path for diffusion profiles.")
        else:
            print("*" * 100 + "\nLoading Diffusion Profiles\n" + "*" * 100)
            self.drug_or_indication2diffusion_profile = dict()
            for x in drugs_and_indications:
                f = os.path.join(self.save_load_file_path, self.clean_file_name(x) + "_p_visit_array.npy")
                if os.path.exists(f):
                    self.drug_or_indication2diffusion_profile[x] = np.load(f)
                else:
                    raise FileNotFoundError("Loading failed at " + str(x) + " | " + str(f))

            return None
