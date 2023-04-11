#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

# import needed libraries
from msi.drug_to_protein import DrugToProtein
from msi.indication_to_protein import IndicationToProtein
from msi.protein_to_protein import ProteinToProtein
from msi.protein_to_biological_function import ProteinToBiologicalFunction
from msi.biological_function_to_biological_function import BiologicalFunctionToBiologicalFunction
from typing import Optional

import networkx as nx
import pandas as pd
import os
import pickle


class MSI:
    def __init__(self, drug2protein_file=None, indication2protein_file=None, protein2protein_file=None,
                 protein2biological_function_file=None, biological_function2biological_function_file=None,
                 drug2protein_directed=False, indication2protein_directed=False, protein2protein_directed=False,
                 protein2biological_function_directed=False, biological_function2biological_function_directed=True,
                 save_load_file_path="results/"):
        self.graph: nx.Graph = nx.Graph()
        self.components: dict = dict()
        self.cs_adj_dict: Optional[dict] = None
        self.drugs_in_graph: Optional[list] = None
        self.indications_in_graph: Optional[list] = None
        self.drug_or_indication2proteins: Optional[list] = None
        self.weight: str = "weight"
        self.drug: str = "drug"
        self.indication: str = "indication"
        self.protein: str = "protein"
        self.biological_function: str = "biological_function"
        self.drug_protein: str = "drug-protein"
        self.indication_protein: str = "indication-protein"
        self.protein_protein: str = "protein-protein"
        self.protein_biological_function: str = "protein-biological_function"
        self.biological_function_biological_function: str = "biological_function-biological_function"
        self.up_biological_function: str = "up_biological_function"
        self.down_biological_function: str = "down_biological_function"
        self.nodes: list = [self.drug, self.indication, self.protein, self.biological_function]
        self.edges: list = [self.drug_protein, self.indication_protein, self.protein_protein,
                            self.protein_biological_function, self.biological_function_biological_function]
        # file paths
        self.drug2protein_file_path: str = drug2protein_file
        self.indication2protein_file: str = indication2protein_file
        self.protein2protein_file: str = protein2protein_file
        self.protein2biological_function_file: str = protein2biological_function_file
        self.biological_function2biological_function_file: str = biological_function2biological_function_file
        self.save_load_file_path: str = save_load_file_path
        # directed edge indicators
        self.drug2protein_directed: bool = drug2protein_directed
        self.indication2protein_directed: bool = indication2protein_directed
        self.protein2protein_directed: bool = protein2protein_directed
        self.protein2biological_function_directed: bool = protein2biological_function_directed
        self.biological_function2biological_function_directed: bool = biological_function2biological_function_directed
        # node metadata
        self.node2type: Optional[dict] = None
        self.type2nodes: Optional[dict] = None
        self.node2name: Optional[dict] = None
        self.name2node: Optional[dict] = None
        self.nodelist: Optional[list] = None
        self.node2idx: Optional[dict] = None
        self.idx2node: Optional[dict] = None

    def add_edges(self, edge_list, from_node_type, to_node_type) -> None:
        """Function adds edges from processed file to a Networkx graph. While building the Networkx graph the function
        also adds metadata in the form the node type.

        Args:
            edge_list: A list of tuples where each tuple contains two nodes (e.g., [("DB00363", "3355")]).
            from_node_type: A string containing the subject node type (e.g., "drug").
            to_node_type: A string containing the object node type (e.g., "protein").

        Returns:
             None
        """

        for from_node, to_node in edge_list:
            self.graph.add_edge(from_node, to_node)
            self.graph.nodes[from_node]["type"] = from_node_type
            self.graph.nodes[to_node]["type"] = to_node_type

    @staticmethod
    def merge_one_to_one_dicts(dict_list: list) -> dict:
        """Function combines the node2type dictionaries for all node types into a single dictionary.

        Args:
            dict_list: A list of dictionaries where keys are node identifiers and values are node types.

        Returns:
            None
        """

        out_dict = dict()
        for dict_ in dict_list:
            for k, v in dict_.items():
                if k in out_dict:
                    assert ((out_dict[k] == v) or (pd.isnull(out_dict[k]) and pd.isnull(v)))
                else:
                    out_dict[k] = v
        return out_dict

    def load_node2type(self) -> None:
        """Function loops over all the node2type dictionaries for each node type and calls the merge_one_to_one_dict
        function in order to combine them into a single dictionary.

         Returns:
              None
         """

        node2type__list = []
        for node2node__name, node2node__obj in self.components.items():
            node2type__list.append(node2node__obj.node2type)
        node2type = self.merge_one_to_one_dicts(node2type__list)
        self.node2type = node2type

    def load_type2nodes(self) -> None:
        """Function loops over the node2type dictionary and from it, creates the type2node dictionary by reversing the
        keys and values.

         Returns:
              None
         """

        type2nodes = {}
        for node, type_ in self.node2type.items():
            if type_ in type2nodes:
                type2nodes[type_].add(node)
            else:
                type2nodes[type_] = {node}
        self.type2nodes = type2nodes

    def load_node2name(self) -> None:
        """Function loops over all the node2name dictionaries for each node type and calls the merge_one_to_one_dict
        function in order to combine them into a single dictionary.

         Returns:
              None
         """

        node2name__list = []
        for node2node__name, node2node__obj in self.components.items():
            node2name__list.append(node2node__obj.node2name)
        node2name = self.merge_one_to_one_dicts(node2name__list)
        self.node2name = node2name

    def load_name2node(self) -> None:
        """Function loops over the name2node dictionary and from it, creates the type2node dictionary by reversing the
        keys and values.

         Returns:
              None
         """

        name2node = {v: k for k, v in self.node2name.items()}
        self.name2node = name2node

    def load_graph(self) -> None:
        """Function reads in data for each edge type and process it, so it can be used to build the Networkx graph and
        associated metadata.

        Returns:
             None
        """

        d2p, i2p, p2p = "drug_to_protein", "indication_to_protein", "protein_to_protein"
        p2b, b2b = "protein_to_biological_function", "biological_function_to_biological_function"

        # load components and add edges as appropriate
        print("\t- Processing Drug-Protein Data")
        if (self.drug in self.nodes) and (self.drug_protein in self.edges):
            self.components[d2p] = DrugToProtein(self.drug2protein_directed, self.drug2protein_file_path)
            self.add_edges(self.components[d2p].edge_list, self.drug, self.protein)
        print("\t- Processing Indication-Protein Data")
        if (self.indication in self.nodes) and (self.indication_protein in self.edges):
            self.components[i2p] = IndicationToProtein(self.indication2protein_directed, self.indication2protein_file)
            self.add_edges(self.components[i2p].edge_list, self.indication, self.protein)
        print("\t- Processing Protein-Protein Data")
        if (self.protein in self.nodes) and (self.protein_protein in self.edges):
            self.components[p2p] = ProteinToProtein(self.protein2protein_directed, self.protein2protein_file)
            self.add_edges(self.components[p2p].edge_list, self.protein, self.protein)
        print("\t- Processing Protein-Biological Process Data")
        if (self.biological_function in self.nodes) and (self.protein_biological_function in self.edges):
            self.components[p2b] = ProteinToBiologicalFunction(
                self.protein2biological_function_directed, self.protein2biological_function_file)
            self.add_edges(self.components[p2b].edge_list, self.protein, self.biological_function)
        print("\t- Processing Biological Process-Biological Process Data")
        if (self.biological_function in self.nodes) and (self.biological_function_biological_function in self.edges):
            self.components[b2b] = BiologicalFunctionToBiologicalFunction(
                self.biological_function2biological_function_directed, self.biological_function2biological_function_file)
            self.add_edges(self.components[b2b].edge_list, self.biological_function, self.biological_function)

        # make graph directional (copy forward and reverse of each edge)
        self.graph = self.graph.to_directed()

    def load_node_idx_mapping_and_nodelist(self) -> None:
        """Function gets all the graph nodes and creates a dictionary and saves the following objects: (I) nodelist:
        a list of the node identifiers; (II) node2idx: a dictionary where node identifiers are the keys and  integer
        identifiers are the values; and (III) idx2node: a dictionary where integer identifiers are the keys and node
        identifiers are the values.

        Returns:
            None
        """
        nodes = self.graph.nodes()
        node2idx = dict.fromkeys(nodes)
        nodelist = []
        for idx, node in enumerate(nodes):
            nodelist.append(node)
            node2idx[node] = idx
        idx2node = {v: k for k, v in node2idx.items()}
        self.nodelist, self.node2idx,  self.idx2node = nodelist, node2idx, idx2node

    def load_saved_node_idx_mapping_and_nodelist(self, save_load_file_path: str) -> None:
        """Function reads in the node2idx and idx2node dictionary objects and constructs a list of all node identifiers.

        Args:
            save_load_file_path: A string containing the file path to the node2idx dictionary object.

        Returns:
            None

        Raises:
            FileNotFoundError: if the string containing the file path to the node metadata points to a file that does
                not exist.
        """

        # load node2idx

        if not os.path.exists(save_load_file_path + "msi_graph_node2idx.pkl"):
            raise FileNotFoundError("The {} object does not exist".format(save_load_file_path))
        else:
            with open(save_load_file_path + "msi_graph_node2idx.pkl", "rb") as f:
                node2idx = pickle.load(f)
            self.node2idx = node2idx
            # load idx2node
            self.idx2node = {v: k for k, v in self.node2idx.items()}
            # load nodelist
            nodelist = []
            for i in range(0, len(self.idx2node)):
                nodelist.append(self.idx2node[i])
            self.nodelist = nodelist

    @staticmethod
    def save_msi_object(save_load_file: str, data_obj: dict) -> None:
        """Function pickles a dictionary containing node metadata or the networkx graph and writes it to disc.

        Args:
            save_load_file: A string containing a file path.
            data_obj: A node metadata dictionary object.

        Returns:
            None
        """

        if isinstance(data_obj, nx.Graph):
            with open(save_load_file, "wb") as f:
                pickle.dump(data_obj, f)
        else:
            with open(save_load_file, "wb") as f:
                pickle.dump(data_obj, f)

    def load_drugs_in_graph(self) -> None:
        """Function extracts all node identifiers for nodes of type drug and saves them to a new object called
        drugs_in_graph.

        Returns:
            None
        """

        self.drugs_in_graph = list(self.type2nodes[self.drug])

    def load_indications_in_graph(self) -> None:
        """Function extracts all node identifiers for nodes of type indication and saves them to a new object called indications_in_graph.

        Returns:
            None
        """

        self.indications_in_graph = list(self.type2nodes[self.indication])

    def load_drug_or_indication2proteins(self) -> None:
        """Function takes the list of all drug and indication node identifiers and for each drug and indication
        extracts all proteins that are its neighbors. The function returns a dictionary called
        drug_or_indication2protein, where keys are node identifiers and values are sets of node identifiers
        representing the neighbors.


        Returns:
            None
        """

        # initializes a dictionary with disease and drugs as keys and values as None
        drug_or_indication2proteins = dict.fromkeys(self.drugs_in_graph + self.indications_in_graph)
        for x in self.drugs_in_graph:
            if not nx.is_directed(self.components["drug_to_protein"].graph):
                drug_or_indication2proteins[x] = set(self.components["drug_to_protein"].graph.neighbors(x))
        for y in self.indications_in_graph:
            if not nx.is_directed(self.components["indication_to_protein"].graph):
                drug_or_indication2proteins[y] = set(self.components["indication_to_protein"].graph.neighbors(y))
        self.drug_or_indication2proteins = drug_or_indication2proteins

    def load(self) -> None:
        """Function initializes an empty Networkx graph and populates it; returns a directed graph and creates the components dictionary object which returns everything created for each resource by the NodeToNode class: DrugToProtein, IndicationToProtein, ProteinToProtein, ProteinToBiologicalFunction, BiologicalFunctionToBiologicalFunction

        Each of the above scripts calls NodeToNode, which does the following:
         - load_df - loads csv file and tasks as arguments a file path and separator
         - load_edge_list - creates a directional edge list; assumes loaded df contains two columns with variables
         names “node_1” and “node_2”
         - load_graph - if the graph is directed, instantiate a networkx digraph, otherwise instantiate a graph and
         populate it using the edge list derived in the prior step
         - load_node2type - creates a dictionary where the keys are the node identifiers and the values are a string
         representing the node’s type, and can also be null. To do this, it calls the update_node2attr function
         - load_type2nodes - creates a dictionary where types are keys and values are sets of node identifiers
         - load_node2name - creates a dictionary where node identifiers are the keys and labels containing the node
         names are the values. To do this, it calls the update_node2attr function
         - load_name2node - creates a dictionary where the string labels for nodes are keys and the values are the
         nodes identifier

        Returns:
            None
        """

        print("\n\n" + "*" * 100 + "\nConstructing Multi-Scale Interactome\n" + "*" * 100)
        print("---> Loading Data")
        self.load_graph()
        print("---> Creating Node Indexes, Types, and Labels")
        self.load_node_idx_mapping_and_nodelist()
        self.load_node2type()
        self.load_type2nodes()
        self.load_node2name()
        self.load_name2node()
        print("---> Obtaining Drugs in Graph")
        self.load_drugs_in_graph()
        print("---> Obtaining Indications in Graph")
        self.load_indications_in_graph()
        print("---> Obtaining Protein Neighborhoods for Drugs and Indications")
        self.load_drug_or_indication2proteins()

        # save graph data
        print("---> Saving Multi-Interactome and Node Metadata Dictionaries")
        self.save_msi_object(self.save_load_file_path + "msi_graph.pkl", self.graph)
        self.save_msi_object(self.save_load_file_path + "msi_graph_node2idx.pkl", self.node2idx)
        self.save_msi_object(self.save_load_file_path + "msi_graph_node2name.pkl", self.node2name)
        self.save_msi_object(self.save_load_file_path + "msi_graph_node2type.pkl", self.node2type)

        return None

    def add_to_cs_adj_dict(self, node: str, successor_type: str, successor: str) -> None:
        """Function takes a node identifier, successor_type (i.e., string containing the type of node), and a successor
        (node identifier) and returns a nested dictionary object called cs_adj_dict, where the outer keys are node
        identifiers and the inner dictionary contains keys which are strings of node types ("up_biological_function", "down_biological_function" or the node’s assigned type) and the values are a list of node identifiers.

        Args:
            node: A string containing a node identifier.
            successor_type: A string containing a node type.
            successor: A string containing a node identifier.

        Returns:
            None
        """

        if successor_type in self.cs_adj_dict[node]:
            self.cs_adj_dict[node][successor_type].append(successor)
        else:
            self.cs_adj_dict[node][successor_type] = [successor]

    def create_class_specific_adjacency_dictionary(self) -> None:
        """Function creates the class-specific adjacency matrix where for all nodes, connections are established
        between the node and all of its successors in the graph. For biological processes, an additional step is
        added where each node's up and down successors and predecessors are added to the graph with the successor
        type None.

        The function creates a nested dictionary where the first key is the node id, the second key is the successor
        type, and the value is a list of node ids.

        Returns:
            None
        """

        self.cs_adj_dict = {node: {} for node in self.graph.nodes()}
        for node in self.graph.nodes():
            node_type = self.node2type[node]
            up_neighbors, down_neighbors = [], []
            if node_type == self.biological_function:
                up_neighbors = list(
                    self.components["biological_function_to_biological_function"].graph.successors(node))
                down_neighbors = list(
                    self.components["biological_function_to_biological_function"].graph.predecessors(node))

            successors = self.graph.successors(node)
            for successor in successors:
                successor_type = self.node2type[successor]
                if (node_type == self.biological_function) and (successor_type == self.biological_function):
                    if successor in up_neighbors:
                        self.add_to_cs_adj_dict(node, self.up_biological_function, successor)
                    elif successor in down_neighbors:
                        self.add_to_cs_adj_dict(node, self.down_biological_function, successor)
                    else:
                        assert False
                else:
                    self.add_to_cs_adj_dict(node, successor_type, successor)

    def weight_graph(self, weights: dict) -> None:
        """Function creates a weighted adjacency matrix for all nodes in the msi graph object. The weight for each
        node is derived by dividing the input weight value for a specific node type by the count of successor nodes
        of that node type. This information is also added to the msi Networkx graph object directly with the
        attribute label "weight".

            examples:
                - adj matrix: {node id: {node_type: [node_i, node_i1, ..., node_in}}
                - Networkx graph: [(node_i, node_j), {"weight": float}), ...]

        Args:
            weights: A dictionary keyed by node type and containing probabilities as values.

        Returns:
            None
        """

        print("---> Building Adjacency Matrix for all Nodes in Graph")
        self.create_class_specific_adjacency_dictionary()

        print("---> Weighting Edges")
        for from_node, adj_dict in self.cs_adj_dict.items():
            for node_type, to_nodes in adj_dict.items():
                num_typed_nodes = len(to_nodes)
                for to_node in to_nodes:
                    # node type-specific weight / number of successors for that node type
                    self.graph[from_node][to_node][self.weight] = weights[node_type] / float(num_typed_nodes)
