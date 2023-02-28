#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

# import needed libraries
import pandas as pd
import networkx as nx

from typing import Optional


class NodeToNode:
    def __init__(self, directed, file_path, sep="\t") -> None:
        self.file_path: str = file_path
        self.directed: bool = directed
        self.sep: str = sep
        self.df: Optional[pd.DataFrame] = None
        self.graph: Optional[nx.Graph] = None
        self.edge_list: Optional[list] = None
        self.node2type: Optional[dict] = None
        self.type2nodes: Optional[dict] = None
        self.node2name: Optional[dict] = None
        self.name2node: Optional[dict] = None

        # run class to load and process edge data
        self.load()

    def load_df(self) -> None:
        """Function uses variables set in the class init to load edge data. This function assumes that the input data
        is contained in a tsv file and that it contains the following columns:
          - node_1: a string containing subject node identifiers (e.g., DB12010)
          - node_2: a string containing object node identifiers (e.g., 90)
          - node_1_type: a string indicating node_1 type (e.g., drug)
          - node_2_type: a string indicating node_2 type (e.g., protein)
          - node_1_name: a string containing the label for node_1
          - node_2_name: a string containing the label for node_2

        Returns:
            None
        """

        self.df = pd.read_csv(self.file_path, sep=self.sep, index_col=False, dtype=str)

    def load_edge_list(self) -> None:
        """Function creates directional edge list from node_1 to node_2.

        Returns:
            None
        """

        if not (self.df is None):
            edge_list = list(zip(self.df["node_1"], self.df["node_2"]))
            self.edge_list = edge_list

    def load_graph(self) -> None:
        """Function creates Networkx graph from an edge list. Graphs can be created as directed or undirected.

        Returns:
            None
        """

        if self.directed:
            self.graph = nx.DiGraph()
        else:
            self.graph = nx.Graph()

        self.graph.add_edges_from(self.edge_list)

    def update_node2attr(self, nodedict, col_1, col_2) -> dict:
        """Function creates the node2type and node2name dictionary objects, where keys are node identifiers and values
        are node types. An example of what the data in this object look like is provided below:
            node2type: {'DB12010': 'drug', '2167': 'protein', ..., 'GO:0008150': 'biological_function'}
            node2name: {'DB12010': 'fostamatinib', '2167': 'FABP4', ..., 'GO:0008150': 'biological_process'}

        Args:
            nodedict: a dictionary that may or may not already contain data.
            col_1: A string containing the name of the column that contains the type of node_1 or the subject node.
            col_2: A string containing the name of the column that contains the type of node_2 or the object node.

        Returns:
            None
        """

        for node, type_ in zip(self.df[col_1], self.df[col_2]):
            if node in nodedict:
                assert ((nodedict[node] == type_) or (pd.isnull(nodedict[node]) and pd.isnull(type_)))
            else:
                nodedict[node] = type_
        return nodedict

    def load_node2type(self) -> None:
        """Function processes data needed to build the node2type dictionary.

        Returns:
             None
        """

        if not (self.df is None):
            node2type = dict()
            node2type = self.update_node2attr(node2type, "node_1", "node_1_type")
            node2type = self.update_node2attr(node2type, "node_2", "node_2_type")
            self.node2type = node2type

    def load_type2nodes(self) -> None:
        """Function processes data needed to build the type2node dictionary by reversing the node2type dictionary. The
        keys are node types (i.e., 'drug', 'protein', 'indication', and 'biological_function') and the values are sets
        of node identifiers. An example of what the data in this object look like is provided below:
            type2nodes: {'drug': {'DB06775', 'DB00300', 'DB09074', ..., 'DB00212'}, ...}

        Returns:
             None
        """

        type2nodes = dict()
        for node, type_ in self.node2type.items():
            if type_ in type2nodes:
                type2nodes[type_].add(node)
            else:
                type2nodes[type_] = {node}
        self.type2nodes = type2nodes

    def load_node2name(self) -> None:
        """Function processes data needed to build the node2name dictionary.

        Returns:
             None
        """

        if not (self.df is None):
            node2name = dict()
            node2name = self.update_node2attr(node2name, "node_1", "node_1_name")
            node2name = self.update_node2attr(node2name, "node_2", "node_2_name")
            self.node2name = node2name

    def load_name2node(self) -> None:
        """Function processes data needed to build the name2node dictionary by reversing the node2name dictionary. The
        keys are node labels and the values are node identifiers. An example of what the data in this object look like
        is provided below:
            type2nodes: {'fostamatinib': 'DB12010', 'FABP4': '2167', ..., 'biological_process': 'GO:0008150'}

        Returns:
             None
        """

        if not (self.df is None):
            name2node = {v: k for k, v in self.node2name.items()}
            self.name2node = name2node

    def load(self) -> None:
        """Function serves as the main function processing data for all edge types to create a Networkx graph and build
        the associated node metadata dictionaries.

        Returns:
            None
        """

        if not (self.file_path is None):
            self.load_df()
            self.load_edge_list()
            self.load_graph()
            self.load_node2type()
            self.load_type2nodes()
            self.load_node2name()
            self.load_name2node()
