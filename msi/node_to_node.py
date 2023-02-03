import pandas as pd
# import os
import networkx as nx


class NodeToNode:
    def __init__(self, directed, file_path, sep="\t"):
        self.file_path = file_path
        self.directed = directed
        self.sep = sep
        self.df, self.edge_list, self.graph, self.node2type = None, None, None, None
        self.type2nodes, self.node2name, self.name2node = None, None, None
        self.load()

    def load_df(self):
        self.df = pd.read_csv(self.file_path, sep=self.sep, index_col=False, dtype=str)

    def load_edge_list(self):
        # creates directional edge list from node_1 to node_2
        if not (self.df is None):
            edge_list = list(zip(self.df["node_1"], self.df["node_2"]))
            self.edge_list = edge_list

    def load_graph(self):
        if self.directed:
            self.graph = nx.DiGraph()
        else:
            self.graph = nx.Graph()
        self.graph.add_edges_from(self.edge_list)

    def update_node2attr(self, node2type, col_1, col_2):
        for node, type_ in zip(self.df[col_1], self.df[col_2]):
            if node in node2type:
                assert ((node2type[node] == type_) or (pd.isnull(node2type[node]) and pd.isnull(type_)))
            else:
                node2type[node] = type_
        return node2type

    def load_node2type(self):
        if not (self.df is None):
            node2type = dict()
            node2type = self.update_node2attr(node2type, "node_1", "node_1_type")
            node2type = self.update_node2attr(node2type, "node_2", "node_2_type")
            self.node2type = node2type

    def load_type2nodes(self):
        type2nodes = dict()
        for node, type_ in self.node2type.items():
            if type_ in type2nodes:
                type2nodes[type_].add(node)
            else:
                type2nodes[type_] = {node}
        self.type2nodes = type2nodes

    def load_node2name(self):
        if not (self.df is None):
            node2name = dict()
            node2name = self.update_node2attr(node2name, "node_1", "node_1_name")
            node2name = self.update_node2attr(node2name, "node_2", "node_2_name")
            self.node2name = node2name

    def load_name2node(self):
        if not (self.df is None):
            name2node = {v: k for k, v in self.node2name.items()}
            self.name2node = name2node

    def load(self):
        if not (self.file_path is None):
            self.load_df()
            self.load_edge_list()
            self.load_graph()
            self.load_node2type()
            self.load_type2nodes()
            self.load_node2name()
            self.load_name2node()
