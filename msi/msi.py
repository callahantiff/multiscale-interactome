from msi.drug_to_protein import DrugToProtein
from msi.indication_to_protein import IndicationToProtein
from msi.protein_to_protein import ProteinToProtein
from msi.protein_to_biological_function import ProteinToBiologicalFunction
from msi.biological_function_to_biological_function import BiologicalFunctionToBiologicalFunction

import networkx as nx
import pandas as pd
import os
import pickle


class MSI:
    def __init__(self, drug2protein_file_path, indication2protein_file_path, protein2protein_file_path,
                 protein2biological_function_file_path, biological_function2biological_function_file_path,
                 drug2protein_directed=False, indication2protein_directed=False, protein2protein_directed=False,
                 protein2biological_function_directed=False, biological_function2biological_function_directed=True):
        # parameters
        self.cs_adj_dict, self.drug_or_indication2proteins, self.node2type, self.type2nodes = None, None, None, None
        self.node2name, self.name2node, self.nodelist, self.node2idx, self.idx2node = None, None, None, None, None
        self.indications_in_graph, self.drugs_in_graph = None, None
        self.graph = nx.Graph()
        self.components = dict()
        self.drug_protein = "drug-protein"
        self.indication_protein = "indication-protein"
        self.protein_protein = "protein-protein"
        self.protein_biological_function = "protein-biological_function"
        self.biological_function_biological_function = "biological_function-biological_function"
        self.drug = "drug"
        self.indication = "indication"
        self.protein = "protein"
        self.biological_function = "biological_function"
        self.up_bp = "up_biological_function"
        self.down_bp = "down_biological_function"
        self.weight = "weight"
        self.nodes = [self.drug, self.indication, self.protein, self.biological_function]
        self.edges = [self.drug_protein, self.indication_protein, self.protein_protein,
                      self.protein_biological_function, self.biological_function_biological_function]
        # file paths
        self.drug2protein_file_path = drug2protein_file_path
        self.indication2protein_file_path = indication2protein_file_path
        self.protein2protein_file_path = protein2protein_file_path
        self.protein2biological_function_file_path = protein2biological_function_file_path
        self.biological_function2biological_function_file_path = biological_function2biological_function_file_path
        # directed indicators
        self.drug2protein_directed = drug2protein_directed
        self.indication2protein_directed = indication2protein_directed
        self.protein2protein_directed = protein2protein_directed
        self.protein2biological_function_directed = protein2biological_function_directed
        self.biological_function2biological_function_directed = biological_function2biological_function_directed

    def add_edges(self, edge_list, from_node_type, to_node_type):
        for from_node, to_node in edge_list:
            self.graph.add_edge(from_node, to_node)
            self.graph.nodes[from_node]["type"] = from_node_type
            self.graph.nodes[to_node]["type"] = to_node_type

    @staticmethod
    def merge_one_to_one_dicts(dict_list):
        out_dict = dict()
        for dict_ in dict_list:
            for k, v in dict_.items():
                if k in out_dict:
                    assert ((out_dict[k] == v) or (pd.isnull(out_dict[k]) and pd.isnull(v)))
                else:
                    out_dict[k] = v
        return out_dict

    def load_node2type(self):
        # merge the node2type of each component (these are 1:1 dictionaries)
        node2type__list = []
        for node2node__name, node2node__obj in self.components.items():
            node2type__list.append(node2node__obj.node2type)
        node2type = self.merge_one_to_one_dicts(node2type__list)
        self.node2type = node2type

    def load_type2nodes(self):
        type2nodes = {}
        for node, type_ in self.node2type.items():
            if type_ in type2nodes:
                type2nodes[type_].add(node)
            else:
                type2nodes[type_] = {node}
        self.type2nodes = type2nodes

    def load_node2name(self):
        node2name__list = []
        for node2node__name, node2node__obj in self.components.items():
            node2name__list.append(node2node__obj.node2name)
        node2name = self.merge_one_to_one_dicts(node2name__list)
        self.node2name = node2name

    def load_name2node(self):
        name2node = {v: k for k, v in self.node2name.items()}
        self.name2node = name2node

    def load_graph(self):
        d2p, i2p, p2p = "drug_to_protein", "indication_to_protein", "protein_to_protein"
        p2b, b2b = "protein_to_biological_function", "biological_function_to_biological_function"
        # load components and add edges as appropriate
        print('---> Processing Drug-Protein Data')
        if (self.drug in self.nodes) and (self.drug_protein in self.edges):
            self.components[d2p] = DrugToProtein(self.drug2protein_directed, self.drug2protein_file_path)
            self.add_edges(self.components[d2p].edge_list, self.drug, self.protein)
        print('---> Processing Indication-Protein Data')
        if (self.indication in self.nodes) and (self.indication_protein in self.edges):
            self.components[i2p] = IndicationToProtein(
                self.indication2protein_directed, self.indication2protein_file_path)
            self.add_edges(self.components[i2p].edge_list, self.indication, self.protein)
        print('---> Processing Protein-Protein Data')
        if (self.protein in self.nodes) and (self.protein_protein in self.edges):
            self.components[p2p] = ProteinToProtein(self.protein2protein_directed, self.protein2protein_file_path)
            self.add_edges(self.components[p2p].edge_list, self.protein, self.protein)
        print('---> Processing Protein-Biological Process Data')
        if (self.biological_function in self.nodes) and (self.protein_biological_function in self.edges):
            self.components[p2b] = ProteinToBiologicalFunction(
                self.protein2biological_function_directed, self.protein2biological_function_file_path)
            self.add_edges(self.components[p2b].edge_list, self.protein, self.biological_function)
        print('---> Processing Biological Process-Biological Process Data')
        if (self.biological_function in self.nodes) and (self.biological_function_biological_function in self.edges):
            self.components[b2b] = BiologicalFunctionToBiologicalFunction(
                self.biological_function2biological_function_directed,
                self.biological_function2biological_function_file_path)
            self.add_edges(self.components[b2b].edge_list, self.biological_function, self.biological_function)
        self.graph = self.graph.to_directed()  # make graph directional (copy forward and reverse of each edge)

    def load_node_idx_mapping_and_nodelist(self):
        nodes = self.graph.nodes()
        node2idx = dict.fromkeys(nodes)
        nodelist = []
        for idx, node in enumerate(nodes):
            nodelist.append(node)
            node2idx[node] = idx
        idx2node = {v: k for k, v in node2idx.items()}
        # save
        self.nodelist, self.node2idx,  self.idx2node = nodelist, node2idx, idx2node

    def load_saved_node_idx_mapping_and_nodelist(self, save_load_file_path):
        # load node2idx
        assert (os.path.exists(save_load_file_path))
        with open(save_load_file_path, "rb") as f:
            node2idx = pickle.load(f)
        self.node2idx = node2idx
        # load idx2node
        self.idx2node = {v: k for k, v in self.node2idx.items()}
        # load nodelist
        nodelist = []
        for i in range(0, len(self.idx2node)):
            nodelist.append(self.idx2node[i])
        self.nodelist = nodelist

    def save_node2idx(self, save_load_file_path):
        # assert(not(os.path.isfile(node2idx_file_path)))
        with open(save_load_file_path, "wb") as f:
            pickle.dump(self.node2idx, f)

    def load_drugs_in_graph(self):
        self.drugs_in_graph = list(self.type2nodes[self.drug])

    def load_indications_in_graph(self):
        self.indications_in_graph = list(self.type2nodes[self.indication])

    def load_drug_or_indication2proteins(self):
        # initializes a dictionary with disease and drugs as keys and values as None
        drug_or_indication2proteins = dict.fromkeys(self.drugs_in_graph + self.indications_in_graph)
        for x in self.drugs_in_graph:
            if not nx.is_directed(self.components["drug_to_protein"].graph):
                drug_or_indication2proteins[x] = set(self.components["drug_to_protein"].graph.neighbors(x))
        for y in self.indications_in_graph:
            if not nx.is_directed(self.components["indication_to_protein"].graph):
                drug_or_indication2proteins[y] = set(self.components["indication_to_protein"].graph.neighbors(y))
        self.drug_or_indication2proteins = drug_or_indication2proteins

    def load(self):
        print('*' * 100 + '\nLoading Data\n' + '*' * 100)
        self.load_graph()
        print('*' * 100 + '\nCreating Node Indexes, Types, and Labels\n' + '*' * 100)
        self.load_node_idx_mapping_and_nodelist()
        self.load_node2type()
        self.load_type2nodes()
        self.load_node2name()
        self.load_name2node()
        print('*' * 100 + '\nObtaining Drugs in Graph\n' + '*' * 100)
        self.load_drugs_in_graph()
        print('*' * 100 + '\nObtaining Diseases in Graph\n' + '*' * 100)
        self.load_indications_in_graph()
        print('*' * 100 + '\nObtaining Protein Neighborhoods for Drugs and Diseases\n' + '*' * 100)
        self.load_drug_or_indication2proteins()

    def save_graph(self, save_load_file_path):
        # graph_file_path = os.path.join(save_load_file_path, "graph.pkl")
        with open(save_load_file_path, "wb") as f:
            pickle.dump(self.graph, f)

    def add_to_cs_adj_dict(self, node, successor_type, successor):
        if successor_type in self.cs_adj_dict[node]:
            self.cs_adj_dict[node][successor_type].append(successor)
        else:
            self.cs_adj_dict[node][successor_type] = [successor]

    def create_class_specific_adjacency_dictionary(self):
        """Function creates the class-specific adjacency matrix where for all nodes, connections are established
        between the node and all of it's successors in the graph. For biological functions, an additional step is
        added where each node's up and down successors and predecessors are added to the graph with the successor
        type None.

        The function returns a nested dictionary where the first key is the node id, the second key is the successor
        type, and the value is a list of node ids.
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
                        self.add_to_cs_adj_dict(node, self.up_bp, successor)
                    elif successor in down_neighbors:
                        self.add_to_cs_adj_dict(node, self.down_bp, successor)
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
            - adj matrix: {node id: {node_type: [nodei, nodei1, ..., nodein}}
            - Networkx graph: [(i, j), {'weight': float}), ...]

        :param weights: A dictionary keyed by node type and containing probabilities as values.
        :return:
            None.
        """
        print('---> Building Adjacency Matrix for all Nodes in Graph')
        self.create_class_specific_adjacency_dictionary()

        print('---> Weighting Edges')
        for from_node, adj_dict in self.cs_adj_dict.items():
            for node_type, to_nodes in adj_dict.items():
                num_typed_nodes = len(to_nodes)  # number of successors for a specific node type
                for to_node in to_nodes:
                    self.graph[from_node][to_node][self.weight] = weights[node_type] / float(num_typed_nodes)
