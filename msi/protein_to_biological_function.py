#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

# import needed libraries
from msi.node_to_node import NodeToNode


class ProteinToBiologicalFunction(NodeToNode):
	"""
	Class inherits functionality from the NodeToNode Class.
	"""
	def __init__(self, file_path, sep="\t") -> None:
		super().__init__(file_path, sep)
