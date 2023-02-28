#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

# import needed libraries
from msi.node_to_node import NodeToNode


class IndicationToProtein(NodeToNode):
	"""
	Class inherits functionality from the NodeToNode Class.
	"""
	def __init__(self, directed, file_path, sep="\t") -> None:
		super().__init__(directed, file_path, sep)
