"""Calculate metrics on graph"""

import graph_tool.all as gt

from .nodedataframe import NodeDataFrame
from .context import expect_nodes


def graph_component_count(G: gt.Graph, 
                          directed: bool = False) -> NodeDataFrame:
  expect_nodes(G)
  counted_comp, _ = gt.label_components(G, directed=directed)
  return NodeDataFrame({"cc": list(counted_comp)})["cc"]


def graph_largest_component(G: gt.Graph, 
                            directed: bool = False) -> NodeDataFrame:
  expect_nodes(G)
  largest_comp = gt.label_largest_component(G, directed=directed)
  return NodeDataFrame({"lc": list(largest_comp)})["lc"]
