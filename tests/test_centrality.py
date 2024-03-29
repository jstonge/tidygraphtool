import pytest
import pandas as pd

from tidygraphtool.all import *


def get_dat_simple():
    nodes = pd.DataFrame({"name": ["Bob", "Alice", "Joan", "Melvin"]}) 
    edges = pd.DataFrame({"source": ["Bob", "Bob", "Joan", "Alice"],
                          "target": ["Joan", "Melvin", "Alice", "Joan"]})
    return nodes, edges


def test_centrality_degree():
  nodes, edges = get_dat_simple()
  g = gt_graph(nodes=nodes, edges=edges)
  g = add_property(g, "degree_tot", centrality_degree, mode="total")
  nodes_df = as_data_frame(g)
  degree_joan = nodes_df[nodes_df.label == 'Joan']['degree_tot'].values
  degree_bob = nodes_df[nodes_df.label == 'Bob']['degree_tot'].values
  assert degree_joan[0] == 3
  assert degree_bob[0] == 2

