
import pytest
import pandas as pd

from tidygraphtool.all import *


def get_dat_simple(with_node_meta=False):

    if with_node_meta == False:
      nodes = pd.DataFrame({"name": ["Bob", "Alice", "Joan", "Melvin"]})
    else:
      nodes = pd.DataFrame({"name": ["Bob", "Alice", "Joan", "Melvin"],
                            "job":  ["teacher", "driver", "dancer", "influence"]})
    
    edges = pd.DataFrame({"source": ["Bob", "Bob", "Joan", "Alice"],
                          "target": ["Joan", "Melvin", "Alice", "Joan"],
                          "weight": [3,3,2,10]})
    return nodes, edges

def test_filter_on_nodes():
  _, edges = get_dat_simple(with_node_meta=True)
  g = as_gt_graph(edges) 
  df = g >> filter_on('label == "Melvin"') >> as_data_frame()
  assert df.label.values[0] == 'Melvin'


def test_add_property():
  nodes, edges = get_dat_simple(with_node_meta=True)
  g = gt_graph(nodes=nodes, edges=edges)
  g = add_property(g, "degree_in", centrality_degree, mode="total")
  assert "degree_in" in as_data_frame(g).columns


def test_add_property_pipe():
  nodes, edges = get_dat_simple(with_node_meta=True)
  g = gt_graph(nodes=nodes, edges=edges)
  g = (g >> 
        add_property("deg", centrality_degree) >>
        add_property("pr", centrality_pagerank) >>
        add_property("bet", centrality_betweenness))
  assert all([prop in as_data_frame(g).columns 
             for prop 
             in ["deg", "pr", "bet"]])
