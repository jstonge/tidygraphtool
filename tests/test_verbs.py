
import pytest
import pandas as pd

from tidygraphtool.all import *


def get_dat(with_node_meta=False):

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
  _, edges = get_dat()
  g = as_gt_graph(edges)
  df = as_data_frame(filter_on(g, 'label == "Melvin"'))
  assert df.label.values[0] == 'Melvin'


def test_filter_on_edges():
  _, edges = get_dat()
  g = as_gt_graph(edges)
  activate(g, "edges")
  df = as_data_frame(filter_on(g, 'weight == 10'))
  assert df.weight.values[0] == 10
