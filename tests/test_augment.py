
import pytest
import pandas as pd

from tidygraphtool.gt_graph import *


def get_dat():
    nodes = pd.DataFrame({"name": ["Bob", "Alice", "Joan", "Melvin"],
                          "job":  ["teacher", "driver", "dancer", "influence"]}) 
    edges = pd.DataFrame({"source": ["Bob", "Bob", "Joan", "Alice"],
                          "target": ["Joan", "Melvin", "Alice", "Joan"],
                          "weight": [3,3,2,10],
                          "type": ["pro", "friends", "pro", "friends"]})
    return nodes, edges

def test_augment_nodes():
  nodes, edges = get_dat()
  g = as_gt_graph(edges)
  augment_prop(g, nodes, "job")
  
  nodes_g = as_data_frame(g).sort_values("label")\
                            .loc[:, ["label", "job"]]\
                            .to_dict(orient="list")
  
  nodes = nodes.sort_values("name")\
               .rename(columns={"name":"label"})\
               .to_dict(orient="list")
  
  assert nodes == nodes_g

def test_augment_edges():
  nodes, edges = get_dat()
  g = as_gt_graph(edges)
  augment_prop(g, nodes, "job")
  
  nodes_g = as_data_frame(g).sort_values("label")\
                            .loc[:, ["label", "job"]]\
                            .to_dict(orient="list")
  
  nodes = nodes.sort_values("name")\
               .rename(columns={"name":"label"})\
               .to_dict(orient="list")
  
  assert nodes == nodes_g