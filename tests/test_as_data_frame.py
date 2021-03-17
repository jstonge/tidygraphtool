
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

def get_dat2():
    """
    Edges already indexed. Thus they do not correspond to nodes name.

    This situation is typical when we convert from R tidygraph to python.
    """
    nodes = pd.DataFrame({"id": [1,2,3,4], 
                          "name": ["Bob", "Alice", "Joan", "Melvin"],
                          "job":  ["teacher", "driver", "dancer", "influence"]})
    edges = pd.DataFrame({"source": [1, 1, 3, 2],
                          "target": [3, 4, 3, 3],
                          "weight": [3,3,2,10]})
    return nodes, edges

def test_as_data_frame_edges():
  nodes, edges = get_dat()
  g=gt_graph(nodes=nodes, edges=edges)
  nodes_g = as_data_frame(g)
  activate(g, "edges")
  edges_g = as_data_frame(g)

  loc_B = float(nodes_g.query("label == 'Bob'")['name'].values[0])
  loc_J = float(nodes_g.query("label == 'Joan'")['name'].values[0])
  weight_g_BJ = edges_g.query("source==@loc_B & target==@loc_J")["weight"].values[0] == 3
  weight_BJ = edges.query("source=='Bob' & target=='Joan'")['weight'].values[0] == 3
  assert weight_BJ == weight_g_BJ

def test_as_data_frame_diff_node_key():
  nodes, edges = get_dat2()
  g=gt_graph(nodes=nodes, edges=edges, node_key="id")
  df = as_data_frame(g)
  assert df.columns[0] == 'id'


def test_as_data_frame_many_props():
  _, edges = get_dat()
  g = gt_graph(edges=edges)
  activate(g, "edges")
  eprop = g.new_ep("bool")
  g.ep["bp"] = eprop
  assert type(as_data_frame(g)) == pd.DataFrame
