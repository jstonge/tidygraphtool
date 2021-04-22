import pytest
import pandas as pd

from tidygraphtool.gt_graph import *


def get_dat_simple():
    nodes = pd.DataFrame({"name": ["Bob", "Alice", "Joan", "Melvin"]})
    edges = pd.DataFrame({"source": ["Bob", "Bob", "Joan", "Alice"],
                          "target": ["Joan", "Melvin", "Alice", "Joan"]})
    return nodes, edges


def get_dat_edges_already_indexed():
    """
    Edges already indexed. Thus they do not correspond to nodes name.

    This situation is typical when we convert from R tidygraph to python.
    """
    nodes = pd.DataFrame({"name": ["Bob", "Alice", "Joan", "Melvin"],
                          "id": [1,2,3,4]})
    edges = pd.DataFrame({"source": [1, 1, 3, 2],
                          "target": [3, 4, 3, 3]})
    return nodes, edges


def test_gt_graph_dict_index():
    nodes, edges = get_dat_simple()
    g = gt_graph(nodes=nodes, edges=edges)
    nodes_g = as_data_frame(g)
    g = activate(g, "edges")
    edges_g = as_data_frame(g)
    loc_bob_joan = list(nodes_g[nodes_g.label.isin(['Bob', 'Joan'])].index)
    assert list(edges_g.iloc[0, 0:2].astype(int)) == loc_bob_joan


def test_gt_graph_data_frame_index():
    _, edges = get_dat_simple()
    g = as_gt_graph(edges)
    nodes_g = as_data_frame(g)
    g = activate(g, "edges")
    edges_g = as_data_frame(g)
    loc_bob = list(nodes_g[nodes_g.label == 'Bob'].index)
    loc_joan = list(nodes_g[nodes_g.label == 'Joan'].index)
    x = edges_g.query("source == @loc_bob & target == @loc_joan")
    assert list(x.values[0]) == [loc_bob, loc_joan]


def test_gt_graph_dict_diff_index():
    nodes, edges = get_dat_edges_already_indexed()
    g = gt_graph(nodes=nodes, edges=edges, node_key='id')
    assert set(edges.source).union(set(edges.target)) == set(nodes.id.astype(str))


