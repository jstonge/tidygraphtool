from typing import Optional, Union
import re

import graph_tool.all as gt
import pandas as pd

from .as_data_frame import as_data_frame
from .edgedataframe import EdgeDataFrame, EdgeSeries
from .nodedataframe import NodeDataFrame, NodeSeries
from .utils import (
    assert_edges_edges_equal, 
    guess_df_type, 
    check_column, 
    check_null_values, 
    _convert_types2gt, 
    assert_nodes_nodes_equal,
    assert_edges_edges_equal,
    assert_index_reset
)

def augment_prop(
    G: gt.Graph,
    x: pd.DataFrame,
    prop_name: Optional[str] = None
) -> gt.Graph:
    """
    Augment nodes in Graph with additional nodes metadata.

    As opposed to mutate, augment_prop modifies in-place the graph.

    :param G: A `gt.Graph` object.
    :param x: A `data Frame` containing information about the nodes in the G. 
    :param prop_name: String that matches the colname. 
    :return: A `Graph_tool` object
    """
    
    if isinstance(x, (NodeDataFrame, NodeSeries)):
        return _augment_prop_nodes(G, x, prop_name)
    elif isinstance(x, (EdgeDataFrame, EdgeSeries)):
        return _augment_prop_edges(G, x, prop_name)
    elif guess_df_type(x) == 'NodeDataFrame':
        return _augment_prop_nodes(G, x, prop_name)
    elif guess_df_type(x) == 'EdgeDataFrame':
        return _augment_prop_edges(G, x, prop_name)
    else:
        raise ValueError("Unknown format")

def _augment_prop_nodes(G: gt.Graph, 
                        nodes: Union[NodeDataFrame, NodeSeries], 
                        prop_name: Optional[str] = None) -> pd.DataFrame:

    nodes=NodeDataFrame(nodes)

    if prop_name is None:
        prop_name = nodes.columns[0]

    # check_column(nodes, column_names = [prop_name], present=True)
    check_null_values(nodes, [prop_name])
    
    # Create internal properties maps
    prop_type = _convert_types2gt(nodes, prop_name)
    vprop = G.new_vp(f"{prop_type}")
    G.vertex_properties[f"{prop_name}"] = vprop

    assert_nodes_nodes_equal(G, nodes)
    assert_index_reset(nodes)
    
    if 'name' in nodes.columns:
        if all([len(_) == 0 for _ in G.vp.name]) == False:
            nodes = _reorder_nodes_like_g(G, nodes)

    for i in range(len(nodes)):
        # Augment graph with nodes metadata
        vprop[i] = nodes[f"{prop_name}"][i]
    return G


def _reorder_nodes_like_g(G, nodes):
    ordered_name_g = list(G.vp.name)
    x = _find_namecol(G, nodes)
    nodes = nodes.set_index(nodes[f'{x}'])
    nodes = nodes.loc[ordered_name_g]
    nodes = nodes.reset_index(drop=True)
    return nodes

def _find_namecol(G, nodes):
    names_g = set(list(G.vp.name))
    return [c for c in nodes.columns 
            if set(nodes[f"{c}"]) == names_g][0]


def _augment_prop_edges(G: gt.Graph, 
                        edges: Union[EdgeDataFrame, EdgeSeries], 
                        prop_name: str) -> pd.DataFrame:

    edges = EdgeDataFrame(edges)

    # check_column(edges, column_names = [prop_name])
    check_null_values(edges, [prop_name])
    
    # Create internal properties maps
    prop_type = _convert_types2gt(edges, prop_name)
    eprop = G.new_ep(f"{prop_type}")
    G.edge_properties[f"{prop_name}"] = eprop
    
    assert_edges_edges_equal(G, edges)
    assert_index_reset(edges)
    
    if 'source' not in edges:
        edges_df = as_data_frame(G)
        edges = pd.concat([edges_df, edges])
    
    for i in range(len(edges)):
        # Augment graph with edge metadata
        e =  G.edge(edges.source[i], edges.target[i])
        eprop[e] = edges.loc[i, f"{prop_name}"]
    return G

