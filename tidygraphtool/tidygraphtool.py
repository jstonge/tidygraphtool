"""Main module."""
import collections
import datetime as dt
import re
import warnings
from collections import Counter
from functools import partial, reduce, singledispatch
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Pattern,
    Set,
    Tuple,
    Union
)

import numpy as np
import graph_tool.all as gt
import pandas as pd
import networkx as nx

from .utils import (
    check,
    check_column,
    assert_nodes_edges_equal
)

def _augment_prop_nodes(G: gt.Graph, 
                       nodes: pd.DataFrame, 
                       prop_name: Optional[str] = None) -> pd.DataFrame:
    
    if prop_name is None:
        prop_name = nodes.columns[0]
    
    check_column(nodes, column_names = [prop_name], present=True)
    if nodes[f'{prop_name}'].isnull().values.any() == True:
        raise ValueError("There are NAs in the col")
    
    prop_type = str(nodes[f"{prop_name}"].dtype)
    if prop_type == "object":
        prop_type = "string"
    elif prop_type == "float64":
        prop_type = "float"
    elif re.match(r"^int", prop_type): 
        prop_type = prop_type + "_t"

    np = G.new_vp(f"{prop_type}")
    G.vertex_properties[f"{prop_name}"] = np

    if len(list(G.vertices())) == len(nodes):
        for i in range(len(nodes)):
            np[i] = nodes[f"{prop_name}"][i]
        return G
    else:
        raise ValueError("Nodes in G has not the same length than nodes data frame.")


def _augment_prop_edges(G: gt.Graph, edges: pd.DataFrame, prop_name: str) -> pd.DataFrame:
    check_column(edges, column_names = [prop_name])
       
    if edges[f'{prop_name}'].isnull().values.any() == True:
        raise ValueError("There are NAs in the col")

    prop_type = str(edges[f"{prop_name}"].dtype)
    if prop_type == "object":
        prop_type = "string"
    elif prop_type == "float64":
        prop_type = "float"
    elif re.match(r"^int", prop_type): 
        prop_type = prop_type + "_t"
    else:
        raise ValueError("Failed to guess type")
    
    ep = G.new_ep(f"{prop_type}")
    G.edge_properties[f"{prop_name}"] = ep

    if len(list(G.edges())) == len(edges):
        for i in range(len(edges)):
            e =  G.edge(edges.source[i], edges.target[i])
            ep[e] = edges.loc[i, f"{prop_name}"]
        return G
    else:
        raise ValueError("Edges in G has not the same length than edges data frame.")


def augment_prop(
    G: gt.Graph,
    nodes: pd.DataFrame = None,
    edges: pd.DataFrame = None,
    prop_name: Optional[str] = None
) -> gt.Graph:
    """
    Augment nodes in Graph with additional nodes metadata.
    
    :param G: A `gt.Graph` object.
    :param nodes: A `data Frame` containing information about the nodes in the G. 
    :param prop_name: String that matches the colname. 
    :param prop_type: Column encoded as `n`  
    :return: A `Graph_tool` object
    """
    if nodes is not None and edges is None:
        _augment_prop_nodes(G, nodes=nodes, prop_name=prop_name)
    elif nodes is None and edges is not None:
        _augment_prop_edges(G, edges=edges, prop_name=prop_name)


def gt_graph(
    edges: pd.DataFrame, 
    nodes: Optional[pd.DataFrame] = None, 
    directed: bool = True,
    node_key: str = "name"
) -> gt.Graph:
    """
    Creates a graph from scratch using nodes and edges dataframe.

    Example
    =======
    .. code-block:: python

        nodes = pd.DataFrame({"name": ["Bob", "Alice", "Joan", "Melvin"}]) 
        edges = pd.DataFrame({"source": ["Bob", "Bob", "Joan", "Alice"],
                              "target": ["Joan", "Melvin", "Alice", "Joan"]})
        create_gt_graph(nodes, edges)

    :param nodes: A `data frame` containing information about the nodes in the G
    :param edges: A `data Frame` containing information about the nodes in the G. 
                 Following networkx convention, the terminal nodes must be encoded 
                 as `target` and `source`.
    :param directed: Boolean
    :return: A `Graph_tool` object
    """
    return as_gt_graph([nodes, edges], directed=directed, node_key=node_key)


@singledispatch
def as_gt_graph(x, directed=True, node_key='name'):
    return x


@as_gt_graph.register(nx.classes.graph.Graph)
def _networkx_graph(x, directed=True):
    """Convert networkx graph, and returns a graph-tool graph."""
    edges = networkx_to_df(x)
    g = gt.Graph(directed=directed)
    g.add_edge_list(edges[["source","target"]].to_numpy())
    return g


@as_gt_graph.register(pd.DataFrame)
def _data_frame(x, directed=True):
    """Convert networkx graph, and returns a graph-tool graph."""
    edges = as_graph_edge_df(x)
    g = gt.Graph(directed=directed)
    g.add_edge_list(edges[["source","target"]].to_numpy())
    # [augment_prop(g, edges=edges, prop_name=_) for _ in edges.iloc[:, 2:-1].columns]
    return g


@as_gt_graph.register(list)
def _list(x, directed=True, node_key='name'):
    """Convert list containing nodes and edges, and returns a graph-tool graph."""
    nodes, edges = as_graph_node_edge(x, node_key=node_key)
    g = gt.Graph(directed=directed)
    g.add_edge_list(edges[["source","target"]].to_numpy())
    augment_prop(g, nodes=nodes, prop_name="name")
    # [augment_prop(g, edges=edges, prop_name=_) for _ in edges.iloc[:, 2:-1].columns]
    # [augment_prop(g, nodes=nodes, prop_name=_) for _ in nodes.columns]
    return g


def networkx_to_df(G):
    # !TODO: check if nx is installed
    edges = nx.to_pandas_edgelist(G)
    check_column(edges, column_names=["source", "target"], present=True)
    return  make_index(edges)


def as_graph_edge_df(x):
    check_column(x, column_names=["source", "target"], present=True)
    return make_index(x)
      

def as_graph_node_edge(x, node_key='name'):
    """Prep and check that list of nodes and edges is a proper graph_node_edge format"""
    edges = x[0]
    nodes = x[1]
    
    #!TODO: add option to pass node_key
    # if node_key != 'name':
    #     col_pos = np.where(f'{node_key}' == nodes.columns)[0][0] 
    #     nodes = nodes.rename(columns={nodes.columns[col_pos]: "name"}, inplace=True)  

    if "name" not in nodes.columns:
        nodes = nodes.rename(columns={nodes.columns[0]: "name"}, inplace=True)

    check_column(edges, column_names=["source", "target"], present=True)
    edges = make_index(edges, nodes)
    return nodes, edges


def edges2dataframe(G: gt.Graph) -> pd.DataFrame:
    """Takes a Graph-tool graph object and returns edges data frame"""
    tmp_df = pd.DataFrame(list(G.edges()), columns=["source", "target"])
    tmp_df["source"] = tmp_df.source.astype(str)
    tmp_df["target"] = tmp_df.target.astype(str)
    return tmp_df


def nodes2dataframe(G: gt.Graph) -> pd.DataFrame:
    """Takes a Graph-tool graph object and returns nodes data frame"""
    prop_dfs = []
    for prop in G.vp:
        prop_dfs.append(pd.DataFrame({f"{prop}": list(G.vp[f"{prop}"])}))
    prop_dfs = pd.concat(prop_dfs, axis=1)
    
    if 'name' in prop_dfs.columns:
        prop_dfs = prop_dfs.rename(columns={"name":"label"})
    
    prop_dfs["name"] = prop_dfs.index
    prop_dfs["name"] = prop_dfs["name"].astype(str)
    
    return prop_dfs


def extract_nodes(edges: pd.DataFrame) -> pd.DataFrame:
    # we loose order here
    all_users = set(edges.source).union(set(edges.target))  
    return pd.DataFrame(all_users).rename(columns={0:"name"})

def make_index(
    edges: pd.DataFrame,
    nodes: Optional[pd.DataFrame] = None, 
) -> pd.DataFrame:
    """
    Creates edges data frame with `source` and `target` labelled by index in
    nodes data frame. If "name" not in nodes data frame, we use the first col
    as name. If no node data frame, we create one out of edge dataframe
    :param nodes: A `data frame` containing information about the nodes in the G
    :param edges: A `data Frame` containing information about the nodes in the G.
    :return: edge dataframe 
    """
    if nodes is None:
        nodes = extract_nodes(edges) # !TODO:does not keep order. Does that matter?
    elif "name" not in nodes.columns:
        nodes = nodes.rename(columns={nodes.columns[0]: "name"}, inplace=True)

    nodes['id'] = nodes.index
    cols = edges.columns

    edges = edges.merge(nodes, how="left", left_on="source", right_on="name")\
                 .drop("source", axis = 1)\
                 .rename(columns={"id":"source"}) 
    edges = edges.merge(nodes, how="left", left_on="target", right_on="name")\
                 .drop("target", axis = 1)\
                 .rename(columns={"id":"target"}) 
    return edges.drop(columns=["name_x", "name_y"])[cols]


def print_gt(G):
    print(nodes2dataframe(G))
    print(edges2dataframe(G))
    

# Centrality

def centrality_degree(G: gt.Graph, 
                      weights=None,
                      mode='out', 
                      loops=True, 
                      normalized=False) -> pd.Series:
    # expect nodes -> should we care?
    return pd.DataFrame({"deg": list(G.degree_property_map(f"{mode}"))})["deg"]

def node_coreness(G: gt.Graph) -> pd.Series:
    # expect nodes -> should we care?
    return pd.DataFrame({"kcore": list(gt.kcore_decomposition(G))})["kcore"]
    
# verbs

def filter_nodes(G: gt.Graph, criteria: str):
    """The goal here is to filter tidystyle."""
    nodes = nodes2dataframe(G)
    df_tmp = nodes.query(criteria)
    nodes["bp"] = np.where(nodes["name"].isin(df_tmp["name"]), True, False)
    augment_prop(G, nodes=nodes, prop_name="bp")
    G = gt.GraphView(G, vfilt=G.vp.bp)
    G = gt.Graph(G, prune=True)
    del G.properties[("v", "bp")]
    return G

def mutate():
    pass

def mutate_nodes(
    G: gt.Graph,
    column_name: str,
    func: Callable[[gt.Graph], pd.Series]) -> gt.Graph:
    """
    Similar to dplyr::mutate, this function creates a new
    column here based on a function.
    """
    return augment_prop(G, nodes=pd.DataFrame({f"{column_name}": func}))
    
def left_join_nested_blocks(nodes, *args):
    """Take list levels from `NestedBlock` object, and left join them recursively"""
    argCount = len(args)
    
    # build df level 0
    df_lvl_0 = pd.DataFrame({"gr_level0": args[0]})
                    
    if argCount >= 2:
        # build df level 1
        tmp = _merge_level_below(df_lvl_0, args[1])
        
    if argCount >= 3:
        # build df level 2
        tmp = _merge_level_below(tmp, args[2])
        
    if argCount >= 4:
        # build df level 3
        tmp = _merge_level_below(tmp, args[3])

    if argCount >= 5:
        # build df level 4
        tmp = _merge_level_below(tmp, args[4])

    if argCount >= 6:
        # build df level 5
        tmp = _merge_level_below(tmp, args[5])
    
    if argCount >= 7:
        # build df level 6
        tmp = _merge_level_below(tmp, args[6])

    #associate names
    if 'name' in nodes.columns:
        tmp["name"] = nodes["name"]

    else:
        print("No name in nodes dataframe")

    return tmp


def _merge_level_below(df_lvl_below, dat):
    lvl_below = list(df_lvl_below.columns)[-1]
    lvl_below = int(re.sub("gr_level", "", lvl_below))
    
    df_lvl_above = pd.DataFrame({f"gr_level{lvl_below+1}": dat})
    
    # index of df level i == block level i-1/agents
    df_lvl_above[f'gr_level{lvl_below}'] = df_lvl_above.index
    
    # We left join level i-1 of df_lvl_i onto level i-1 of df_lvl_be
    return pd.merge(df_lvl_below, df_lvl_above, 
                    left_on = f"gr_level{lvl_below}", 
                    right_on = f"gr_level{lvl_below}", 
                    how = "left")

