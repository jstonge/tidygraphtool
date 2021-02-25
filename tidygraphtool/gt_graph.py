"""Main module."""
from collections import Counter
from functools import singledispatch
from typing import Optional, Union

from .as_data_frame import as_data_frame
from .augment import augment_prop, _augment_prop_edges, _augment_prop_nodes
from .utils import check_column, guess_df_type, guess_list_type
from .nodedataframe import NodeDataFrame, NodeSeries
from .edgedataframe import EdgeDataFrame, EdgeSeries
from .context import activate

import graph_tool.all as gt
import pandas as pd


def gt_graph(
    nodes: pd.DataFrame = None, 
    edges: pd.DataFrame = None, 
    directed: bool = True,
    node_key: str = "name"
) -> gt.Graph:
    """
    Creates a graph from scratch using nodes and edges dataframe.

    Example
    =======
    .. code-block:: python

        nodes = pd.DataFrame({"name": ["Bob", "Alice", "Joan", "Melvin"]}) 
        edges = pd.DataFrame({"source": ["Bob", "Bob", "Joan", "Alice"],
                              "target": ["Joan", "Melvin", "Alice", "Joan"]})
        gt_graph(nodes, edges)

    :return: A `Graph_tool` object
    """
    return as_gt_graph([nodes, edges], directed=directed, node_key=node_key)


@singledispatch
def as_gt_graph(x, directed=True, node_key='name'):
    return x


@as_gt_graph.register(pd.DataFrame)
@as_gt_graph.register(EdgeDataFrame)
def _data_frame(x: Union[pd.DataFrame, EdgeDataFrame], directed: bool=True) -> gt.Graph:
    """Convert networkx graph, and returns a graph-tool graph."""
    if guess_df_type(x) == 'EdgeDataFrame':
        edges = as_graph_edge_df(x)
        g = gt.Graph(directed=directed)
        activate(g, "nodes")
        g.add_edge_list(edges[["source","target"]].to_numpy())

        return g
    else:
        raise ValueError("as_gt_graph for nodes not implemented yet")


def as_graph_edge_df(x):
    check_column(x, column_names=["source", "target"], present=True)
    return _make_index(x)


@as_gt_graph.register(list)
def _list(x, directed=True, node_key='name'):
    """Convert list containing nodes and edges, and returns a graph-tool graph."""
    if guess_list_type(x) == 'node_edge':
        nodes, edges = as_graph_node_edge(x, node_key=node_key)
        g = gt.Graph(directed=directed)
        activate(g, "nodes")
        g.add_edge_list(edges[["source","target"]].to_numpy())

        # g = augment_prop(g, NodeDataFrame(nodes), prop_name="name")

        [augment_prop(g, x=NodeDataFrame(nodes), prop_name=_) for _ in nodes.columns]
        [augment_prop(g, x=EdgeDataFrame(edges), prop_name=_) for _ in edges.iloc[:, 2:-1].columns]

        return g
    else:
        raise ValueError("Other types not implemented yet")


def as_graph_node_edge(x, node_key='name'):
    """Prep and check that list of nodes and edges is a proper graph_node_edge format"""
    x0, x1 = [guess_df_type(_) for _ in x]
    if x0 == 'NodeDataFrame':
        nodes = x[0]
        edges = x[1]
    else:
        nodes = x[1]
        edges = x[0]
    
    #!TODO: add option to pass node_key
    # if node_key != 'name':
    #     col_pos = np.where(f'{node_key}' == nodes.columns)[0][0] 
    #     nodes = nodes.rename(columns={nodes.columns[col_pos]: "name"}, inplace=True)  

    if "name" not in nodes.columns:
        nodes = nodes.rename(columns={nodes.columns[0]: "name"}, inplace=True)

    check_column(edges, column_names=["source", "target"], present=True)
    edges = _make_index(edges, nodes)
    return nodes, edges


def _extract_nodes(edges: pd.DataFrame) -> pd.DataFrame:
    # we loose order here
    all_users = set(edges.source).union(set(edges.target))  
    return pd.DataFrame(all_users).rename(columns={0:"name"})


def _make_index(
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
        nodes = _extract_nodes(edges) # !TODO:does not keep order. Does that matter?
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
    if G.active == 'nodes':
        print(as_data_frame(G))
        activate(G, "edges")
        print(as_data_frame(G))
        activate(G, "nodes")
    elif G.active == 'edges':
        print(as_data_frame(G))
        activate(G, "nodes")
        print(as_data_frame(G))
        activate(G, "edges")
