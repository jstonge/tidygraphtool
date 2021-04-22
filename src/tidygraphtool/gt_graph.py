"""Main module to convert to get graph"""

from functools import singledispatch
from typing import Optional, Dict
from pipey import Pipeable

from .as_data_frame import as_data_frame
from .augment import augment_prop
from .utils import assert_nodes_edges_equal, assert_nodes_mergeable, check_column, guess_df_type, guess_dict_type, _extract_nodes
from .nodedataframe import NodeDataFrame, NodeSeries
from .edgedataframe import EdgeDataFrame, EdgeSeries
from .context import activate

import graph_tool.all as gt
import pandas as pd


def gt_graph(
    nodes: Optional[pd.DataFrame] = None,
    edges: Optional[pd.DataFrame] = None,
    directed: bool = True,
    node_key: str = "name"
) -> gt.Graph:
    """
    Creates a graph-tool object using nodes and edges dataframe.

    Example
    =======
    .. code-block:: python

        nodes = pd.DataFrame({"name": ["Bob", "Alice", "Joan", "Melvin"]}) 
        edges = pd.DataFrame({"source": ["Bob", "Bob", "Joan", "Alice"],
                              "target": ["Joan", "Melvin", "Alice", "Joan"]})
        gt_graph(nodes, edges)

    NOTE
    ====
    Params are adapted directly from thomasp85's tidygraph api in R 

    :param nodes: A `data.frame` containing information about the nodes in the graph.
        If `edges.target` and/or `edges.from` are characters then they will be
        matched to the column named according to `node_key` in nodes, if it exists.
        If not, they will be matched to the first column.
    :param edges:  A `data.frame` containing information about the edges in the
        graph. The terminal nodes of each edge must either be encoded in a `target` and
        `source` column, or in the two first columns, as integers. These integers refer to
        `nodes` index.
    :directed:  Should the constructed graph be directed (defaults to `TRUE`)
    :node_key: The name of the column in `nodes` that character represented
        `target` and `source` columns should be matched against. 
    :return: A `Graph_tool` object
    """
    return as_gt_graph({"nodes": nodes, "edges": edges}, 
                        directed=directed, 
                        node_key=node_key)


@singledispatch
def as_gt_graph(x):
    raise ValueError(f"{x} is not implemented")


@as_gt_graph.register(pd.DataFrame)
@as_gt_graph.register(EdgeDataFrame)
def _data_frame(x, directed=True, node_key='name') -> gt.Graph:
    """Convert dataframe, and returns a graph-tool graph."""
    if guess_df_type(x) == 'EdgeDataFrame':
        nodes = _extract_nodes(x)
        node_edge_df = {"nodes": nodes, "edges": x}
        nodes, edges = _as_graph_node_edge(node_edge_df, node_key=f'{node_key}')

        # Create Graph
        g = gt.Graph(directed=directed) >> activate("nodes")
        g.add_edge_list(edges[["source", "target"]].to_numpy())
        g = augment_prop(g, NodeDataFrame(nodes), prop_name=f"{node_key}")

        # # Add edge metadata
        g = activate(g, "edges")
        edgecols = edges.iloc[:, 2::].columns
        if len(edgecols) == 1:
            augment_prop(g, EdgeDataFrame(edges), prop_name=edgecols[0])
        elif len(edgecols) > 1:
            [augment_prop(g, EdgeDataFrame(edges), prop_name=c) for c in edgecols]

        return activate(g, "nodes") # nodes by default
    else:
        raise ValueError("as_gt_graph for nodes not implemented yet")


@as_gt_graph.register(dict)
def _dict(x, directed=True, node_key='name') -> gt.Graph:
    """Convert dict containing nodes and edges, and returns a graph-tool graph."""
    if guess_dict_type(x) == 'node_edge':

        # Check if one of the value is None
        if any([v is None for v in x.values()]):
            x = [v for v in x.values() if v is not None][0]
            if guess_df_type(x) == 'EdgeDataFrame':
                nodes = _extract_nodes(x)
                x = {"nodes": nodes, "edges": x}
                # edges = _as_graph_edge_df(x)

        nodes, edges = _as_graph_node_edge(x, node_key=node_key)

        # Create Graph
        g = gt.Graph(directed=directed) >> activate("nodes")
        g.add_edge_list(edges[["source", "target"]].to_numpy())

        # Add node metadata
        nodecols = nodes.columns
        if len(nodecols) == 1:
            augment_prop(g,  NodeDataFrame(nodes), prop_name=nodecols[0])
        elif len(nodecols) > 1:
            [augment_prop(g, NodeDataFrame(nodes), prop_name=c) for c in nodecols]

        # Add edge metadata
        g = activate(g, "edges")
        edgecols = edges.iloc[:, 2::].columns
        if len(edgecols) == 1:
            augment_prop(g,  x=EdgeDataFrame(edges), prop_name=edgecols[0])
        elif len(edgecols) > 1:
            [augment_prop(g, x=EdgeDataFrame(edges), prop_name=c) for c in edgecols]

        return activate(g, "nodes")
    else:
        raise ValueError("Other types not implemented yet")


def _as_graph_node_edge(x: Dict[pd.DataFrame, pd.DataFrame], node_key: str = 'name'):
    """Prep and check that list of nodes and edges is a proper graph_node_edge format"""
    nodes = [v[1] for v in x.items() if v[0] in ['nodes', 'vertices']][0]
    edges = [v[1] for v in x.items() if v[0] in ['edges', 'links']][0]

    assert_nodes_edges_equal(nodes, edges, node_key), "Is `name` the node key?"

    # Making sure that node_key is in front of everything and of type string
    nodes = nodes.reorder_columns([f'{node_key}'])
    nodes[f"{node_key}"] = nodes[f"{node_key}"].astype(str)

    edges.columns = edges.columns.str.lower()
    check_column(edges, column_names=["source", "target"], present=True)
    edges = _indexify_edges(edges=edges, nodes=nodes, node_key=node_key)
    return nodes, edges


def _indexify_edges(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    node_key: str
) -> pd.DataFrame:
    """
    Creates edges data frame with `source` and `target` labelled by index in
    nodes data frame. If "name" not in nodes data frame, we use the first col
    as name. If no node data frame, we create one out of edge dataframe

    :return: edge dataframe 
    """

    nodes = nodes.copy()

    # Prep nodes and edges to be merged
    nodes['index'] = nodes.index
    edges["source"] = edges.source.astype(str)
    edges["target"] = edges.target.astype(str)

    nodes = nodes.drop_duplicates(f'{node_key}')\
                 .assign(name=lambda x: x[f'{node_key}'].astype(str))

    # Keeping track of the original col order
    cols = edges.columns

    # Using node index to name source target in graph object
    assert_nodes_mergeable(nodes, edges, left_key="source", right_key=f"{node_key}")
    assert_nodes_mergeable(nodes, edges, left_key="target", right_key=f"{node_key}")

    edges = edges.merge(nodes, how="left", left_on="source", right_on=f"{node_key}")\
                 .drop("source", axis=1)\
                 .rename(columns={"index": "source"})
    edges = edges.merge(nodes, how="left", left_on="target", right_on=f"{node_key}")\
                 .drop("target", axis=1)\
                 .rename(columns={"index": "target"})

    return edges.drop(columns=[f"{node_key}_x", f"{node_key}_y"])[cols]

@Pipeable(try_normal_call_first=True)
def summary(G):
    if G.gp.active == 'nodes':
        print(as_data_frame(G))
        G = activate(G, "edges")
        print(as_data_frame(G))
        G = activate(G, "nodes")
    elif G.gp.active == 'edges':
        print(as_data_frame(G))
        G = activate(G, "nodes")
        print(as_data_frame(G))
        G = activate(G, "edges")
