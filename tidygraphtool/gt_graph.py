"""Main module."""
from functools import singledispatch
from typing import Optional, Dict

from .as_data_frame import as_data_frame
from .augment import augment_prop
from .utils import assert_nodes_edges_equal, check_column, guess_df_type, guess_dict_type, _extract_nodes
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

    :return: A `Graph_tool` object
    """
    return as_gt_graph({"nodes":nodes, "edges":edges}, 
                        directed=directed, 
                        node_key=node_key)


@singledispatch
def as_gt_graph(x, directed=True, node_key='name'):
    return x


@as_gt_graph.register(pd.DataFrame)
@as_gt_graph.register(EdgeDataFrame)
def _data_frame(x, directed=True) -> gt.Graph:
    """Convert networkx graph, and returns a graph-tool graph."""
    if guess_df_type(x) == 'EdgeDataFrame':
        nodes = _extract_nodes(x)
        node_edge_df = {"nodes":nodes, "edges":x}
        nodes, edges = _as_graph_node_edge(node_edge_df, node_key='name')

        # Create Graph
        g = gt.Graph(directed=directed)
        g.add_edge_list(edges[["source","target"]].to_numpy())
        augment_prop(g, x=NodeDataFrame(nodes), prop_name="name")
        
        # # Add edge metadata
        edgecols = edges.iloc[:, 2::].columns
        if len(edgecols) == 1:
            augment_prop(g, x=EdgeDataFrame(edges), prop_name=edgecols[0])
        elif len(edgecols > 1):
            [augment_prop(g, x=EdgeDataFrame(edges), prop_name=c) for c in edgecols]
        
        activate(g, "nodes")
        return g
    else:
        raise ValueError("as_gt_graph for nodes not implemented yet")


# def _as_graph_edge_df(x):
#     x.columns = x.columns.str.lower()
#     check_column(x, column_names=["source", "target"], present=True)
#     return _indexify_edges(edges=x)


@as_gt_graph.register(dict)
def _dict(x, directed=True, node_key='name') -> gt.Graph:
    """Convert dict containing nodes and edges, and returns a graph-tool graph."""
    if guess_dict_type(x) == 'node_edge':
        
        # Check if one of the value is None
        if any([v is None for v in x.values()]):
            x = [v for v in x.values() if v is not None][0]
            if guess_df_type(x) == 'EdgeDataFrame':
                nodes = _extract_nodes(x)
                x = {"nodes":nodes, "edges":x}
                # edges = _as_graph_edge_df(x)
        
        nodes, edges = _as_graph_node_edge(x, node_key=node_key)
        
        # Create Graph
        g = gt.Graph(directed=directed)
        activate(g, "nodes")
        g.add_edge_list(edges[["source","target"]].to_numpy())
        
        # Add node metadata
        nodecols = nodes.columns
        if len(nodecols) == 1:
            augment_prop(g,  x=NodeDataFrame(nodes), prop_name=nodecols[0])
        elif len(nodecols) > 1:
            [augment_prop(g, x=NodeDataFrame(nodes), prop_name=c) for c in nodecols]
        
        # Add edge metadata
        edgecols = edges.iloc[:, 2::].columns
        if len(edgecols) == 1:
            augment_prop(g,  x=EdgeDataFrame(edges), prop_name=edgecols[0])
        elif len(edgecols) > 1:
            [augment_prop(g, x=EdgeDataFrame(edges), prop_name=c) for c in edgecols]

        return g
    else:
        raise ValueError("Other types not implemented yet")


def _as_graph_node_edge(x: Dict[pd.DataFrame, pd.DataFrame], node_key: str ='name'):
    """Prep and check that list of nodes and edges is a proper graph_node_edge format"""
    nodes = [v[1] for v in x.items() if v[0] in ['nodes', 'vertices']][0]
    edges = [v[1] for v in x.items() if v[0] in ['edges', 'links']][0]
    
    assert_nodes_edges_equal(nodes, edges)
    if "name" not in nodes.columns:
        nodes = nodes.rename(columns={nodes.columns[0]: "name"})

    edges.columns = edges.columns.str.lower()
    check_column(edges, column_names=["source", "target"], present=True)
    edges = _indexify_edges(edges=edges, nodes=nodes)
    return nodes, edges


def _indexify_edges(
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
        nodes = _extract_nodes(edges) # does not keep order. Does that matter?
    elif "name" not in nodes.columns:
        nodes = nodes.rename(columns={nodes.columns[0]: "name"})
    
    nodes = nodes.copy()

    # Prep nodes and edges to be merged
    edges["source"] = edges.source.astype(str)
    edges["target"] = edges.target.astype(str)
    nodes['index']  = nodes.index
    
    nodes = nodes.drop_duplicates('name')\
                 .assign(name = lambda x: x.name.astype(str))

    # Keeping track of the original col order
    cols = edges.columns
    
    # Using node index to name source target in graph object
    edges = edges.merge(nodes, how="left", left_on="source", right_on="name")\
                 .drop("source", axis = 1)\
                 .rename(columns={"index":"source"}) 
    edges = edges.merge(nodes, how="left", left_on="target", right_on="name")\
                 .drop("target", axis = 1)\
                 .rename(columns={"index":"target"})

    return edges.drop(columns=["name_x", "name_y"])[cols]


def print_gt(G):
    if G.gp.active == 'nodes':
        print(as_data_frame(G))
        activate(G, "edges")
        print(as_data_frame(G))
        activate(G, "nodes")
    elif G.gp.active == 'edges':
        print(as_data_frame(G))
        activate(G, "nodes")
        print(as_data_frame(G))
        activate(G, "edges")
