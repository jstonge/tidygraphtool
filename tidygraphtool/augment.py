from typing import Optional, Union
import re

import graph_tool.all as gt
import pandas as pd

from .edgedataframe import EdgeDataFrame, EdgeSeries
from .nodedataframe import NodeDataFrame, NodeSeries
from .utils import guess_df_type, check_column

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
    if guess_df_type(x) is 'NodeDataFrame': x=NodeDataFrame(x)
    if guess_df_type(x) is 'EdgeDataFrame': x=EdgeDataFrame(x)

    if isinstance(x, (NodeDataFrame, NodeSeries)):
        return _augment_prop_nodes(G, x, prop_name)
    else:
        return _augment_prop_edges(G, x, prop_name)

def _augment_prop_nodes(G: gt.Graph, 
                        nodes: Union[NodeDataFrame, NodeSeries], 
                        prop_name: Optional[str] = None) -> pd.DataFrame:

    nodes=NodeDataFrame(nodes)

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


def _augment_prop_edges(G: gt.Graph, 
                        edges: Union[EdgeDataFrame, EdgeSeries], 
                        prop_name: str) -> pd.DataFrame:

    edges = EdgeDataFrame(edges)

    if edges[f'{prop_name}'].isnull().values.any() == True:
        raise ValueError("There are NAs in the col")
    check_column(edges, column_names = [prop_name])

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
