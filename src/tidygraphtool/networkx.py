"""Graphtool to Networkx graph"""

from typing import Union, Optional

import graph_tool.all as gt
import networkx as nx

from .context import activate
from .gt_graph import as_gt_graph, _indexify_edges
from .as_data_frame import as_data_frame
from .utils import check_column

@as_gt_graph.register(nx.classes.graph.Graph)
def _networkx_graph(x, directed=True):
    """Convert networkx graph, and returns a graph-tool graph."""
    edges = networkx_to_df(x)
    g = gt.Graph(directed=directed)
    activate(g, "nodes")
    g.add_edge_list(edges[["source","target"]].to_numpy())
    return g


def networkx_to_df(G):
    # !TODO: check if nx is installed
    edges = nx.to_pandas_edgelist(G)
    check_column(edges, column_names=["source", "target"], present=True)
    return  _indexify_edges(edges)


def as_networkx(G: gt.Graph,
                directed: bool = False,
                prop_name: Optional[str] = None) -> Union[nx.Graph, nx.DiGraph]:
    activate(G, "nodes")
    nodes = as_data_frame(G)
    activate(G, "edges")
    edges = as_data_frame(G)
    new_g = nx.from_pandas_edgelist(edges)
    if directed is True:
        new_g = new_g.to_directed()

    attr = [(i, dict(group=gr)) for i,gr in enumerate(nodes[f"{prop_name}"])]
    new_g.add_nodes_from(attr)
    return new_g
