from .nodedataframe import NodeDataFrame
from .edgedataframe import EdgeDataFrame
from .context import expect_nodes, expect_edges

import pandas as pd
import graph_tool.all as gt 



def centrality_betweenness(G: gt.Graph) -> pd.Series:
    expect_nodes(G)
    bet, _ = gt.betweenness(G)
    return NodeDataFrame({"bet": list(bet)})["bet"]


def centrality_betweenness(G: gt.Graph) -> pd.Series:
    expect_nodes(G)
    bet, _ = gt.betweenness(G)
    return NodeDataFrame({"bet": list(bet)})["bet"]


def centrality_edge_betweenness(G: gt.Graph) -> pd.Series:
    expect_edges(G)
    _, bet = gt.betweenness(G)
    return EdgeDataFrame({"bet": list(bet)})["bet"]


def centrality_closeness(G: gt.Graph) -> pd.Series:
    expect_nodes(G)
    return NodeDataFrame({"closeness": list(gt.closeness(G))})["closeness"]


def centrality_degree(G: gt.Graph,
                      weights=None,
                      mode='out',
                      loops=True,
                      normalized=False) -> pd.Series:
    expect_nodes(G)
    return NodeDataFrame({"deg": list(G.degree_property_map(f"{mode}"))})["deg"]


def centrality_eigen(G: gt.Graph) -> pd.Series:
    expect_nodes(G)
    _, eigen = gt.eigenvector(G)
    return NodeDataFrame({"eigen": list(eigen)})["eigen"]


def centrality_pagerank(G: gt.Graph) -> pd.Series:
    expect_nodes(G)
    return NodeDataFrame({"pr": list(gt.pagerank(G))})["pr"]