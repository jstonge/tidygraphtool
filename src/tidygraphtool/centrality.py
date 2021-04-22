"""Calculate node and edge centrality"""

from graph_tool import VertexPropertyMap
import pandas as pd
import graph_tool.all as gt 
from typing import Optional

from .nodedataframe import NodeDataFrame
from .edgedataframe import EdgeDataFrame
from .as_data_frame import as_data_frame
from .context import expect_nodes, expect_edges


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
                      weight=None,
                      mode='out') -> pd.Series:
    expect_nodes(G)
    deg = G.degree_property_map(f"{mode}", weight=weight)
    return NodeDataFrame({"deg": list(deg)})["deg"]


def centrality_eigenvector(G: gt.Graph,
                           weight: Optional[float] = None,
                           vprop: Optional[gt.VertexPropertyMap] = None,
                           epsilon: float = 1e-06,
                           max_iter: int = None) -> pd.Series:
    """
    Calculate the eigenvector centrality of each vertex in the graph.

    Wrapper for [graph_tool.centrality.eigenvector()]. See ?gt.eigenvector

    Functional usage syntax with pipes:
    
    .. code-block:: python
    
        np.random.seed(42)
        import matplotlib.pyplot as plt
            
        g = (gt.collection.data["polblogs"] >> 
              activate("nodes") >>
              filter_largest_component())

        g = (g >> 
              activate("edges") >>
              augment_prop(w, prop_name="w") >>
              activate("nodes") >>
              add_property("x", centrality_eigenvector))

        gt.graph_draw(g, pos=g.vp["pos"], vertex_fill_color=g.vp.x,
                      vertex_size=gt.prop_to_size(g.vp.x, mi=5, ma=15),
                      vcmap=plt.cm.gist_heat,
                      vorder=g.vp.x)

    :returns: A Node Series containing the eigenvector values.
    """
    expect_nodes(G)
    _, eigen_vector = gt.eigenvector(G, weight=weight, vprop=vprop,
                                     epsilon=epsilon,
                                     max_iter=max_iter)
    return NodeDataFrame({"eigen": list(eigen_vector)})["eigen"]


def centrality_eigentrust(G: gt.Graph,
                          trust_map: Optional[gt.EdgePropertyMap] = None,
                          vprop: Optional[gt.VertexPropertyMap] = None,
                          norm: bool = False,
                          epsilon: float = 1e-06,
                          max_iter: int = 0,
                          ret_iter: bool =False) -> pd.Series:
    """
    Calculate the eigentrust centrality of each vertex in the graph.

    Wrapper for [graph_tool.centrality.eigentrust()]. See ?gt.eigentrust

    Functional usage syntax with pipes:
    
    .. code-block:: python
      
      np.random.seed(42)
      import matplotlib.pyplot as plt
    
      g = (
        gt.collection.data["polblogs"] >> 
          activate("nodes") >>
          filter_largest_component()
      )
      w = EdgeDataFrame({"w":np.random.random(len(list(g.edges()))) * 42})
      g = (
        g >> 
          activate("edges") >> 
          augment_prop(w, prop_name="w") >> 
          activate("nodes")
      )
      g = augment_prop(g, centrality_eigentrust(g, g.ep.w), prop_name="t")
      gt.graph_draw(
          g, pos=g.vp["pos"], vertex_fill_color=g.vp.t,
          vertex_size=gt.prop_to_size(g.vp.t, mi=5, ma=15),
          vcmap=plt.cm.gist_heat,
          vorder=g.vp.t
      )


    :param trust_map: Edge property map with the values of trust associated with each edge. A float bounded between [0,1].
    :returns: A Node Series containing the eigentrust values.
    """
    expect_nodes(G)
    eigen_trust = gt.eigentrust(G, trust_map=trust_map, vprop=vprop, norm=norm, 
                                 epsilon=epsilon, max_iter=max_iter, ret_iter=ret_iter)
    return NodeDataFrame({"t": list(eigen_trust)})["t"]


def centrality_pagerank(G: gt.Graph) -> pd.Series:
    """
    Calculate the eigentrust centrality of each vertex in the graph.

    Wrapper for [graph_tool.centrality.pagerank()]. See ?gt.pagerank

    Functional usage syntax with pipes:
    
    .. code-block:: python

        g = (gt.collection.data["polblogs"] >> 
              activate("nodes") >>
              filter_largest_component() >>
              add_property("pr", centrality_pagerank))

        gt.graph_draw(g, pos=g.vp["pos"], vertex_fill_color=g.vp.pr,
                      vertex_size=gt.prop_to_size(g.vp.pr, mi=5, ma=15),
                      vorder=g.vp.pr, vcmap=plt.cm.gist_heat)

    """
    expect_nodes(G)
    return NodeDataFrame({"pr": list(gt.pagerank(G))})["pr"]

