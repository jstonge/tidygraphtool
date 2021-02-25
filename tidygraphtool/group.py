import graph_tool.all as gt

from .nodedataframe import NodeDataFrame, NodeSeries
from .verbs import unnest_state, _merge_level_below


def group_sbm(G: gt.Graph, deg_corr: bool = True) -> NodeDataFrame:
  """Run degree-corrected Stochastic Block Model"""
  state = gt.minimize_blockmodel_dl(G, deg_corr=deg_corr)
  return NodeDataFrame({"group": list(state.get_blocks())})["group"]


def group_hsbm(G: gt.Graph, deg_corr: bool = True) -> NodeDataFrame:
  """Run hierarchical degree-corrected Stochastic Block Model
  
  :return: node dataframe with a column for each level
  """
  state = gt.minimize_nested_blockmodel_dl(G, deg_corr=deg_corr)
  group_all_levels = unnest_state(state)
  return NodeDataFrame(group_all_levels)
