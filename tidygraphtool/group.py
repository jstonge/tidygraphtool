"""Calculate communities"""

from typing import Union
import graph_tool.all as gt
import numpy as np
import pandas as pd

from .nodedataframe import NodeDataFrame
from .state import state_unnest


def group_sbm(G: gt.Graph, deg_corr: bool = True) -> NodeDataFrame:
  """Run degree-corrected Stochastic Block Model"""
  state = gt.minimize_blockmodel_dl(G, deg_corr=deg_corr)
  ent = state.entropy()
  blocks = list(state.get_blocks())
  return NodeDataFrame({"group": [blocks], "ent": ent, "state_obj": state})


def group_sbm_covar(G: gt.Graph, 
                    edge_covar: gt.EdgePropertyMap, 
                    family: str,
                    deg_corr: bool = True) -> NodeDataFrame:
  """Run Stochastic Block Model with weight models
  
  See `edge-weights-and-covariates` section in graph-tool HOWTO.

  :return: Node dataframe with list-columns for each level, and the model entropy
  """
  if family in ['real-exponential', "real-normal"]:
    assert all([type(v) is float for v in list(G.ep[f"{edge_covar}"])])
  elif family in ['discrete-geometric', 'discrete-binomial', 'discrete-poisson']:
    assert all([type(v) is int for v in list(G.ep[f"{edge_covar}"])])
  
  state = gt.minimize_blockmodel_dl(
    G,
    state_args=dict(recs=[G.ep[f"{edge_covar}"]], rec_types=[f"{family}"]), 
    deg_corr=deg_corr
  )

  ent = state.entropy()
  blocks = list(state.get_blocks())
  
  return NodeDataFrame({"group": [blocks], "ent": ent, "state_obj": state})


def group_sbm_marginal(G):
  """Model averaging on SBM by sampling partitions of the posterior

  EXAMPLE
  =======
  g = gt.collection.data["lesmis"]
  group_marginals = group_sbm_marginal(g)
  state = group_marginals.state_obj[0]
  pv = group_marginals.marginals[0]
  state.draw(pos=g.vp.pos, vertex_shape="pie", vertex_pie_fractions=pv)

  See `inferring-the-best-partition` section in graph-tool HOWTO.

  :return: Node dataframe with list-columns for each level, 
           the model entropy, and the marginals
  """
  state = gt.BlockState(G)

  # Equilibrate markov chain
  gt.mcmc_equilibrate(state, wait=1000, mcmc_args=dict(niter=10))
  
  bs = []
  def collect_partitions(s):
    nonlocal bs
    bs.append(s.b.a.copy())

  # Collect partitions for exactly 100K sweeps, at intervals of 10 sweeps
  gt.mcmc_equilibrate(state, force_niter=10000, mcmc_args=dict(niter=10),
                      callback=collect_partitions)

  # Disambiguate partitions and obtain marginals
  pmode = gt.PartitionModeState(bs, converge=True)
  pv = pmode.get_marginal(G)
  ent = state.entropy()
  blocks = list(state.get_blocks())

  return NodeDataFrame({"group": [blocks], "marginals": [pv], 
                        "ent": ent, "state_obj": state})


def group_hsbm(G: gt.Graph, 
               deg_corr: bool = True, 
               merge_split: bool = True) -> NodeDataFrame:
  """Run hierarchical degree-corrected Stochastic Block Model
  
  :return: node dataframe with a column for each level
  """
  state = gt.minimize_nested_blockmodel_dl(G, deg_corr=deg_corr)

  if merge_split == True:
    state = state.copy(bs=state.get_bs() + [np.zeros(1)] * 4, sampling=True)

    for i in range(100):
      ret = state.multiflip_mcmc_sweep(niter=10, beta=np.inf)

  group_all_levels = state_unnest(state)
  group_all_levels = pd.concat(
    [NodeDataFrame({f"{col}": [group_all_levels[f"{col}"]]}) 
     for col in group_all_levels.columns], axis=1
  )
  return pd.concat(
    [group_all_levels, NodeDataFrame({"ent": [ent], "state_obj": state})], 
     axis=1)


def group_hsbm_covar(G: gt.Graph, 
                     edge_covar, 
                     family: str,
                     deg_corr: bool = True, 
                     merge_split: bool = True) -> NodeDataFrame:
  """Run hierarchical Stochastic Block Model with weight models
  
  :return: node dataframe with a column for each level
  """
  if family in ['real-exponential', "real-normal"]:
    assert all([type(v) is float for v in list(edge_covar)])
  elif family in ['discrete-geometric', 'discrete-binomial', 'discrete-poisson']:
    assert all([type(v) is int for v in list(edge_covar)])
  
  state = gt.minimize_nested_blockmodel_dl(
    G, 
    state_args=dict(recs=[edge_covar], rec_types=[f"{family}"])
  )

  if merge_split == True:
    state = state.copy(bs=state.get_bs() + [np.zeros(1)] * 4, sampling=True)

    for i in range(100):
      ret = state.multiflip_mcmc_sweep(niter=10, beta=np.inf)
  
  ent = state.entropy()
  group_all_levels = state_unnest(state)
  group_all_levels = pd.concat(
    [NodeDataFrame({f"{col}": [group_all_levels[f"{col}"]]}) 
     for col in group_all_levels.columns], axis=1
  )
  return pd.concat([group_all_levels, NodeDataFrame({"ent": [ent], "state_obj": state})], axis=1)


def group_hsbm_marginal(G):
  """Model averaging on SBM by sampling partitions of the posterior

  EXAMPLE
  =======
  g = gt.collection.data["lesmis"]
  group_marginals = group_hsbm_marginal(g)
  state = group_marginals.state_obj[0]
  pv = group_marginals.marginals[0]
  state.draw(pos=g.vp.pos, vertex_shape="pie", vertex_pie_fractions=pv)

  See `Hierarchical partitions` section in graph-tool HOWTO.

  :return: Node dataframe with list-columns for each level, 
           the model entropy, and the marginals
  """
  state = gt.NestedBlockState(G)

  # Equilibrate markov chain
  gt.mcmc_equilibrate(state, wait=1000, mcmc_args=dict(niter=10))
  
  bs = []
  def collect_partitions(s):
    nonlocal bs
    bs.append(s.get_bs())

  # Collect partitions for exactly 100K sweeps, at intervals of 10 sweeps
  gt.mcmc_equilibrate(state, force_niter=10000, mcmc_args=dict(niter=10),
                      callback=collect_partitions)

  # Disambiguate partitions and obtain marginals
  pmode = gt.PartitionModeState(bs, nested=True, converge=True)
  pv = pmode.get_marginal(G)

  # Get consensus estimate
  bs = pmode.get_max_nested()

  state = state.copy(bs=bs)

  group_all_levels = state_unnest(state)
  group_all_levels = pd.concat(
    [NodeDataFrame({f"{col}": [group_all_levels[f"{col}"]]}) 
     for col in group_all_levels.columns], axis=1
  )

  ent = state.entropy()
  return pd.concat(
    [group_all_levels, NodeDataFrame({"marginals": [pv], "ent": [ent], "state_obj": state})], 
     axis=1)


# minimize_blockmodel_dl()        --> BlockState 
# minimize_nested_blockmodel_dl() --> NestedBlockState

# state.mcmc_sweep                -->
#   - single node is moved at a time
# state.multiflip_mcmc_sweep      -->
#   - merge and splits
# gt.mcmc_anneal(state, ...)      -->
# gt. mcmc_equilibrate()          -->