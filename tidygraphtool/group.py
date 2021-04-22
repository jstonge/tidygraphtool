"""Calculate communities"""

from typing import Union
import graph_tool.all as gt
import numpy as np
import pandas as pd

from .nodedataframe import NodeDataFrame
from .state import state_unnest


def group_sbm(G: gt.Graph, deg_corr: bool = True) -> NodeDataFrame:
  """Run degree-corrected Stochastic Block Model.

  See the `stochastic block model` section in graph-tool HOWTO. 

  :return: Node dataframe with group membership, model entropy, and state object.
  """
  state = gt.minimize_blockmodel_dl(G, deg_corr=deg_corr)
  ent = state.entropy()
  blocks = list(state.get_blocks())
  return NodeDataFrame({"group": [blocks], "ent": ent, "state_obj": state})


def group_sbm_covar(G: gt.Graph, 
                    edge_covar: gt.EdgePropertyMap, 
                    family: str,
                    deg_corr: bool = True) -> NodeDataFrame:
  """Run Stochastic Block Model with weight models.
  
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


def group_sbm_marginal(G: gt.Graph) -> NodeDataFrame:
  """Model averaging on SBM by sampling partitions of the posterior.

  EXAMPLE
  =======
  g = gt.collection.data["lesmis"]
  group_marginals = group_sbm_marginal(g)
  state = group_marginals.state_obj[0]
  pv = group_marginals.marginals[0]
  state.draw(pos=g.vp.pos, vertex_shape="pie", vertex_pie_fractions=pv)

  See `inferring-the-best-partition` section in graph-tool HOWTO.

  :return: Node dataframe with list-columns for each level, 
           model entropy, state object, and the marginals
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


def group_measured(G: gt.Graph, 
                   n: gt.EdgePropertyMap, 
                   x: gt.EdgePropertyMap,
                   n_default: int = 1,
                   x_default: int = 0):
  """Model averaging on SBM by sampling partitions of the posterior.

  EXAMPLE
  =======
  g = gt.collection.data["lesmis"]
  group_marginals = group_sbm_marginal(g)
  state = group_marginals.state_obj[0]
  pv = group_marginals.marginals[0]
  state.draw(pos=g.vp.pos, vertex_shape="pie", vertex_pie_fractions=pv)

  See `Measured networks` section in graph-tool HOWTO.

  :return: Node dataframe with list-columns for each level, 
           model entropy, state object, and the marginals
  """
  state = gt.MeasuredBlockState(G, n=n, n_default=n_default, x=x, x_default=x_default)

  gt.mcmc_equilibrate(state, wait=100, mcmc_args=dict(niter=10))
  
  u = None              # marginal posterior edge probabilities
  bs = []               # partitions
  cs = []               # average local clustering coefficient
  def collect_partitions(s):
    nonlocal bs
    bs.append(s.b.a.copy())

  def collect_marginals(s):
    nonlocal u, bs, cs
    u = s.collect_marginal(u)
    bstate = s.get_block_state()
    bs.append(bstate.levels[0].b.a.copy())
    cs.append(gt.local_clustering(s.get_graph()).fa.mean())

  gt.mcmc_equilibrate(state, force_niter=10000, mcmc_args=dict(niter=10),
                      callback=collect_marginals)

  ent = state.entropy()
  #!TODO: check what kind of object that bstate returns. probably a nested one.
  bstate = state.get_block_state()

  return NodeDataFrame({"edge_prob_ob": [u], "ent": ent, "state_obj": state})


def group_sbm_assortative(G: gt.Graph) -> NodeDataFrame:
  """Run planted partition model.

  See `Assortative community structure` section in graph-tool HOWTO.

  :return: Node dataframe with list-columns for each level, 
           model entropy, state object, and the marginals
  """
  state = gt.PPBlockState(G)

  # 1,000 sweeps of the MCMC with zero temperature
  state.multiflip_mcmc_sweep(beta=np.inf, niter=1000)

  ent = state.entropy()
  blocks = list(state.get_blocks())

  return NodeDataFrame({"group": [blocks], "ent": ent, "state_obj": state})


def group_hsbm(G: gt.Graph, 
               deg_corr: bool = True, 
               merge_split: bool = True) -> NodeDataFrame:
  """Run hierarchical degree-corrected Stochastic Block Model.

  See the `nested stochastic block model` section in graph-tool HOWTO.
  
  :return: node dataframe with a column for each level, model entropy, and state object.
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
  
  See the `Edge weights and covariates` section in graph-tool HOWTO.

  :return: node dataframe with a column for each level, model entropy, and state object.
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


def group_hsbm_layered(G: gt.Graph, 
                       edge_covar, 
                       merge_split: bool = True) -> NodeDataFrame:
  """Run hierarchical Stochastic Block Model with weight models.
  
  EXAMPLE
  =======
  g = gt.collection.ns["new_guinea_tribes"]
  group_layered = group_hsbm_layered(g, g.ep.weight, merge_split=False)
  state = group_layered.state_obj[0]
  state.draw(edge_color=g.ep.weight, edge_gradient=[],
             ecmap=(plt.cm.coolwarm_r, .6), edge_pen_width=5)

  See `Layered networks` section in graph-tool HOWTO.

  :return: node dataframe with a column for each level, model entropy, and state object.
  """

  state = gt.minimize_nested_blockmodel_dl(
    G, 
    state_args=dict(ec=G.ep.weight, layers=True),
    layers=True
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
  """Model averaging on SBM by sampling partitions of the posterior.

  EXAMPLE
  =======
  g = gt.collection.data["lesmis"]
  group_marginals = group_hsbm_marginal(g)
  state = group_marginals.state_obj[0]
  pv = group_marginals.marginals[0]
  state.draw(pos=g.vp.pos, vertex_shape="pie", vertex_pie_fractions=pv)

  See `Hierarchical partitions` section in graph-tool HOWTO.

  :return: Node dataframe with list-columns for each level, 
           model entropy, state object, and the marginals
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


def group_latent_poisson(G: gt.Graph):
  """Model averaging on SBM by sampling partitions of the posterior.

  EXAMPLE
  =======
  g = gt.collection.data["polbooks"]
  group_poisson = group_latent_poisson(g)
  bstate = group_poisson.state_obj[0]
  u = group_poisson.u[0]
  pv = group_poisson.marginals[0]
  ew = group_poisson.ew[0]
  bstate.draw(pos=u.own_property(g.vp.pos), vertex_shape="pie", vertex_pie_fractions=pv,
              edge_pen_width=gt.prop_to_size(ew, .1, 8, power=1), edge_gradient=None)

  See `Latent Poisson multigraphs¶` section in graph-tool HOWTO.

  :return: Node dataframe with list-columns for each level, 
           model entropy, state object, and the marginals
  """
  state = gt.LatentMultigraphBlockState(G)

  gt.mcmc_equilibrate(state, wait=100, mcmc_args=dict(niter=10))
  
  u = None   # marginal posterior multigraph
  bs = []    # partitions
  def collect_marginals(s):
    nonlocal bs, u
    u = s.collect_marginal_multigraph(u)
    bstate = state.get_block_state()
    bs.append(bstate.levels[0].b.a.copy())


  gt.mcmc_equilibrate(state, force_niter=10000, mcmc_args=dict(niter=10),
                      callback=collect_marginals)

  # compute average multiplicities
  ew = u.new_ep("double")
  w = u.ep.w
  wcount = u.ep.wcount
  for e in u.edges():
      ew[e] = (wcount[e].a * w[e].a).sum() / wcount[e].a.sum()

  bstate = state.get_block_state()
  bstate = bstate.levels[0].copy(g=u)

  # Disambiguate partitions and obtain marginals
  pmode = gt.PartitionModeState(bs, converge=True)
  pv = pmode.get_marginal(u)

  blocks = list(bstate.get_blocks())
  ent = bstate.entropy()
  return NodeDataFrame({"group": [blocks], "marginals": [pv], "ent": [ent], 
                       "ew": [ew], "u":[u], "state_obj": bstate})


# minimize_blockmodel_dl()        --> BlockState 
# minimize_nested_blockmodel_dl() --> NestedBlockState

# state.mcmc_sweep                -->
#   - single node is moved at a time
# state.multiflip_mcmc_sweep      -->
#   - merge and splits
# gt.mcmc_anneal(state, ...)      -->
# gt.mcmc_equilibrate()           -->
# gt.PPBlockState                 -->
# MeasuredBlockState
# MixedMeasuredBlockState
# UncertainBlockState
# LatentMultigraphBlockState
# EpidemicsBlockState