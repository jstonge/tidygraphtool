"""Helper functions to work with `state` object"""

import re
import pandas as pd
import numpy as np
import graph_tool.all as gt
from pipey import Pipeable


def state_marginals(G: gt.Graph, state) -> pd.DataFrame:
  """Get marginals of state object
  
  EXAMPLE
  =======
  import seaborn as sns
  import matplotlib.pyplot as plt

  g = gt.collection.data["lesmis"]
  group_marginals = group_sbm_marginal(g)
  d = get_marginal_probability(g, state)

  sns.set_style("whitegrid")
  sns.barplot(x="x", y="h", dodge = True, linewidth=0.5,
              data=d, palette="deep", edgecolor=".2")
  plt.xlabel("B")
  plt.ylabel("P(B|A)");
  
  See `inferring-the-best-partition` section in graph-tool HOWTO.

  :return: Dataframe with marginal probabilities of blocks
  """
  h = np.zeros(G.num_vertices() + 1)

  def collect_num_groups(s):
      B = s.get_nonempty_B()
      h[B] += 1

  gt.mcmc_equilibrate(state, force_niter=10000, 
                      mcmc_args=dict(niter=10),
                      callback=collect_num_groups)

  return pd.DataFrame({"h":h})\
           .query("h != 0")\
           .assign(h = lambda x: x.h / x.h.sum(),
                   B = lambda x: x.index)


def state_nested_marginals(G: gt.Graph, state) -> pd.DataFrame:
  """Get marginals of NestedBlockState object
  
  import seaborn as sns
  import matplotlib.pyplot as plt

  EXAMPLE
  =======
  g = gt.collection.data["lesmis"]
  group_marginals = group_sbm_marginal(g)
  d = get_marginal_probability(g, state)
  d_long = d.apply(pd.Series.explode)

  levels = d_long.levels.unique()[:-1] #trivially 1
  fig, ax = plt.subplots(1,len(levels),figsize=(25,5))

  for i, level in enumerate(levels):
    sns.barplot(x="B", y="h", data=d_long[d_long.levels == level],  ax=ax[i],
                dodge = True, linewidth=0.5,
                palette="deep", edgecolor=".2")
    ax[i].set_xlabel(f"$B_{level}$")
    ax[i].set_ylabel(f"$P(B_{level}|A)$")

  sns.set_style("whitegrid")
  sns.barplot(x="B", y="h", dodge = True, linewidth=0.5,
              data=d_long, palette="deep", edgecolor=".2")
  plt.xlabel("B")
  plt.ylabel("P(A|B)");
  
  See `Hierarchical partitions` section in graph-tool HOWTO.

  :return: Dataframe with marginal probabilities of blocks
  """
  h = [np.zeros(G.num_vertices() + 1) for s in state.get_levels()]

  def collect_num_groups(s):
      for l, sl in enumerate(s.get_levels()):
        B = sl.get_nonempty_B()
        h[l][B] += 1

  # Collect marginal for 100K sweeps
  gt.mcmc_equilibrate(state, force_niter=10000, mcmc_args=dict(niter=10),
                      callback=collect_num_groups)

  df_long = pd.DataFrame({"h":h}).explode("h")\
              .assign(levels = lambda x: x.index,
                      B = lambda x: x.groupby("levels").cumcount(+1))\
              .query("h != 0")\
              .assign(h = lambda x: x.h/x.h.sum())\
              .assign(h = lambda x: x.h.astype(float).round(decimals=3))
  
  nb_levels = len(df_long.levels.unique())

  return pd.DataFrame(
    {"h": [df_long.query(f"levels == {lvl}").h.tolist() for lvl in range(nb_levels)],
     "B": [df_long.query(f"levels == {lvl}").B.tolist() for lvl in range(nb_levels)]}
    ).assign(levels = lambda x: x.index)


def _merge_level_below(df_lvl_below, dat):
    lvl_below = list(df_lvl_below.columns)[-1]
    lvl_below = int(re.sub("hsbm_level", "", lvl_below))
    
    df_lvl_above = pd.DataFrame({f"hsbm_level{lvl_below+1}": dat})
    
    # index of df level i == block level i-1/agents
    df_lvl_above[f'hsbm_level{lvl_below}'] = df_lvl_above.index
    
    # We left join level i-1 of df_lvl_i onto level i-1 of df_lvl_be
    return pd.merge(df_lvl_below, df_lvl_above, 
                    left_on = f"hsbm_level{lvl_below}", 
                    right_on = f"hsbm_level{lvl_below}", 
                    how = "left")


@Pipeable(try_normal_call_first=True)
def state_unnest(state):
    """
    Unnest gt.BlockState object into a dataframe.
    
    The function will create one column by level.
    """
    if isinstance(state, gt.BlockState):
        raise ValueError("Not hierarchical block state")

    levels = state.get_levels()
    list_r_level = [list(r for r in levels[i].get_blocks()) for i in range(len(levels))]
    com_all_lvl = pd.DataFrame({"hsbm_level0": list_r_level[0]})
    
    i=1
    while (i < len(list_r_level)):
        com_all_lvl = _merge_level_below(com_all_lvl, list_r_level[i])
        i += 1
    
    return com_all_lvl
