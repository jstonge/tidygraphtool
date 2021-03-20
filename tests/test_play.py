
from tidygraphtool.centrality import centrality_degree, centrality_edge_betweenness
from tidygraphtool.verbs import add_property
import pytest
import pandas as pd

from tidygraphtool.play import play_sbm
from tidygraphtool.group import group_hsbm

def test_play_sbm():
  g = play_sbm(n_k=500, k=3, p_rs=0.001)
  df_group = group_hsbm(g)
  assert len(df_group.hsbm_level0.unique()) == 3