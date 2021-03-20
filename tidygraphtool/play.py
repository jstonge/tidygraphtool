"""Popular networks simulation"""

from math import ceil

import numpy as np
import graph_tool.all as gt
from pipey import Pipeable

from .context import activate

@Pipeable(try_normal_call_first=True)
def play_sbm(n_k: int = 100,
             k: int = 3,
             p_rs: float = 0.001,
             directed: bool = True,
             self_loops: bool = False) -> gt.Graph:
             """Play SBM"""
             n_rs = ceil((p_rs * (n_k * k)) / (k - 1))
             n_rr = (1-p_rs) * (n_k * k)

             prob_mat = np.full((k,k), n_rs)
             np.fill_diagonal(prob_mat, n_rr)

             n = prob_mat.sum()
             membership = np.repeat(range(k), n/k)

             g = gt.generate_sbm(
                     b=membership, 
                     probs=prob_mat,
                     directed=directed)
             if self_loops == True:
                 gt.remove_self_loops(g)

             g = gt.extract_largest_component(g, prune=True, directed=False)

             vprop=g.new_vp("string")
             g.vp["name"] = vprop
             
             for i in range(len(list(g.vertices()))):
               vprop[i] = g.vertex(i)


             return activate(g, "nodes")

