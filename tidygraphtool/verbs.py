import re
from tidygraphtool.as_data_frame import as_data_frame
import graph_tool.all as gt
import pandas as pd
from typing import Callable
import numpy as np

from .augment import augment_prop, _augment_prop_nodes, _augment_prop_edges
from .as_data_frame import as_data_frame
from .nodedataframe import NodeDataFrame, NodeSeries
from .edgedataframe import EdgeDataFrame, EdgeSeries


def filter_on(G: gt.Graph, criteria: str) -> gt.Graph:
    """Filter tidystyle on a particular criteria.

    Name and method is heavily inspired from pyjanitor.
    """
    df = as_data_frame(G)
    #!TODO: check_column(nodes, ...)
    df_tmp = df.query(criteria)
    df["bp"] = np.where(df["name"].isin(df_tmp["name"]), True, False)
    G = augment_prop(G, df, prop_name="bp")
    G = gt.GraphView(G, vfilt=G.vp.bp)
    G = gt.Graph(G, prune=True)
    del G.properties[("v", "bp")]
    return G


def mutate(
    G: gt.Graph,
    column_name: str,
    func: Callable[[gt.Graph], pd.Series]
) -> gt.Graph:
    """
    Creates a new column here based on a function.
    """
    G = G.copy()
    x = func.rename(f"{column_name}")

    if isinstance(x, (NodeDataFrame, NodeSeries)):
        return _augment_prop_nodes(G, nodes=x, prop_name=column_name)
    else:
        return _augment_prop_edges(G, edges=x, prop_name=column_name)


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


def unnest_state(state):
    levels = state.get_levels()
    list_r_level = [list(r for r in levels[i].get_blocks()) for i in range(len(levels))]
    com_all_lvl = pd.DataFrame({"hsbm_level0": list_r_level[0]})
    
    i=1
    while (i < len(list_r_level)):
        print(i)
        com_all_lvl = _merge_level_below(com_all_lvl, list_r_level[i])
        i += 1
    
    return com_all_lvl
