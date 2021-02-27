import re
from tidygraphtool.context import expect_nodes
from tidygraphtool.as_data_frame import as_data_frame
import graph_tool.all as gt
import pandas as pd
from typing import Callable, List
import numpy as np

from .augment import augment_prop
from .as_data_frame import as_data_frame
from .utils import check_column
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


def left_join(G: gt.Graph, 
              y: pd.DataFrame, 
              on: str = None
) -> gt.Graph:
    """Left join dataframe on corresponding nodes or edges in graph.
    
    IMPORTANT
    =========
    left and right key must be of the name. 
    If necessary, rename columns in y beforehand.

    """
    expect_nodes(G)
    df = NodeDataFrame(as_data_frame(G))
    df = df.merge(y, how="left", on=on)
    colnames = df.columns
    [augment_prop(G, x=df, prop_name=c) for c in colnames]
    return G


def add_column(
    G: gt.Graph,
    column_name: str,
    func: Callable[[gt.Graph], pd.Series]
) -> gt.Graph:
    """
    Creates a new column here based on a function. Behave like mutate in dplyr.
    """
    G = G.copy()
    df = func.rename(f"{column_name}")

    return augment_prop(G, df, prop_name=column_name)


def add_columns(
    G: gt.Graph,
    func: Callable[[gt.Graph], pd.Series],
) -> gt.Graph:
    """
    Creates a new column here based on a function. Behave like mutate in dplyr.
    """
    G = G.copy()
    df = func

    [augment_prop(G, df, f"{c}") for c in df.columns]
    
    return G


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


def rename(G: gt.Graph, old_column_name: str, new_column_name: str) -> gt.Graph:
    """Rename nodes or edges property."""
    G = G.copy()
    df = as_data_frame(G)
    check_column(df, [old_column_name])
    df = df.rename(columns={old_column_name: new_column_name})
    
    if G.gp.active == 'nodes':
        del G.vp[f"{old_column_name}"]
    else:
        del G.vp[f"{old_column_name}"]

    augment_prop(G, df, new_column_name)
    return G


def unnest_state(state):

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
