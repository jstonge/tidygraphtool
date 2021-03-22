"""Verbs for network wranling"""

import re
import graph_tool.all as gt
import pandas as pd
from typing import Callable
import numpy as np

from pipey import Pipeable

from .augment import augment_prop
from .as_data_frame import as_data_frame
from .utils import check_column, check_args_order
from .context import expect_edges, expect_nodes
from .nodedataframe import NodeDataFrame
from .edgedataframe import EdgeDataFrame
from .node import node_largest_component


#!TODO: Not working with function that takes as extra args EdgePropertyMap.
#       The reason seems that we cannot properly unpack EdgePropertyMap value.
@Pipeable(try_normal_call_first=True)
def add_property(G, colname:str = None, *args, **kwargs) -> gt.Graph:
    """
    Creates a new column based on a function. 
    Behave similar to mutate in R's dplyr.

    USAGE
    =====
    `.. code-block:: python
    add_property(g, "degree", centrality_degree) >> summary()
    
    Supplementary arguments to the function can be provided via the ``kwargs``
    parameter::
    u = (
      g >>
        activate("nodes") >>
        add_property("degree_tot", centrality_degree, mode="total") >>
        filter_on("degree_tot >= 2") 
    )


    :param args: implicitly, we expect the following params. 
       :param G: A gt.Graph object you wish to add property.
       :param column_name: A string that provide the name of the property.
       :param func: A callable that calcuate the desired property.
    :param kwargs: named arguments that correspond to callable extra args.
    :returns: A gt.Graph object augmented with labelled property.
    """
    # assert len(args) == 3

    # G = args[0]
    G = G.copy()
    # colname = args[1]
    f = args[0]

    check_args_order(G, colname, f)
    if len(kwargs) == 0:
        df = f(G).rename(f"{colname}")
    else:
        df = f(G, **kwargs).rename(f"{colname}")

    return augment_prop(G, df, prop_name=colname)


#!TODO: Only works with group_hsbm, not generally, as we cannot pass colnames.
@Pipeable(try_normal_call_first=True)
def add_properties(G: gt.Graph, 
                   func: Callable[[gt.Graph], pd.Series]) -> gt.Graph:
    """
    Creates a new column here based on a function. Behave like mutate in dplyr.
    """
    G = G.copy()
    df = func

    [augment_prop(G, df, f"{c}") for c in df.columns]
    
    return G


@Pipeable(try_normal_call_first=True)
def filter_largest_component(G:gt.Graph) -> gt.Graph:
    """Extract largest component"""
    expect_nodes(G)
    G = add_property(G, "lc", node_largest_component)
    G = filter_on(G, "lc == 1")
    return G


@Pipeable(try_normal_call_first=True)
def filter_on(G: gt.Graph, criteria: str) -> gt.Graph:
    """Filter tidystyle on a particular criteria.
    
    edges = pd.DataFrame({"source":[1,4,3,2], "target":[2,3,2,1]})
    g = as_gt_graph(edges)
    filter_on(g, "source == 2")
    """
    if G.gp.active == "nodes":
        expect_nodes(G)
        df = NodeDataFrame(as_data_frame(G))
        #!TODO: check_column(nodes, ...)
        df_tmp = df.query(criteria)
        df["bp"] = np.where(df.iloc[:,0].isin(df_tmp.iloc[:,0]), True, False)
        G = augment_prop(G, df, prop_name="bp")
        G = gt.GraphView(G, vfilt=G.vp.bp)
        G = gt.Graph(G, prune=True)
        del G.properties[("v", "bp")]
        return G
    elif G.gp.active == "edges":
        expect_edges(G)
        df = EdgeDataFrame(as_data_frame(G))
        df_tmp = df.query(criteria)
        df["bp"] = np.where(df["source"].isin(df_tmp["source"]) & \
            df["target"].isin(df_tmp["target"]), True, False)
        G = augment_prop(G, df, prop_name="bp")
        G = gt.GraphView(G, efilt=G.ep.bp)
        G = gt.Graph(G, prune=True)
        del G.properties[("e", "bp")]
        return G
    else:
        raise ValueError("Context must be activated to nodes or edges")


@Pipeable(try_normal_call_first=True)
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


@Pipeable(try_normal_call_first=True)
def rename(G: gt.Graph, old_column_name: str, new_column_name: str) -> gt.Graph:
    """Rename nodes or edges property.
    
    USAGE
    =====
    
    edges = pd.DataFrame({"source":[1,4,3,2], "target":[2,3,2,1]})
    g = as_gt_graph(edges)
    filter_on(g, "source == 2")
    
    """
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


@Pipeable(try_normal_call_first=True)
def simplify_edges(G:gt.Graph, 
                   remove_directed: bool = True,
                   remove_parallel: bool = True, 
                   remove_loop: bool = True) -> gt.Graph:
    """Remove parallel and directed edges and self loops"""
    G = G.copy()
    if remove_parallel and remove_loop and remove_directed:
        G.set_directed(False)
        gt.remove_self_loops(G)
        gt.remove_parallel_edges(G)
    elif remove_parallel and remove_directed and not remove_loop:
        G.set_directed(False)
        gt.remove_parallel_edges(G)
    elif remove_parallel and not remove_directed and not remove_loop:
        gt.remove_parallel_edges(G)
    elif remove_loop and not remove_parallel and not remove_directed:
        gt.remove_self_loops(G)
    else:
        raise ValueError("Not implemented yet")
    return G
