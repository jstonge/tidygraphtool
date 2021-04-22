"""Graph to dataframe functions"""

from tidygraphtool.nodedataframe import NodeDataFrame
from tidygraphtool.edgedataframe import EdgeDataFrame
from .context import expect_edges, expect_nodes
import pandas as pd
from pipey import Pipeable
import graph_tool.all as gt 

@Pipeable(try_normal_call_first=True)
def as_data_frame(G: gt.Graph) -> pd.DataFrame:
    if G.gp.active == 'nodes':
        return _nodes2dataframe(G)
    elif G.gp.active == 'edges':
        return _edges2dataframe(G)
    else:
        raise ValueError("Nodes or edges must be active")

def _edges2dataframe(G: gt.Graph) -> EdgeDataFrame:
    """Takes a Graph-tool graph object and returns edges data frame.
    
    Edges will always correspond to node index.
    """
    expect_edges(G)
    edges_df = EdgeDataFrame(list(G.edges()), columns=["source", "target"])\
                 .assign(source = lambda x: x.source.astype(int),
                         target = lambda x: x.target.astype(int))
    
    edgecols = list(G.ep)
    if len(edgecols) >= 1:
        if len(edgecols) == 1:
            edges_meta = EdgeDataFrame({f"{edgecols[0]}": list(G.ep[f"{edgecols[0]}"])})
        elif len(edgecols) > 1:
            edges_meta = pd.concat([EdgeDataFrame({f"{edgecols[i]}": list(G.ep[f"{edgecols[i]}"])})
                                    for i in range(len(edgecols))], axis=1)
        return pd.concat([edges_df, edges_meta], axis=1)
    else:
        return edges_df


def _nodes2dataframe(G: gt.Graph) -> NodeDataFrame:
    """
    Takes a Graph-tool graph object and returns nodes data frame.
    """
    expect_nodes(G)
    if len(G.vp) != 0:
        prop_dfs = []
        for prop in G.vp:
            prop_dfs.append(NodeDataFrame({f"{prop}": list(G.vp[f"{prop}"])}))
        prop_dfs = pd.concat(prop_dfs, axis=1)

        main_vprop = list(G.vp)[0]
        if 'name' in prop_dfs.columns and 'name' == main_vprop:
            # Making sure that name var does not conflict with anything
            name_convertible_to_int = all(prop_dfs.name.str.isdigit())
            if name_convertible_to_int == False:
                prop_dfs = prop_dfs.rename(columns={"name":"label"})
                prop_dfs["name"] = prop_dfs.index
                prop_dfs["name"] = prop_dfs["name"].astype(str)
            
            else:
                if all(prop_dfs.name.astype(int) != prop_dfs.index):
                    prop_dfs = prop_dfs.rename(columns={"name":"label"})
                    prop_dfs["name"] = prop_dfs.index
                    prop_dfs["name"] = prop_dfs["name"].astype(str)

            other_cols = list(prop_dfs.loc[:, prop_dfs.columns != 'name'])
            return prop_dfs.reorder_columns(["name"] + other_cols)
        else: 
            other_cols = list(prop_dfs.loc[:, prop_dfs.columns != f'{main_vprop}'])
            return prop_dfs.reorder_columns([f"{main_vprop}"] + other_cols)
    else:
        return NodeDataFrame({"name":list(G.vertices())})
