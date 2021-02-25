
import pandas as pd
import graph_tool.all as gt

def as_data_frame(G: gt.Graph) -> pd.DataFrame:
    if G.active == 'nodes':
        return _nodes2dataframe(G)
    elif G.active == 'edges':
        return _edges2dataframe(G)
    else:
        raise ValueError("Nodes or edges must be active")


def _edges2dataframe(G: gt.Graph) -> pd.DataFrame:
    """Takes a Graph-tool graph object and returns edges data frame"""
    tmp_df = pd.DataFrame(list(G.edges()), columns=["source", "target"])
    tmp_df["source"] = tmp_df.source.astype(str)
    tmp_df["target"] = tmp_df.target.astype(str)
    return tmp_df


def _nodes2dataframe(G: gt.Graph) -> pd.DataFrame:
    """Takes a Graph-tool graph object and returns nodes data frame"""
    if len(G.vp) != 0:
        prop_dfs = []
        for prop in G.vp:
            prop_dfs.append(pd.DataFrame({f"{prop}": list(G.vp[f"{prop}"])}))
        prop_dfs = pd.concat(prop_dfs, axis=1)

        if 'name' in prop_dfs.columns:
            prop_dfs = prop_dfs.rename(columns={"name":"label"})

        prop_dfs["name"] = prop_dfs.index
        prop_dfs["name"] = prop_dfs["name"].astype(str)

        return prop_dfs

    else:
        return pd.DataFrame({"name":list(G.vertices())})  
