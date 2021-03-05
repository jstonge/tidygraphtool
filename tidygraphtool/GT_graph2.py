"""Main module"""
from functools import singledispatchmethod
from typing import Optional, Dict, Union

import graph_tool.all as gt
import pandas as pd
import numpy as np

from .as_data_frame import as_data_frame
from .edgedataframe import EdgeDataFrame, EdgeSeries
from .nodedataframe import NodeDataFrame, NodeSeries
from .augment import augment_prop, _augment_prop_nodes, _augment_prop_edges
from .gt_graph import _as_graph_node_edge, _indexify_edges
from .utils import (
    assert_edges_edges_equal,
    guess_df_type,
    guess_dict_type,
    check_column,
    check_null_values,
    _convert_types2gt,
    assert_nodes_nodes_equal,
    assert_edges_edges_equal,
    assert_nodes_edges_equal,
    assert_index_reset,
    _extract_nodes
)

class GT_graph(gt.Graph):
    def __init__(
        self, 
        nodes: pd.DataFrame = None, 
        edges: pd.DataFrame = None, 
        directed: bool = True, 
        node_key: str = 'name'
    ):
        super().__init__()
        self.activate("nodes")
        self.__from_dict(x={"nodes": nodes, "edges": edges}, directed=directed, node_key=node_key)


    def __from_dict(self, x, directed, node_key):

        if guess_dict_type(x) == 'node_edge':
            # Check if one of the value is None
            if any([v is None for v in x.values()]):
                x = [v for v in x.values() if v is not None][0]
                if guess_df_type(x) == 'EdgeDataFrame':
                    nodes = _extract_nodes(x)
                    x = {"nodes": nodes, "edges": x}
            
            nodes, edges = _as_graph_node_edge(x, node_key=node_key)

            self.add_edge_list(edges[["source", "target"]].to_numpy())

            # Add node metadata
            nodecols = nodes.columns
            if len(nodecols) == 1:
                self.augment_prop(NodeDataFrame(nodes), prop_name=nodecols[0])
            elif len(nodecols) > 1:
                [self.augment_prop(NodeDataFrame(nodes), prop_name=c) for c in nodecols]
            # Add edge metadata
            edgecols = edges.iloc[:, 2::].columns
            if len(edgecols) == 1:
                self.augment_prop(x=EdgeDataFrame(edges), prop_name=edgecols[0])
            elif len(edgecols) > 1:
                [self.augment_prop(x=EdgeDataFrame(edges), prop_name=c) for c in edgecols]

        else:
            raise ValueError("Other types not implemented yet")


    def activate(self, what: str):
        """Activate context of graph object to be in node or edges modes. """
        if what == 'nodes' or what == 'edges':
            gprop = self.new_gp('string')
            self.gp["active"] = gprop
            self.gp["active"] = f'{what}'
        else:
            raise ValueError("Can only activate nodes or edges")


    def augment_prop(self, x: pd.DataFrame, 
                     prop_name: Optional[str] = None) -> gt.Graph:
        """
        Augment nodes in Graph with additional nodes metadata.

        As opposed to mutate, augment_prop modifies in-place the graph.

        :param G: A `gt.Graph` object.
        :param x: A `data Frame` containing information about the nodes in the G. 
        :param prop_name: String that matches the colname. 
        :return: A `Graph_tool` object
        """
        
        if isinstance(x, (NodeDataFrame, NodeSeries)):
            return self.__augment_prop_nodes(x, prop_name)
        elif isinstance(x, (EdgeDataFrame, EdgeSeries)):
            return self.__augment_prop_edges(x, prop_name)
        elif guess_df_type(x) == 'NodeDataFrame':
            return self.__augment_prop_nodes(x, prop_name)
        elif guess_df_type(x) == 'EdgeDataFrame':
            return self.__augment_prop_edges(x, prop_name)
        else:
            raise ValueError("Unknown format")


    def __augment_prop_nodes(self,
                            nodes: Union[NodeDataFrame, NodeSeries],
                            prop_name: Optional[str] = None) -> pd.DataFrame:

        nodes = NodeDataFrame(nodes)

        if prop_name is None:
            prop_name = nodes.columns[0]

        # check_column(nodes, column_names = [prop_name], present=True)
        check_null_values(nodes, [prop_name])

        # Create internal properties maps
        prop_type = _convert_types2gt(nodes, prop_name)
        vprop = self.new_vp(f"{prop_type}")
        self.vp[f"{prop_name}"] = vprop

        assert_nodes_nodes_equal(self, nodes)
        assert_index_reset(nodes)

        main_vprop = list(self.vp)[0]
        if main_vprop in nodes.columns:
            if all([len(_) == 0 for _ in self.vp[f'{main_vprop}']]) == False:
                nodes = self.__reorder_nodes_like_g(main_vprop, nodes)

        for i in range(len(nodes)):
            # Augment graph with nodes metadata
            vprop[i] = nodes[f"{prop_name}"][i]

        return self


    def __reorder_nodes_like_g(self, prop, nodes):
        vprop_vals = list(self.vp[f"{prop}"])
        
        # Finding col with same vals than main vprop
        x = [c for c in nodes.columns if set(nodes[f"{c}"]) == set(vprop_vals)][0]

        nodes = nodes.set_index(nodes[f'{x}'])
        nodes = nodes.loc[vprop_vals]
        nodes = nodes.reset_index(drop=True)
        return nodes


    def __augment_prop_edges(self: gt.Graph,
                            edges: Union[EdgeDataFrame, EdgeSeries],
                            prop_name: str) -> pd.DataFrame:

        edges = EdgeDataFrame(edges)

        # check_column(edges, column_names = [prop_name])
        check_null_values(edges, [prop_name])

        # Create internal properties maps
        prop_type = _convert_types2gt(edges, prop_name)
        eprop = self.new_ep(f"{prop_type}")
        self.edge_properties[f"{prop_name}"] = eprop

        assert_edges_edges_equal(self, edges)
        assert_index_reset(edges)

        if 'source' not in edges:
            edges_df = as_data_frame(self)
            edges = pd.concat([edges_df, edges])

        for i in range(len(edges)):
            # Augment graph with edge metadata
            e = self.edge(edges.source[i], edges.target[i])
            eprop[e] = edges.loc[i, f"{prop_name}"]
        return self


    def filter_on(self, criteria: str):
        """Filter tidystyle on a particular criteria.

        edges = pd.DataFrame({"source":[1,4,3,2], "target":[2,3,2,1]})
        g = as_gt_graph(edges)
        filter_on(g, "source == 2")
        """
        if self.gp.active == "nodes":
            df = NodeDataFrame(as_data_frame(self))
            #!TODO: check_column(nodes, ...)
            df_tmp = df.query(criteria)
            df["bp"] = np.where(df["name"].isin(df_tmp["name"]), True, False)
            self.augment_prop(df, prop_name="bp")
            self = gt.GraphView(self, vfilt=self.vp.bp)
            self = gt.Graph(self, prune=True)
            del self.properties[("v", "bp")]
            
            return self

        # elif self.gp.active == "edges":
        #     df = EdgeDataFrame(as_data_frame(self))
        #     df_tmp = df.query(criteria)
        #     df["bp"] = np.where(df["source"].isin(df_tmp["source"]) &
        #                         df["target"].isin(df_tmp["target"]), True, False)
        #     new_G = augment_prop(self, df, prop_name="bp")
        #     new_G = gt.GraphView(new_G, efilt=new_G.ep.bp)
        #     new_G = gt.Graph(new_G, prune=True)
        #     del new_G.properties[("e", "bp")]
        #     self.g = new_G
        
        else:
            raise ValueError("Context must be activated to nodes or edges")





