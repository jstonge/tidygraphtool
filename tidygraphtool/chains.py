
"""Module to chain function: https://github.com/explosion/thinc/blob/master/thinc/layers/chain.py"""
from tidygraphtool.node import node_coreness
from tidygraphtool.verbs import add_property, filter_on
from typing import Tuple, Callable, Optional, TypeVar, Any, Dict

from .edgedataframe import EdgeDataFrame
from .nodedataframe import NodeDataFrame
from .context import activate
from .as_data_frame import as_data_frame

from graph_tool.all import gt


# class CustomGtGraph(gt.Graph):
#   def __init__(self):
#     super().__init__()

#   def self_activate(self):
#     activate(self, "nodes")

#   def filter_on(self, criteria):
#     return filter_on(self, criteria)

#   def print(self):
#     print(as_data_frame(self))


#     # def line(self):
#     #     self.kind = 'line'
#     #     return self
#     # def bar(self):
#     #     self.kind='bar'
#     #     return self

# g = CustomGtGraph()

# u = g.add_property("coreness", lambda x: node_coreness())\ 
#      .filter("name == '3'")\
#      .activate('edges')\
#      .add_property(...)