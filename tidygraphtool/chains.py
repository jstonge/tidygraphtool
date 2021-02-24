
"""Module to chain function: https://github.com/explosion/thinc/blob/master/thinc/layers/chain.py"""
from typing import Tuple, Callable, Optional, TypeVar, Any, Dict

from .edgedataframe import EdgeDataFrame
from .nodedataframe import NodeDataFrame

from graph_tool.all import gt


# InT = TypeVar("InT")
# OutT = TypeVar("OutT")
# MidT = TypeVar("MidT")

# def chain(
#     layer1: gt.Graph[InT, MidT], 
#     layer2: gt.Graph[MidT, OutT], 
#     *layers: gt.Graph
# ) -> gt.Graph[InT, XY_YZ_OutT]:
#     """Compose two models `f` and `g` such that they become layers of a single
#     feed-forward model that computes `g(f(x))`.
#     Also supports chaining more than 2 layers.
#     """
#     layers = (layer1, layer2) + layers
#     # set output dimension according to last layer

#     model: Model[InT, Any] = Model(
#         ">>".join(layer.name for layer in layers),
#         forward,
#         init=init,
#         dims=dims,
#         layers=layers,
#     )
#     return model
