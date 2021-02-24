
from typing import Tuple, Callable, Optional, TypeVar, Any, Dict

from .edgedataframe import EdgeDataFrame
from .nodedataframe import NodeDataFrame

from graph_tool.all import gt

 
# We do not have the activate function that tidygraph uses
# to tell what we want to filter on. One way out would be to use
# context manager + chaining operator, as per the thinc library, i.e.
#
# model = with_array(
#     chain(
#         MultiEmbed(width),
#         Hidden(width),
#         clone(CNN(width), depth),
#         Softmax(n_tags)
#     )
#
# with Model.define_operators({">>": chain}):
#     model = Relu(512) >> Relu(512) >> Softmax()
#
# becomes something like that...
#
# g = with_edges(
#        chain(
#           mutate("degree_in", centrality_degree(g, mode="in")),
#           mutate("coreness", node_coreness(g)),
#           filter("coreness >  5")
#     )
# )
#
# with Model.define_operators({">>": chain}):
#    g = with_edges(
#           mutate("degree_in", centrality_degree(g, mode="in")) 
#           >> mutate("coreness", node_coreness(g)) 
#           >> filter("coreness > 5")
#         )
#


# https://github.com/explosion/thinc/blob/master/thinc/layers/with_array.py

# @registry.layers("with_array.v1")
# def with_array(layer: Model[ArrayXd, ArrayXd], pad: int = 0) -> Model[SeqT, SeqT]:
#     """Transform sequence data into a contiguous 2d array on the way into and
#     out of a model. Handles a variety of sequence types: lists, padded and ragged.
#     If the input is a 2d array, it is passed through unchanged.
#     """
#     return Model(
#         f"with_array({layer.name})",
#         forward,
#         init=init,
#         layers=[layer],
#         attrs={"pad": pad},
#         dims={name: layer.maybe_get_dim(name) for name in layer.dim_names},
#     )

# def with_edge(layer: Model[ArrayXd, ArrayXd], pad: int = 0) -> Model[SeqT, SeqT]:
#     """
#     Transform sequence data into a series of step in edge context.
#     """
#     return Model(
#         f"with_array({layer.name})",
#         forward,
#         init=init,
#         layers=[layer],
#         attrs={"pad": pad},
#         dims={name: layer.maybe_get_dim(name) for name in layer.dim_names},
#     )
