
import graph_tool.all as gt
 
def activate(G: gt.Graph, what: str) -> gt.Graph:
  """Activate context of graph object to be in node or edges modes. 
  
  Sugary syntax that builds on top of `setattr`.
  """
  if what == 'nodes':
    return setattr(G, "active", "nodes")
  elif what == 'edges':
    return setattr(G, "active", "edges")
  else:
    raise ValueError("Can only activate nodes or edges")


def expect_nodes(G: gt.Graph):
  assert G.active == 'nodes', 'This call requires nodes to be active'


def expect_edges(G: gt.Graph):
  assert G.active == 'edges', 'This call requires edges to be active'

 


 
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




# IMPLEMENTATION TIDYGRAPH: https://rdrr.io/cran/tidygraph/src/R/context.R
# #' @importFrom R6 R6Class
# ContextBuilder <- R6Class(
#   'ContextBuilder',
#   public = list(
#     set = function(graph) {
#       stopifnot(inherits(graph, 'tbl_graph'))                        # must inherits from gt.Graph
#       private$context <- c(private$context, list(graph))
#       invisible(self)
#     },
#     clear = function() {
#       private$context <- private$context[-length(private$context)]  
#     },
#     alive = function() {
#       length(private$context) != 0
#     },
#     graph = function() {
#       private$check()
#       private$context[[length(private$context)]]
#     },
#     nodes = function() {
#       as_tibble(self$graph(), active = 'nodes')
#     },
#     edges = function() {
#       as_tibble(self$graph(), active = 'edges')
#     },
#     active = function() {
#       private$check()
#       active(self$graph())
#     },
#     free = function() {
#       private$FREE != 0 || inherits(self$graph(), 'free_context_tbl_graph')
#     },
#     force_free = function() {
#       private$FREE <- private$FREE + 1
#     },
#     force_unfree = function() {
#       private$FREE <- private$FREE - 1
#     }
#   ),
#   private = list(
#     context = list(),
#     FREE = 0,
#     check = function() {
#       if (!self$alive()) {
#         stop('This function should not be called directly', call. = FALSE)
#       }
#     }
#   )
# )
# #' @export
# .graph_context <- ContextBuilder$new()
# expect_nodes <- function() {
#   if (!.graph_context$free() && .graph_context$active() != 'nodes') {
#     stop('This call requires nodes to be active', call. = FALSE)
#   }
# }
# expect_edges <- function() {
#   if (!.graph_context$free() && .graph_context$active() != 'edges') {
#     stop('This call requires edges to be active', call. = FALSE)
#   }
# }

# #' Access graph, nodes, and edges directly inside verbs
# #'
# #' These three functions makes it possible to directly access either the node
# #' data, the edge data or the graph itself while computing inside verbs. It is
# #' e.g. possible to add an attribute from the node data to the edges based on
# #' the terminating nodes of the edge, or extract some statistics from the graph
# #' itself to use in computations.
# #'
# #' @return Either a `tbl_graph` (`.G()`) or a `tibble` (`.N()`)
# #'
# #' @rdname context_accessors
# #' @name context_accessors
# #'
# #' @examples
# #'
# #' # Get data from the nodes while computing for the edges
# #' create_notable('bull') %>%
# #'   activate(nodes) %>%
# #'   mutate(centrality = centrality_power()) %>%
# #'   activate(edges) %>%
# #'   mutate(mean_centrality = (.N()$centrality[from] + .N()$centrality[to])/2)
# NULL

# #' @describeIn context_accessors Get the tbl_graph you're currently working on
# #' @export
# .G <- function() {
#   .graph_context$graph()
# }
# #' @describeIn context_accessors Get the nodes data from the graph you're currently working on
# #' @export
# .N <- function() {
#   .graph_context$nodes()
# }
# #' @describeIn context_accessors Get the edges data from the graph you're currently working on
# #' @export
# .E <- function() {
#   .graph_context$edges()
# }

# #' Register a graph context for the duration of the current frame
# #'
# #' This function sets the provided graph to be the context for tidygraph
# #' algorithms, such as e.g. [node_is_center()], for the duration of the current
# #' environment. It automatically removes the graph once the environment exits.
# #'
# #' @param graph A `tbl_graph` object
# #'
# #' @param free Should the active state of the graph be ignored?
# #'
# #' @param env The environment where the context should be active
# #'
# #' @export
# #' @keywords internal
# .register_graph_context <- function(graph, free = FALSE, env = parent.frame()) {
#   stopifnot(is.tbl_graph(graph))
#   if (identical(env, .GlobalEnv)) {
#     stop('A context cannot be registered to the global environment', call. = FALSE)
#   }
#   if (free) {
#     class(graph) <- c('free_context_tbl_graph', class(graph))
#   }
#   .graph_context$set(graph)
#   do.call(on.exit, alist(expr = .graph_context$clear(), add = TRUE), envir = env)
#   invisible(NULL)
# }
# .free_graph_context <- function(env = parent.frame()) {
#   if (identical(env, .GlobalEnv)) {
#     stop('A context cannot be freed in the global environment', call. = FALSE)
#   }
#   .graph_context$force_free()
#   do.call(on.exit, alist(expr = .graph_context$force_unfree(), add = TRUE), envir = env)
#   invisible(NULL)
# }
# #' Evaluate a tidygraph algorithm in the context of a graph
# #'
# #' All tidygraph algorithms are meant to be called inside tidygraph verbs such
# #' as `mutate()`, where the graph that is currently being worked on is known and
# #' thus not needed as an argument to the function. In the off chance that you
# #' want to use an algorithm outside of the tidygraph framework you can use
# #' `with_graph()` to set the graph context temporarily while the algorithm is
# #' being evaluated.
# #'
# #' @param graph The `tbl_graph` to use as context
# #'
# #' @param expr The expression to evaluate
# #'
# #' @return The value of `expr`
# #'
# #' @export
# #'
# #' @examples
# #' gr <- play_erdos_renyi(10, 0.3)
# #'
# #' with_graph(gr, centrality_degree())
# #'
# with_graph <- function(graph, expr) {
#   .register_graph_context(graph, free = TRUE)
#   expr
# }