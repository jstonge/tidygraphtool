
# tidygraphtool

A tidy API for network manipulation with Graph-tool inspired by tidygraph.

## Usage

```python
from tidygraphtool.all import *
g = play_sbm(n_k=500)
(
  g >>
    activate("nodes") >>
    add_property("degree", centrality_degree(g, mode="total")) >>
    filter_on("degree == 2") >>
    summary()
)

```


## Features

  - tidy and functional API for graph wranling with the graph-tool library.

## Setting up conda environment

Graph-tool is best installed in a fresh environment. 

```bash
conda create --name graph_tool_env &&
conda activate graph_tool_env &&
conda install -c conda-forge graph-tool &&
conda install -c conda-forge ipython jupyter pandas pandas-flavor &&
pip install networkx matplotlib ipykernel pyarrow 

# optional on jupyter notebook
# python -m ipykernel install --user --name=graph_tool_env
```

## INSPIRATION
 - `graph_tool`: tidygraph is thin wrapper of graph_tool (https://graph-tool.skewed.de/).
 - `networkx`: we also draw inspiration from networkx.
 - `tidygraph`: the tidy API that cast graph analysis as two dataframe (https://github.com/thomasp85/tidygraph).
 - `dplyr`: verbs like API from which tidygraph draw inspiration .
 - `pyjanitor`: general organisation of python data wranling with pandas (https://github.com/ericmjl/pyjanitor).
 - `thinc`: functional API for deep learning (especially chaining and operator overloading; https://github.com/explosion/thinc).

## MODELS TO INTEGRATE
 - `hSBM_Topicmodel`: topic modeling based on graph_tool (https://github.com/martingerlach/hSBM_Topicmodel)
 - `bipartiteSBM`: bipartite community detection based on graph_tool (https://github.com/junipertcy/bipartiteSBM)



## PROTOYPE FUNCTIONAL INTERFACE INSPIRED FROM THINC

```Python
#### Current
g = gt_graph(nodes=nodes, edges=edges)
g = add_property(g, "node_coreness", node_coreness(g))
g = add_property(g, "pr", centrality_pagerank(g))
g = filter_on(g, "node_coreness > 3")
activate(g, "edges")
g = add_property(g, "edge_bet", centrality_edge_betweenness(g))
```
```Python
#### With pipey, we can currently do
g = gt_graph(nodes=nodes, edges=edges)
g = g 
  >> add_property("node_coreness", node_coreness(g))
  >> add_property("pr", centrality_pagerank(g))
  >> filter_on("node_coreness > 3 & pr > 10")

activate(g, "edges")

g = g >> add_property("edge_bet", centrality_edge_betweenness(g))
```

```Python
#### Functional like THINC
g = gt_graph(nodes=nodes, edges=edges)

g =  with_graph(
  chain(
    add_property("node_coreness", lambda x: x.node_coreness())
    add_property("pr", lamba x: x.pagerank())
    filter_on("node_coreness > 3")
    activate("edges")
    add_property("edge_bet", lambda x: x.centrality_edge_betweenness())
  )
)
```

```Python
#### With operator overloading
with gt_graph.define_operators({">>": chain}):
    g = with_graph(
      add_property("node_coreness", lambda x: x.node_coreness()) 
      >> add_property("pr", lamba x: x.pagerank()
      >> filter_on("node_coreness > 3")
      >> activate("edges")
      >> add_property("edge_bet", lmabda x: x.centrality_edge_betweenness())
    )
```
