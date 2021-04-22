
# tidygraphtool

A tidy API for network manipulation with Graph-tool inspired by tidygraph. This is an experimental project of mine. Use at your own risk.

## Usage

```python
from tidygraphtool.all import *
g = play_sbm(n_k=500)
(
  g >>
    activate("nodes") >>
    add_property("degree", centrality_degree, mode="total") >>
    filter_on("degree == 2") >>
    summary()
)

```


## Features

  - tidy and functional API for graph wranling with the [graph-tool library](https://graph-tool.skewed.de/).

## Installation

Because that graph-tool is mostly a c++ library, tidygraphtool is best installed from anaconda in a fresh environment. 

```bash
git clone https://github.com/jstonge/tidygraphtool.git
conda env create --file gt.yml
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



## PROTOYPE FUNCTIONAL INTERFACE

```Python
#### Current
g = gt_graph(nodes=nodes, edges=edges)
g = add_property(g, "node_coreness", node_coreness)
g = add_property(g, "pr", centrality_pagerank)
g = filter_on(g, "node_coreness > 3")
g = activate(g, "edges")
g = add_property(g, "edge_bet", centrality_edge_betweenness)
```
```Python
#### With #pipey, we can currently do
g = gt_graph(nodes=nodes, edges=edges)
g = (g >>
       add_property("node_coreness", node_coreness(g)) >>
       add_property("pr", centrality_pagerank(g)) >>
       filter_on("node_coreness > 3 & pr > 10") >>
       activate("edges") >>
       add_property("edge_bet", centrality_edge_betweenness(g)))
```
