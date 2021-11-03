
# tidygraphtool

A tidy API for network manipulation with [graph-tool library](https://graph-tool.skewed.de/) inspired by [tidygraph](https://github.com/jstonge/tidygraphtool). This is an experimental project of mine. Use at your own risk.

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
## Installation

Installation is a bit weird because graph-tool cannot be pip-install. The easiest way to install is to clone this repo, then from the root directory of the package: 

```bash
conda env create --file tidy_gt.yml &&
conda activate tidy_gt &&
pip install tidygraphtool
```
The first and second lines create an environement with all the depenencies from `gt.yml`. The third line install tidygraphtool from `pypi` the package at large within the `gt` environment. 

## INSPIRATION
 - `graph_tool`: tidygraph is thin wrapper of graph_tool (https://graph-tool.skewed.de/).
 - `tidygraph`: the tidy API that cast graph analysis as two dataframe (https://github.com/thomasp85/tidygraph).
 - `dplyr`: verbs like API from which tidygraph draw inspiration .

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
g = gt_graph(nodes=nodes, edges=edges)
g = (g >>
       add_property("node_coreness", node_coreness(g)) >>
       add_property("pr", centrality_pagerank(g)) >>
       filter_on("node_coreness > 3 & pr > 10") >>
       activate("edges") >>
       add_property("edge_bet", centrality_edge_betweenness(g)))
```
