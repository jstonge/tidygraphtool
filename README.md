
# tidygraphtool

A tidy API for network manipulation with Graph-tool inspired by tidygraph.


* Free software: MIT license
* Documentation: https://tidygraphtool.readthedocs.io.


## Features

  - tidy and functional API for graph wranling with the graph-tool library.

## Setting up conda environment

Graph-tool is best installed in a fresh environment. 

`
conda create --name graph_tool_env python=3.6 &&
conda activate graph_tool_env &&
conda install -c conda-forge graph-tool &&
conda install -c conda-forge ipython jupyter pandas pyarrow &&
pip install networkx matplotlib ipykernel &&
python -m ipykernel install --user --name=graph_tool_env
`

## INSPIRATION
 - `graph_tool`: tidygraph is thin wrapper of graph_tool (https://graph-tool.skewed.de/).
 - `networkx`: we also draw inspiration from networkx.
 - `tidygraph`: the tidy API that cast graph analysis as two dataframe (https://github.com/thomasp85/tidygraph).
 - `dplyr`: verbs like API from which tidygraph draw inspiration .
 - `pyjanitor`: general organisation of python data wranling with pandas (https://github.com/ericmjl/pyjanitor).
 - `thinc`: functional API for deep learning (especially chaining and operator overloading; https://github.com/explosion/thinc).
 - `geopandas`: how they expand pandas dataframe class (https://github.com/geopandas/geopandas/tree/master/geopandas).

## MODELS TO INTEGRATE
 - `hSBM_Topicmodel`: topic modeling based on graph_tool (https://github.com/martingerlach/hSBM_Topicmodel)
 - `bipartiteSBM`: bipartite community detection based on graph_tool (https://github.com/junipertcy/bipartiteSBM)
