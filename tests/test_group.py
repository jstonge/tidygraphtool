
import pytest
import pandas as pd

from tidygraphtool.all import *


def get_dat_simple(with_node_meta=False):

    if with_node_meta == False:
      nodes = pd.DataFrame({"name": ["Bob", "Alice", "Joan", "Melvin"]})
    else:
      nodes = pd.DataFrame({"name": ["Bob", "Alice", "Joan", "Melvin"],
                            "job":  ["teacher", "driver", "dancer", "influence"]})
    
    edges = pd.DataFrame({"source": ["Bob", "Bob", "Joan", "Alice"],
                          "target": ["Joan", "Melvin", "Alice", "Joan"],
                          "weight": [3,3,2,10]})
    return nodes, edges

#!TODO: Write tests to make sure that groups correspond to nodes in g.