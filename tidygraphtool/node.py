import graph_tool.all as gt 

from .nodedataframe import NodeDataFrame, NodeSeries
from .context import expect_nodes

def node_coreness(G: gt.Graph) -> NodeSeries:
    """
    Measures coreness of each node. See ?gt.kcore_decomposition
    
    Example
    =======
    .. code-block:: python
        
        mutate(g, "coreness", node_coreness(g))
    """
    expect_nodes(G)
    return NodeDataFrame({"nc": list(gt.kcore_decomposition(G))})["nc"]