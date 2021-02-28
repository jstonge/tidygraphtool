
import graph_tool.all as gt
 

def activate(G: gt.Graph, what: str) -> gt.Graph:
  """Activate context of graph object to be in node or edges modes. 
  
  Sugary syntax that builds on top of internal graph property maps.
  """
  if what == 'nodes' or what == 'edges':
    gprop = G.new_gp('string')
    G.gp["active"] = gprop
    G.gp["active"] = f'{what}' 
  else:
    raise ValueError("Can only activate nodes or edges")


def expect_nodes(G: gt.Graph):
  assert G.gp.active == 'nodes', 'This call requires nodes to be active'


def expect_edges(G: gt.Graph):
  assert G.gp.active == 'edges', 'This call requires edges to be active'

