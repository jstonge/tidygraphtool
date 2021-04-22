"""Node extension for dataframe"""

from pandas import DataFrame, Series

class NodeDataFrame(DataFrame): 
    @property
    def _constructor(self): 
        return NodeDataFrame 
      
    @property
    def _constructor_sliced(self): 
        return NodeSeries

class NodeSeries(Series): 
    @property
    def _constructor(self): 
        return NodeSeries 
      
    @property
    def _constructor_sliced(self): 
        return NodeSeries 