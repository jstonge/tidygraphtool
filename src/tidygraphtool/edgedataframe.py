"""Edge extension for dataframe"""

from pandas import DataFrame, Series
class EdgeDataFrame(DataFrame): 
    @property
    def _constructor(self): 
        return EdgeDataFrame 
      
    @property
    def _constructor_sliced(self): 
        return EdgeSeries 

        
class EdgeSeries(Series): 
    @property
    def _constructor(self): 
        return EdgeSeries 
      
    @property
    def _constructor_sliced(self): 
        return EdgeSeries 