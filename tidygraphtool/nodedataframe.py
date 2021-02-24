from pandas import DataFrame, Series

# class NodeDataframe(DataFrame):

#     """
#     A GeoDataFrame object is a pandas.DataFrame that has a column
#     with geometry. In addition to the standard DataFrame constructor arguments,
#     GeoDataFrame also accepts the following keyword arguments:
#     Parameters
#     ----------
#     crs : value (optional)
#         Coordinate Reference System of the geometry objects. Can be anything accepted by
#         :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
#         such as an authority string (eg "EPSG:4326") or a WKT string.
#     geometry : str or array (optional)
#         If str, column to use as geometry. If array, will be set as 'geometry'
#         column on GeoDataFrame.
#     Examples
#     --------
#     Constructing GeoDataFrame from a dictionary.
#     >>> from shapely.geometry import Point
#     >>> d = {'col1': ['name1', 'name2'], 'geometry': [Point(1, 2), Point(2, 1)]}
#     >>> gdf = geopandas.GeoDataFrame(d, crs="EPSG:4326")
#     >>> gdf
#         col1                 geometry
#     0  name1  POINT (1.00000 2.00000)
#     1  name2  POINT (2.00000 1.00000)
#     Notice that the inferred dtype of 'geometry' columns is geometry.
#     >>> gdf.dtypes
#     col1          object
#     geometry    geometry
#     dtype: object
#     See also
#     --------
#     GeoSeries : Series object designed to store shapely geometry objects
#     """
    # def __init__(self, *args,  **kwargs):


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