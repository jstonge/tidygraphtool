"""Miscellaneous internal tidygraphtool helper functions."""

from tidygraphtool.as_data_frame import as_data_frame
from typing import (
    Iterable,
    Union, 
    Hashable,
    Optional
)

import re
import graph_tool.all as gt
import numpy as np
import pandas as pd
import pandas_flavor as pf


def check(varname: str, value, expected_types: list):
    """
    One-liner syntactic sugar for checking types.
    It can also check callables.
    Should be used like this::
        check('x', x, [int, float])

    :param varname: The name of the variable.
    :param value: The value of the varname.
    :param expected_types: The types we expect the item to be.
    :raises TypeError: if data is not the expected type.
    """
    is_expected_type: bool = False
    for t in expected_types:
        if t is callable:
            is_expected_type = t(value)
        else:
            is_expected_type = isinstance(value, t)
        if is_expected_type:
            break

    if not is_expected_type:
        raise TypeError(
            "{varname} should be one of {expected_types}".format(
                varname=varname, expected_types=expected_types
            )
        )


def check_column(
    df: pd.DataFrame, column_names: Union[Iterable, str], present: bool = True
):
    """
    One-liner syntactic sugar for checking the presence or absence of columns.
    Should be used like this::
        check(df, ['a', 'b'], present=True)
    This will check whether columns "a" and "b" are present in df's columns.
    One can also guarantee that "a" and "b" are not present
    by switching to ``present = False``.

    :param df: The name of the variable.
    :param column_names: A list of column names we want to check to see if
        present (or absent) in df.
    :param present: If True (default), checks to see if all of column_names
        are in df.columns. If False, checks that none of column_names are
        in df.columns.
    :raises ValueError: if data is not the expected type.
    """
    for column_name in column_names:
        if present and column_name not in df.columns:  # skipcq: PYL-R1720
            raise ValueError(
                f"{column_name} not present in dataframe columns!"
            )
        elif not present and column_name in df.columns:
            raise ValueError(
                f"{column_name} already present in dataframe columns!"
            )

def check_null_values(df: pd.DataFrame,  column_names: Union[Iterable, str]):
    """
    One-liner syntactic sugar for checking the null values of columns.
    :raises ValueError: if data is null values.
    """
    check_column(df, column_names)
    assert any([df[f'{colname}'].isnull().values.any() == True 
               for colname 
               in column_names]) == False

def _extract_nodes(edges: pd.DataFrame) -> pd.DataFrame:
    # We loose order here. Issues when we first create graph with
    # edgelist, then we want to augment it.
    all_users = set(edges.source).union(set(edges.target))  
    return pd.DataFrame(all_users).rename(columns={0:"name"}).astype(str)


def assert_nodes_edges_equal(nodes, edges):
    check_column(nodes, ["name"])
    
    x1 = nodes["name"].astype(str)
    x2 = _extract_nodes(edges)["name"]

    assert set(x1) == set(x2)

def assert_nodes_nodes_equal(G, nodes_df):
    nb_nodes_g = len(list(G.vertices()))
    nb_nodes_df = len(nodes_df)
    assert nb_nodes_g == nb_nodes_df

def assert_edges_edges_equal(G, edges_df):
    nb_edges_g = len(list(G.edges()))
    nb_edges_df = len(edges_df)
    assert nb_edges_g == nb_edges_df

def assert_index_reset(df):
    assert all(df.index == range(len(df)))

# def assert_gt_edgelist_equal(G: gt.Graph, edges2: pd.DataFrame):
#     activate(G, "edges")
#     edges1 = as_data_frame(G).loc[:, ["source", "target"]]\
#                                 .assign(source = lambda x: x.source.map(int),
#                                         target = lambda x: x.target.map(int))\
#                                 .sort_values(["source", "target"])\
#                                 .reset_index(drop=True)

#     edges2 = make_index(edges2).loc[:, ["source", "target"]]\
#                                 .assign(source = lambda x: x.source.map(int),
#                                         target = lambda x: x.target.map(int))\
#                                 .sort_values(["source", "target"])\
#                                 .reset_index(drop=True)

#   if edges1.equals(edges2) == False:
#     raise ValueError("Not the same")


def _convert_types2gt(df, prop_name):
    check_column(df, [prop_name], present=True)
    prop_type = str(df[f"{prop_name}"].dtype)
    
    if prop_type == "object":
        prop_type = "string"
    elif prop_type == "float64":
        prop_type = "float"
    elif re.match(r"^int", prop_type): 
        prop_type = prop_type + "_t"
    elif prop_type == 'bool':
        prop_type = 'bool'
    else:
        raise ValueError("Failed to guess type")
    
    return prop_type


def guess_df_type(x:pd.DataFrame) -> str:
    x = pd.DataFrame(x)
    colnames = x.columns.str.lower()
    if any(colnames.isin(["source", "target", "from", "to", "weight"])):
        return "EdgeDataFrame"
    elif any(colnames.isin(["id", "name", "user"])):
        return "NodeDataFrame"
    else:
        raise ValueError("Unable to guess dataframe type")


def guess_dict_type(x:list) -> str:
    if len(x) == 2:
        x1, x2 =  [_ for _ in x]  
        if any([_ in ['nodes', 'vertices'] for _ in (x1, x2)])  & \
            any([_ in ['edges', 'links'] for _ in (x1, x2)]) :
            return 'node_edge'
        else:
            raise ValueError("Unknowen list format")


@pf.register_dataframe_method
def reorder_columns(
    df: pd.DataFrame, column_order: Union[Iterable[str], pd.Index, Hashable]
) -> pd.DataFrame:
    """Reorder DataFrame columns by specifying desired order as list of col names.
    Columns not specified retain their order and follow after specified cols.
    Validates column_order to ensure columns are all present in DataFrame.
    This method does not mutate the original DataFrame.

    Functional usage syntax:
    Given `DataFrame` with column names `col1`, `col2`, `col3`:
    .. code-block:: python
        df = reorder_columns(df, ['col2', 'col3'])
    Method chaining syntax:
    .. code-block:: python
        import pandas as pd
        import janitor
        df = pd.DataFrame(...).reorder_columns(['col2', 'col3'])
    The column order of `df` is now `col2`, `col3`, `col1`.
    Internally, this function uses `DataFrame.reindex` with `copy=False`
    to avoid unnecessary data duplication.
    :param df: `DataFrame` to reorder
    :param column_order: A list of column names or Pandas `Index`
        specifying their order in the returned `DataFrame`.
    :returns: A pandas DataFrame with reordered columns.
    :raises IndexError: if a column within ``column_order`` is not found
        within the DataFrame.
    """
    check("column_order", column_order, [list, tuple, pd.Index])

    if any(col not in df.columns for col in column_order):
        raise IndexError(
            "A column in ``column_order`` was not found in the DataFrame."
        )

    # if column_order is a Pandas index, needs conversion to list:
    column_order = list(column_order)

    return df.reindex(
        columns=(
            column_order
            + [col for col in df.columns if col not in column_order]
        ),
        copy=False,
    )


