"""Miscellaneous internal tidygraphtool helper functions."""

from typing import (
    Iterable,
    Union
)

import graph_tool.all as gt
import numpy as np
import pandas as pd


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


def assert_nodes_edges_equal(edges, nodes2):
  nodes1 = _extract_nodes(edges)

  nlist1 = list(nodes1)
  nlist2 = list(nodes2)
  try:
      d1 = dict(nlist1)
      d2 = dict(nlist2)
  except (ValueError, TypeError):
      d1 = dict.fromkeys(nlist1)
      d2 = dict.fromkeys(nlist2)
  assert d1 == d2


def assert_gt_edgelist_equal(G: gt.Graph, edges2: pd.DataFrame):
  edges1 = edges2dataframe(G).loc[:, ["source", "target"]]\
                             .assign(source = lambda x: x.source.map(int),
                                     target = lambda x: x.target.map(int))\
                             .sort_values(["source", "target"])\
                             .reset_index(drop=True)

  edges2 = _make_index(edges2).loc[:, ["source", "target"]]\
                             .assign(source = lambda x: x.source.map(int),
                                     target = lambda x: x.target.map(int))\
                             .sort_values(["source", "target"])\
                             .reset_index(drop=True)

  if edges1.equals(edges2) == False:
    raise ValueError("Not the same")


def guess_df_type(x:pd.DataFrame) -> str:
    x = pd.DataFrame(x)
    colnames = x.columns.str.lower()
    if any(colnames.isin(["id", "name"])):
        return "NodeDataFrame"
    elif any(colnames.isin(["source", "target", "from", "to", "weight"])):
        return "EdgeDataFrame"
    else:
        raise ValueError("Unable to guess dataframe type")


def guess_list_type(x:list) -> str:
    if len(x) == 2:
        x1, x2 = [guess_df_type(_) for _ in x]
        if set([x1, x2]) == set(['NodeDataFrame', 'EdgeDataFrame']):
            return 'node_edge'
        else:
            raise ValueError("Unknowen list format")

