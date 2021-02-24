"""Console script for tidygraphtool."""
import argparse
import sys


def main():
    """Console script for tidygraphtool."""
    parser = argparse.ArgumentParser()
    parser.add_argument('_', nargs='*')
    args = parser.parse_args()

    print("Arguments: " + str(args._))
    print("Replace this message by putting your code into "
          "tidygraphtool.cli.main")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover



# BASH script to set up the environment with conda
# conda create --name graph_tool_env python=3.6
# conda activate graph_tool_env
# conda install -c conda-forge graph-tool
# conda install -c conda-forge ipython jupyter pandas
# pip install cython networkx fa2 matplotlib ipykernel
# python -m ipykernel install --user --name=graph_tool_env