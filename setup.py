from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='tidygraphtool',
    url='https://github.com/jstonge/tidygraphtool',
    author='Jonathan St-Onge',
    author_email='jonathanstonge7@gmail.com',
    packages=['tidygraphtool'],
    install_requires=['pipey'],
    version='0.2',
    package_dir={'': 'src'},
    license='MIT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    description='A tidy API for network manipulation with Graph-tool inspired by tidygraph.',
)
