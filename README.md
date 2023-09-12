# Introduction
The code for the project Multi-agent Motion Planning on the Hybrid Graph.

The core functionalities are encapsulated in the *panav* Python package.

## Required libraries
* Jupyter lab
* matplotlib
* polytope: a Python package that allows flexible construction of convex and non-convex polytopes, called polytope, which is part of the TuLip control toolbox. Install by `pip install polytope`.
* cvxopt: a convex optimization library. Install by `conda install -c conda-forge cvxopt`. Required by pypoman.
* pypoman: this library implements common operations over convex polyhedra such as polytope projection, double description (conversion between halfspace and vertex representations), computing the Chebyshev center, etc. Install by `pip install pypoman`.
* shapely: a powerful library for geometry shape manipulations. Useful in visualization. Install by `conda install -c conda-forge shapely`
* cvxpy: a convenient Python interface to write and solve convex and mix-interger programming problems. `pip install cvxpy`.
* **Gurobi**: a semi-commericalized optimization solver. Install its python interface + gurobi license. No need to install the entire Gurobi software.
  * Install Gurobi Python interface: https://www.gurobi.com/documentation/10.0/quickstart_windows/cs_anaconda_and_grb_conda_.html.
  * Create a Gurobi account, request for an academic license, run grbt get key to install the license on your computer. https://www.gurobi.com/login/
* Networkx: used to construct the high-level MAMP search tree, as well as determining the order in a partial ordering. Install by 


## Structure of the repo

* Virtual navigation environment construction and visualization.
* Global multi-agent motion planning.
* Local path tracking & distributed multi-agent collision avoidance.
* Experiments written on Python notebooks.
