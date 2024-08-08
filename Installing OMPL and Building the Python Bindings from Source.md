# Introduction
This documentation describes details steps of installing the OMPL motion planning library and building its Python bindings so that we can call OMPL funtionalities in Python.

Platform: M1 Mac, MacOS 13.0 Venture

References: 

- OMPL installation guide (very brief): https://ompl.kavrakilab.org/installation.html
- OMPL Github: https://github.com/ompl/ompl

# First steps

The most important dependencies are cmake, boost-python3, castxml, and Eigen, which are needed to build the C++ modules. 

Typically, you will have the latest version of cmake ready on MacOS. Otherwise  you may install it from brew

``` sh 
brew install cmake
```

Install other important dependencies from brew

```sh
brew install boost-python3 eigen castxml flann doxygen spot
```

Install Py++ package for your Python environment. You may do this in your anaconda environment. Py++ is the most important package for generating the  Python bindings.

``` 
pip install pyplusplus
```

Clone the latest OMPL library release

``` 
git clone https://github.com/ompl/ompl.git
```

Create a build and release workspace in the OMPL folder

``` shell
cd ompl
mkdir build
cd build
mkdir Release
cd Release
```

# The most important step: call cmake to generate the makefiles 

``` sh
cmake -DPYTHON_EXEC=$ENTER_YOUR_PYTHON_PATH ../..
```

**It is very important to set the value $ENTER_YOUR_PYTHON_PATH above to the Python where the Py++ (pyplusplus) package is installed. **

Particularly, if you are using conda, first activate the environment where you have Py++ (pyplusplus) installed, then find out the path to the corresponding Python by

``` sh
which python
```

Copy the terminal output to be the value for ``$ENTER_YOUR_PYTHON_PATH``.

During the execution of cmake, the output log may point out some packages missing on your computer. Install them accordingly if possible.

After successful execution of cmake, run

``` sh make -j 4 update_bindings
make update_bindings
```

to build the Python bindings. Alternatively, you may run

```sh 
make -j $N_cores update_bindings
```

to build the bindings with multiple cores, replace `$N_cores` with the number of cores on your computer.

Building the Python bindings takes a lot of computational resources. It is recommended that you have at least 6G of RAM. The execution time also takes 10 minutes or more.

When finished, the Python bindings will be located in the folder of

```sh
ompl/build/Release/py-bindings/bindings
```

# Using the bindings in Python

Copy the `bindings` folder to your own Python repo, and change the name of the folder from ``bindings`` to ``omplpy``. Then you may treat the current ``omplpy`` folder as a regular Python package, and import OMPL libraries normally in Python in the following style

```python
from omplpy import base as ob
from omplpy import geometric as og
```

And so on.
