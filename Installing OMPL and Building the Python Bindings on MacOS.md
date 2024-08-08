# Introduction
This documentation describes details steps of installing the OMPL motion planning library and building its Python bindings so that we can call OMPL funtionalities in Python.

Platform: M1 Mac, MacOS 13.0 Venture

References: 

- OMPL installation guide (very brief): https://ompl.kavrakilab.org/installation.html
- OMPL Github: https://github.com/ompl/ompl

# Install dependencies from Homebrew

The most important dependencies are cmake, boost-python3, castxml, and Eigen, which are needed to build the C++ modules. 

Typically, you will have the latest version of cmake ready on MacOS. Otherwise  you may install it from brew

``` sh 
brew install cmake
```

Install other important dependencies from brew

```sh
brew install boost-python3 eigen castxml flann doxygen spot
```

# Install Py++ to Homebrew's Python 

 Py++ is the most important package for generating the  Python bindings. On MacOS, we need to ensure it is installed with the Homebrew's Python. However, Homebrew's Python does not allow installing external packages using `pip`, and we have to install potential external packages through either 1) `brew install python-xx` where xx is the package name. 2) If the package is not available through 1), need to install it through `setup.py` with `setuptools`. For Py++, we have to follow the second approach.

First, install `python-setuptools` to Homebrew's Python

```sh
brew install python-setuptools
```

Then, clone Py++ repo from Github, and enter the `pyplusplus` folder.

```sh
git clone https://github.com/ompl/pyplusplus.git
cd pyplusplus
```

Then call `setup.py` in the `pyplusplus` folder using Homebrew's Python.

```sh
$(brew --prefix python)/libexec/bin/python setup.py
```

Here `(brew --prefix python)/libexec/bin/python` is the path to Homebrew's Python.

Finally, check if Py++ is properly installed through

``` sh
$(brew --prefix python)/libexec/bin/pip list
```

# Get OMPL from Github

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

# Call cmake to generate the makefiles 

``` sh
cmake -DPYTHON_EXEC=$(brew --prefix python)/libexec/bin/python ../..
```



**It is very important to specify `-DPYTHON_EXEC` as the Homebrew Python at  `$(brew --prefix python)/libexec/bin/python` where the Py++ (pyplusplus) package is installed and boost-python3 is linked.** 

During the execution of cmake, the output log may point out some packages missing on your computer. Install them accordingly if possible.

# Build the Python bindings

After successful execution of cmake, run the following command under `ompl/build/Release`

```sh 
make -j $N_cores update_bindings
```

to build the Python bindings, replace `$N_cores` with the number of cores on your computer.

Building the Python bindings takes a lot of computational resources. It is recommended that you have at least 6G of RAM, and use multiple cores. The execution time also takes 10 minutes or more.

**Remark: one error you may encouter during the `make` process is that the path to the ompl folder contains '(' or ')' and becomes illegal. It that arises, move the ompl folder to a simpler place like `~/`**.

When finished, the Python bindings will be located in the folder of

```sh
ompl/build/Release/py-bindings/bindings
```

# Using the bindings in Python

Copy the `ompl/build/Release/py-bindings/bindings` folder to your own Python repo, and change the name of the folder from ``bindings`` to ``omplpy``. Then you may treat the current ``omplpy`` folder as a regular Python package, and import OMPL libraries normally in Python in the following style

```python
from omplpy import base as ob
from omplpy import geometric as og
```
