# Novelty face authentication with liveness detection using depth and IR camera

## Requirements

To run our code, you need a few programs and libraries.
Code was writen for Linux.

```
python (>=3.6)
OpenCV # https://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html
```

All python library dependencies are in requirements.txt file. Notice that
you may need cmake, gcc, python-dev and python's setuptools to install them.
As well as CUDA to run ( look at official site
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html to
find informations about installation).
You may try calculations on CPU (no gpu tensor-flow) but GPU is hardly recommended
and only tested verssion.



## Style guide

### C++

Before pushing, code should be formatted with `clang-format` using the `-style=file` option, which will use the `.clang-format` file located in the main directory of this repository. Things not covered by `.clang-format`:

* Names of everything that can be used as a type - e.g. classes, structs, enums - should be written in CamelCase.
* Every other name - e.g. functions, variables - should be lowercase and use underscores.
* Use `.cpp` and `.hpp` file extensions.
* Header files should have definitions at the top of the file, and implementations below them. Don't split the implementation into a second file.

### Python

Refer to [PEP 8](https://www.python.org/dev/peps/pep-0008/).
