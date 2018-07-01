# Novelty face authentication with liveness detection using depth and IR camera

Currently, commonly used authentication methods are becoming obsolete,
due to the lack of convenience, adversarial technology improvements
or easy to make user errors.
We believe that biometric authentication has the potential to become
one of the safest, most convenient and most efficient methods.
In this project we attempt to improve face recognition
using a camera with depth-perception and infrared capabilities, as well as
our search for new liveness detection methods such as skin detection
using multispectral imaging.
More details can be found in our paper in the `whitepaper/`.

## Requirements

Python >= 3.6 is required to run Python code, and C++ code requries a compiler
supporting C++17.
Detailed Python requirements are in the `requirements.txt` file, while C++
requirements are described in `libkinect/README.md`.
Python dependencies might require `cmake`, `gcc`, `python-dev`, and Python's
`setuptools` to install them.

You might also need to install [OpenCV](https://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html)
and [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
to run code in `face_auth/`.
Running Tensorflow on CPU might be possible, but the code was only tested on GPU.

## Style guide

### C++

Before pushing, code should be formatted with `clang-format` using the `-style=file` option, which will use the `.clang-format` file located in `libkinect/`. Things not covered by `.clang-format`:

* Names of everything that can be used as a type - e.g. classes, structs, enums - should be written in CamelCase.
* Every other name - e.g. functions, variables - should be lowercase and use underscores.
* Use `.cpp` and `.hpp` file extensions.
* Header files should have definitions at the top of the file, and implementations below them. Don't split the implementation into a second file.

### Python

Refer to [PEP 8](https://www.python.org/dev/peps/pep-0008/).
