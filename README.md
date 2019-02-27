[Original Keras2Cpp Repo](https://github.com/pplonski/keras2cpp)

This is a touchup of the Keras2Cpp repo to get it running on the latest (Jan 2019) version of Keras as well as to interface it back into Python.
The [C1Games Terminal](https://terminal.c1games.com/) AI challenge currently (Feb 2019) doesn't allow for Keras/Tensorflow/Numpy dependencies, so
this library is a workaround for gettings things up and running in Python-based algos that want to explore ML.

Feel free to make a pull request if an issue arises/this goes out of date. The idea is to make ML implementations easier for everyone who wants to
try them, so sharing fixes is encouraged.

## Usage

 1. Save your Keras network weights and architecture.
 2. Dump network structure to plain text file with `dump_to_simple_cpp.py` script.
 3. Compile network with code from `keras_model.h` and `keras_model.cc` files - see examples below.
 4. In the case of compiling for uploading to [C1Games' Terminal](https://terminal.c1games.com/), use the `c_wrapper.py` interface in your algo.

## Example (Terminal)
 1. Run one iteration of simple CNN on MNIST data with `example/mnist_cnn_one_iteration.py` script. It will produce files with architecture `example/my_nn_arch.json` and weights in HDF5 format `example/my_nn_weights.h5`.
 2. Dump network to plain text file `python dump_to_simple_cpp.py -a example/my_nn_arch.json -w example/my_nn_weights.h5 -o example/dumped.nnet`.
 3. Compile example with either CMake or the alternative `./alt_build.sh` when you realize the CMakeLists.txt is broken.
 4. Use the c_wrapper.py interface in your algo.

## Example (Pure C++)

 1. Run one iteration of simple CNN on MNIST data with `example/mnist_cnn_one_iteration.py` script. It will produce files with architecture `example/my_nn_arch.json` and weights in HDF5 format `example/my_nn_weights.h5`.
 2. Dump network to plain text file `python dump_to_simple_cpp.py -a example/my_nn_arch.json -w example/my_nn_weights.h5 -o example/dumped.nnet`.
 3. Compile example `g++ -std=c++11 keras_model.cc example_main.cc` - see code in `example_main.cc`.
 4. Run binary `./a.out` - you shoul get the same output as in step one from Keras.

## Testing

[Ryan D note: The testing script is not updated to the latest version of Keras. You can make a pull request if you fix it, but it's not a _needed_ script.]
If you want to test dumping for your network, please use `test_run.sh` script. Please provide there your network architecture and weights. The script does the following job:

 1. Dump network into text file.
 2. Generate random sample.
 3. Compute predictions from keras and keras2cpp on generated sample.
 4. Compare predictions.