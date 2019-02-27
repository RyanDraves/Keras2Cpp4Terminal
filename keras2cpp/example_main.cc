#include "keras_model.h"

#include <iostream>

using namespace std;
using namespace keras;

// Step 1
// Dump keras model and input sample into text files
// python dump_to_simple_cpp.py -a example/my_nn_arch.json -w example/my_nn_weights.h5 -o example/dumped.nnet

// Step 2
// Use text files in c++ example. To compile (just C++):
// g++ -std=c++11 keras_model.cc example_main.cc
// To execute:
// ./a.out

// Step 2 (Terminal)
// Use CMake to compile the shared library into a .so file
// make .
// Alternatively, use the hacked up shell script because who really knows how CMake works
// ./alt_build.sh
// Modify c_wrapper.py to make sure the shared library is correct interfaced back into Python
// Use c_wrapper.py just like any other Python module

int main() {
  cout << "This is simple example with Keras neural network model loading into C++.\n"
           << "Keras model will be used in C++ for prediction only." << endl;

  // Initialize an input data chunk of all zeroes (assuming that's your model's input)
  DataChunk *sample = new DataChunkFlat(INPUT_SIZE_GOES_HERE, 0);
  std::cout << "size: " << sample->get_1d().size() << std::endl;

  // Alternatively this is supported by the library
  // sample->read_from_file("./example/sample_mnist.dat");
  // sample->generate_blank_set();

  KerasModel m("example/model_dumped.nnet", true);
  m.compute_output(sample);
  delete sample;

  return 0;
}
