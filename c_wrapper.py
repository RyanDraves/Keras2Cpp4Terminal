import ctypes
import sys

# Having troubles here? Me too. Feel free to make a pull request if you know how to make this work reliably...
lib_net = ctypes.cdll.LoadLibrary(sys.path[0] + '/libneural_net.so')

class NeuralNet(object):
    def __init__(self, nnet_filename, input_size, output_size, verbose = False):
        self.INPUT_SIZE = input_size
        self.OUTPUT_SIZE = output_size

        lib_net.NeuralNet_new.argtypes = [ctypes.c_char_p, ctypes.c_bool]
        lib_net.NeuralNet_new.restype = ctypes.c_void_p

        lib_net.NeuralNet_compute_output.argtypes = [ctypes.c_void_p, ctypes.c_float * self.INPUT_SIZE, ctypes.c_int, ctypes.c_int]
        lib_net.NeuralNet_compute_output.restype = ctypes.POINTER(ctypes.c_float * self.OUTPUT_SIZE)

        lib_net.NeuralNet_free_memory.argtypes = []
        lib_net.NeuralNet_free_memory.restype = ctypes.c_void_p

        filepath = sys.path[0] + "/" + nnet_filename
        c_s = ctypes.c_char_p(filepath.encode('utf-8'))
        self.obj = lib_net.NeuralNet_new(c_s, verbose)
    
    def compute_output(self, input_data):
        array_input_data = (ctypes.c_float * len(input_data))(*input_data)
        array = lib_net.NeuralNet_compute_output(self.obj, array_input_data, self.INPUT_SIZE, self.OUTPUT_SIZE)
        dereference = array.contents
        prediction = []
        for i in range(self.OUTPUT_SIZE):
            prediction.append(dereference[i])
        self._free_memory()
        return prediction
    
    # "But I want to tell it what array to delete"
    # BuT i WaNt tO tELl iT wHaT aRraY tO DeLetE
    def _free_memory(self):
        lib_net.NeuralNet_free_memory(self.obj)
