DELETE_THIS="delete_this.o"
g++ -std=c++11 -c -fPIC keras2cpp/keras_model.cc -o $DELETE_THIS
g++ -shared $DELETE_THIS -o libneural_net.so
rm $DELETE_THIS
