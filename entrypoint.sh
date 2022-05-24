#c++ -O3 -Wall -shared -std=c++17 -fPIC `python3 -m pybind11 --includes` lib/kernels.cpp -o kernels
c++ -O3 -Wall -std=c++17 -fPIC `python3 -m pybind11 --includes` lib/kernels.cpp -o kernels
