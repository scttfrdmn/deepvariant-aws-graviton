# Build against a oneDNN v2.7.3 tree (ONEDNN) and TF 2.13.1's Eigen headers (TFINC).
# -mavx without FMA: TF's x86 pip wheel is built for AVX, so Eigen's rsqrt matches.
ONEDNN ?= $(HOME)/dv-dev/ox/p2/oneDNN
TFINC ?= $(HOME)/dv-dev/ox/tf2.13.1/lib/python3.11/site-packages/tensorflow/include
cvdnn: cvdnn.cpp
	g++ -O2 -mavx -fopenmp -std=c++17 -I$(ONEDNN)/include -I$(ONEDNN)/build/include -I$(TFINC) -o $@ $< \
	  -L$(ONEDNN)/build/src -ldnnl -Wl,-rpath,$(ONEDNN)/build/src
