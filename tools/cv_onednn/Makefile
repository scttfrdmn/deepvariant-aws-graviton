# Build against a oneDNN v2.7.3 tree (ONEDNN) and TF 2.13.1's Eigen headers (TFINC).
# -mavx without FMA: TF's x86 pip wheel is built for AVX, so Eigen's rsqrt matches.
ONEDNN ?= $(HOME)/dv-dev/ox/p2/oneDNN
TFINC ?= $(HOME)/dv-dev/ox/tf2.13.1/lib/python3.11/site-packages/tensorflow/include
cvdnn: cvdnn.cpp
	g++ -O2 -mavx -fopenmp -std=c++17 -I$(ONEDNN)/include -I$(ONEDNN)/build/include -I$(TFINC) -o $@ $< \
	  -L$(ONEDNN)/build/src -ldnnl -Wl,-rpath,$(ONEDNN)/build/src

# Same executor against a v2.7.3 tree with onednn-v2.7.3-brgconv-channel-slice-dst.patch applied
# (brgconv / brgconv_1x1 store into a channel slice of a wider nhwc tensor: in-place concat).
ONEDNN_LD ?= $(HOME)/dv-dev/ox/p2/oneDNN-ld
cvdnn-ld: cvdnn.cpp
	g++ -O2 -mavx -fopenmp -std=c++17 -I$(ONEDNN_LD)/include -I$(ONEDNN_LD)/build/include -I$(TFINC) -o $@ $< \
	  -L$(ONEDNN_LD)/build/src -ldnnl -Wl,-rpath,$(ONEDNN_LD)/build/src

ldtest: ldtest.cpp
	g++ -O2 -fopenmp -std=c++17 -I$(ONEDNN_LD)/include -I$(ONEDNN_LD)/build/include -o $@ $< \
	  -L$(ONEDNN_LD)/build/src -ldnnl -Wl,-rpath,$(ONEDNN_LD)/build/src
