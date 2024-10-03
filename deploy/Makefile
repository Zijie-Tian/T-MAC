ROOT_DIR:=$(shell dirname $(dir $(realpath $(firstword $(MAKEFILE_LIST)))))
TVM_ROOT:=${ROOT_DIR}/3rdparty/tvm
CXX:=clang++

PKG_CFLAGS = -march=armv8.2-a+fp16 -std=c++17 -O0 -fno-inline -fPIC\
	-I${TVM_ROOT}/include\
	-I${TVM_ROOT}/3rdparty/dmlc-core/include\
	-I${TVM_ROOT}/3rdparty/dlpack/include\
	-I${ROOT_DIR}/include\
	-I${ROOT_DIR}/install/include\
	-DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>

PKG_LDFLAGS = -L${TVM_ROOT}/build -ldl -pthread

.PHONY: clean all

all: build/kernels.dll build/benchmark

build/kernels.dll: compile.py
	@echo "ROOT_DIR is $(ROOT_DIR)"
	@mkdir -p $(@D)
	
	# python compile.py -t -o tuned -da -d jetson -nt 1 -tb -gc -gs 32 -ags 64 -md /data/hf_models/bitnet_b1_58-3B
	$(CXX) -shared -o $@ tuned/kernels.cc $(PKG_CFLAGS) $(PKG_LDFLAGS)

	@cp ./tuned/kernels.h ../include/t-mac/

build/benchmark: benchmark.cc
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -o $@ $< -ltvm_runtime $(PKG_LDFLAGS)

clean:
	rm -rf build
