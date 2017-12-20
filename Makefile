# Warnings
WFLAGS	:= -Wall -Wextra -Wsign-conversion -Wsign-compare

# Optimization and architecture
OPT		:= -O3
ARCH   	:= -march=native

# Language standard
CCSTD	:= -std=c99
CXXSTD	:= -std=c++11

# Linker options
LDOPT 	:= $(OPT)
LDFLAGS :=
BIN = "/usr/local/gcc/6.4.0/bin/gcc"
.DEFAULT_GOAL := all

.PHONY: debug
debug : OPT  := -O0 -g -fno-omit-frame-pointer -fsanitize=address
debug : LDFLAGS := -fsanitize=address
debug : ARCH :=
debug : $(EXEC)

all : problem1A problem1B problem2A problem2B


problem2A: Sobel.cu 2Dconvolution.h
	module load cuda;nvcc -o Sobel $(OPT) Sobel.cu -ccbin $(BIN)



# TODO: add targets for building executables

.PHONY: clean
clean:
	rm -f problem1A problem1B problem2A problem2B