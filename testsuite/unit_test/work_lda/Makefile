NVCC=nvcc
INCLUDES=../../../src
INCLUDES2=../../../
LIBS=../../../objdir
CFLAGS_GPU=-I. -I${INCLUDES} -I${INCLUDES2} --ptxas-options=-v
CLIBS_GPU=-L${LIBS} -lxc_cuda -lm

all: test_functionals

test_functionals: test_functionals.cu
	nvcc -o test_functionals test_functionals.cu $(CFLAGS_GPU) $(CLIBS_GPU)

clean:
	rm -f *.o test_functionals
