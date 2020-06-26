NVCC=nvcc
INCLUDES=../../../src
INCLUDES2=../../../
LIBS=../../../objdir
CFLAGS_GPU=-I. -I${INCLUDES} -I${INCLUDES2} --ptxas-options=-v
CLIBS_GPU=-L${LIBS} -lxc_cuda -lm

all: test_work_gga_x

test_work_gga_x: test_work_gga_x.cu
	nvcc -o test_work_gga_x test_work_gga_x.cu $(CFLAGS_GPU) $(CLIBS_GPU)

clean:
	rm -f *.o test_work_gga_x

