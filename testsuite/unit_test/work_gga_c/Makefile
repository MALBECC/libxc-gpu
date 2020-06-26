NVCC=nvcc
INCLUDES=/home/gonzalo/gonzalo/codes/libxc-gpu/include
LIBS=/home/gonzalo/gonzalo/codes/libxc-gpu/lib
CFLAGS_GPU=-I. -I${INCLUDES} -I${INCLUDES2} --ptxas-options=-v
CLIBS_GPU=-L${LIBS} -lxc_cuda -lm

#CFLAGS_GPU=-I. -I${LIBXC_INCLUDES} --ptxas-options=-v
#CLIBS_GPU=-L${LIBXC_LIBS} -lxc_cuda -lm

all: test_work_gga_c

test_work_gga_c: test_work_gga_c.cu
	nvcc -o test_work_gga_c test_work_gga_c.cu $(CFLAGS_GPU) $(CLIBS_GPU)

clean:
	rm -f *.o test_work_gga_c

