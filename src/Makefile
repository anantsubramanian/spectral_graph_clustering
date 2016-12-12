# Makefile to compile and test code locally, as well as on the latedays cluster
cc=nvcc
mpicc=mpic++
lapackL=/usr/lib64/atlas-sse3/
mpiL=/opt/openmpi/lib
mpiI=/opt/openmpi/include
cpuflags=-llapack -lcblas -lgfortran
extraflags=-lmpi_cxx -lmpi -libverbs -ldat -lrt -lnsl -lutil \
					 -lm -ldl -lrt -lnsl
gpuflags=-lcublas -lcusparse

all: run_test_construction run_spectral_clustering run_spectral_clustering_cuda

run_spectral_clustering: spectral_clustering
	mpirun -np 1 spectral_clustering ../data/as20000102_row.txt 0 200 > \
		spectral_output_csr_1
	mpirun -np 8 spectral_clustering ../data/as20000102_row.txt 0 200 > \
		spectral_output_csr_8
	mpirun -np 1 spectral_clustering ../data/as20000102_col.txt 1 200 > \
		spectral_output_csc_1
	mpirun -np 8 spectral_clustering ../data/as20000102_col.txt 1 200 > \
		spectral_output_csc_8
	mpirun -np 1 spectral_clustering ../data/Gowalla_200k_nodes_col.txt 1 500 > \
		spectral_output_csc_big_1
	mpirun -np 8 spectral_clustering ../data/Gowalla_200k_nodes_col.txt 1 500 > \
		spectral_output_csc_big_8
	mpirun -np 1 spectral_clustering ../data/Gowalla_200k_nodes_row.txt 0 500 > \
		spectral_output_csr_big_1
	mpirun -np 8 spectral_clustering ../data/Gowalla_200k_nodes_row.txt 0 500 > \
		spectral_output_csr_big_8

run_spectral_clustering_cuda: spectral_clustering
	mpirun -np 1 spectral_clustering_cuda ../data/as20000102_row.txt 2 200 > \
		spectral_output_cuda_csr_1

spectral_clustering: spectral_clustering.cpp lanczos.cpp headers/lanczos.hpp \
	lanczos_cuda.cpp headers/lanczos_cuda.hpp headers/graph.hpp graph.cpp \
	headers/utils.hpp utils.cpp headers/cu_utils.h cu_utils.cu
	$(cc) spectral_clustering.cpp utils.cpp graph.cpp lanczos.cpp lanczos_cuda.cpp cu_utils.cu -o \
		spectral_clustering -L $(lapackL) -L $(mpiL) -I $(mpiI) $(cpuflags) $(gpuflags) $(extraflags) \
		$(gpuflags)

run_test_construction: test_construction
	mpirun -np 8 test_construction

test_construction: test_construction.cpp graph.cpp utils.cpp headers/graph.hpp headers/utils.hpp
	$(mpicc) test_construction.cpp graph.cpp utils.cpp -o test_construction $(cpuflags)

clean:
	rm spectral_clustering test_construction spectral_output_* latedays.qsub.* *.o

