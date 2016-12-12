default:
	mpic++ -c test_construction.cpp -o test_construction.o
	mpic++ -c graph.cpp -o graph.o
	mpic++ -c utils.cpp -o utils.o
	nvcc -arch=sm_20 -c multiply.cu -o multiply.o
	mpic++ test_construction.o graph.o utils.o multiply.o -lcudart -L/usr/local/cuda/lib64/ -o test_construction -lcublas

all: run_test_construction run_spectral_clustering # run_lanczos_mpi_cuda_csr

run_spectral_clustering: spectral_clustering
	mpirun -np 1 spectral_clustering ../data/as20000102_row.txt 0 200 > \
		spectral_output_csr_1
	mpirun -np 8 spectral_clustering ../data/as20000102_row.txt 0 200 > \
		spectral_output_csr_8
	mpirun -np 1 spectral_clustering ../data/as20000102_col.txt 1 200 > \
		spectral_output_csc_1
	mpirun -np 8 spectral_clustering ../data/as20000102_col.txt 1 200 > \
		spectral_output_csc_8
	mpirun -np 1 spectral_clustering ../data/Gowalla_200k_nodes_col.txt 1 1500 > \
		spectral_output_csc_big_1
	mpirun -np 8 spectral_clustering ../data/Gowalla_200k_nodes_col.txt 1 1500 > \
		spectral_output_csc_big_8
	mpirun -np 1 spectral_clustering ../data/Gowalla_200k_nodes_row.txt 0 1500 > \
		spectral_output_csr_big_1
	mpirun -np 8 spectral_clustering ../data/Gowalla_200k_nodes_row.txt 0 1500 > \
		spectral_output_csr_big_8

#spectral_clustering: spectral_clustering.cpp lanczos.cpp headers/lanczos.hpp \
	#headers/graph.hpp graph.cpp headers/utils.hpp utils.cpp
	#mpic++ spectral_clustering.cpp utils.cpp graph.cpp lanczos.cpp -o spectral_clustering -L/usr/lib64/atlas-sse3/ -llapack -lcblas -lgfortran

run_spectral_clustering_cuda: spectral_clustering
	mpirun -np 1 spectral_clustering_cuda ../data/as20000102_row.txt 2 200 > \
		spectral_output_csr_1

spectral_clustering: spectral_clustering.cpp lanczos.cpp headers/lanczos.hpp lanczos_cuda.cpp headers/lanczos_cuda.hpp \
	headers/graph.hpp graph.cpp headers/utils.hpp utils.cpp headers/cu_utils.h cu_utils.cu
	nvcc spectral_clustering.cpp utils.cpp graph.cpp lanczos.cpp lanczos_cuda.cpp cu_utils.cu -o \
		spectral_clustering -L /usr/lib64/atlas-sse3/ -llapack -lcblas -lgfortran -L/opt/openmpi/lib \
		-lmpi_cxx -lmpi -libverbs -ldat -lrt -lnsl -lutil -lm -ldl -lm -lrt -lnsl -lutil -lm -ldl \
		-lcublas -lcusparse -I/opt/openmpi/include

run_lanczos_mpi_cuda_csr: lanczos_mpi_cuda_csr
	mpirun -np 1 lanczos_mpi_cuda_csr
	#mpirun -np 8 lanczos_mpi_cuda_csr

lanczos_mpi_cuda_csr: lanczos_mpi_cuda_csr.cpp graph.cpp utils.cpp headers/graph.hpp headers/utils.hpp
	nvcc lanczos_mpi_cuda_csr.cpp graph.cpp utils.cpp cu_utils.cu -L/opt/openmpi/lib -lmpi_cxx -lmpi -libverbs -ldat -lrt -lnsl -lutil -lm -ldl -lm -lrt -lnsl -lutil -lm -ldl -lcublas -lcusparse -I/opt/openmpi/include -o lanczos_mpi_cuda_csr

run_test_construction: test_construction
	mpirun -np 8 test_construction

test_construction: test_construction.cpp graph.cpp utils.cpp headers/graph.hpp headers/utils.hpp
	mpic++ test_construction.cpp graph.cpp utils.cpp -o test_construction

clean:
	rm spectral_clustering lanczos_mpi_cuda_csr test_construction spectral_output_* \
		latedays.qsub.* *.o

