
all: run_test_construction run_lanczos_mpi_only_csr

run_lanczos_mpi_only_csr: lanczos_mpi_only_csr_1 lanczos_mpi_only_csr_4

lanczos_mpi_only_csr_1: lanczos_mpi_only_csr
	mpirun -np 1 lanczos_mpi_only_csr > lanczos_output_csr_1

lanczos_mpi_only_csr_4: lanczos_mpi_only_csr
	mpirun -np 4 lanczos_mpi_only_csr > lanczos_output_csr_4

lanczos_mpi_only_csr: lanczos_mpi_only_csr.cpp graph.cpp utils.cpp headers/graph.hpp headers/utils.hpp
	mpic++ lanczos_mpi_only_csr.cpp graph.cpp utils.cpp -o lanczos_mpi_only_csr

run_test_construction: test_construction
	mpirun -np 4 test_construction

test_construction: test_construction.cpp graph.cpp utils.cpp headers/graph.hpp headers/utils.hpp
	mpic++ test_construction.cpp graph.cpp utils.cpp -o test_construction

clean:
	rm lanczos_mpi_only_csr test_construction lanczos_output_csr_1 lanczos_output_csr_4 2>/dev/null >/dev/null

