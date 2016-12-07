
all: run_test_construction run_lanczos_mpi_only

run_lanczos_mpi_only: lanczos_mpi_only_1 lanczos_mpi_only_4

lanczos_mpi_only_1: lanczos_mpi_only
	mpirun -np 1 lanczos_mpi_only > lanczos_output_1

lanczos_mpi_only_4: lanczos_mpi_only
	mpirun -np 4 lanczos_mpi_only > lanczos_output_4

lanczos_mpi_only: lanczos_mpi_only.cpp graph.cpp utils.cpp headers/graph.hpp headers/utils.hpp
	mpic++ lanczos_mpi_only.cpp graph.cpp utils.cpp -o lanczos_mpi_only

run_test_construction: test_construction
	mpirun -np 4 test_construction

test_construction: test_construction.cpp graph.cpp utils.cpp headers/graph.hpp headers/utils.hpp
	mpic++ test_construction.cpp graph.cpp utils.cpp -o test_construction

clean:
	rm lanczos_mpi_only test_construction lanczos_output_1 lanczos_output_4 2>/dev/null >/dev/null

