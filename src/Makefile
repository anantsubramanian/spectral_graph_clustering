
run_test_construction: test_construction
	mpirun -np 4 test_construction

test_construction: test_construction.cpp graph.cpp utils.cpp headers/graph.hpp headers/utils.hpp
	mpic++ test_construction.cpp graph.cpp utils.cpp -o test_construction

clean:
	rm test_construction 2>/dev/null >/dev/null

