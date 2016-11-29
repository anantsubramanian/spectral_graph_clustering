
run_test: test
	mpirun -np 4 test

test: test.cpp graph.cpp utils.cpp headers/graph.hpp headers/utils.hpp
	mpic++ test.cpp graph.cpp utils.cpp -o test

clean:
	rm test 2>/dev/null >/dev/null

