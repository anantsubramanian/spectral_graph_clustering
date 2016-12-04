default:
	mpic++ -c test_construction.cpp -o test_construction.o
	mpic++ -c graph.cpp -o graph.o
	mpic++ -c utils.cpp -o utils.o 
	nvcc -arch=sm_20 -c multiply.cu -o multiply.o
	mpic++ test_construction.o graph.o utils.o multiply.o -lcudart -L/usr/local/cuda/lib64/ -o test_construction -lcublas

run_test_construction: test_construction
	mpirun -np 4 test_construction

test_construction: test_construction.cpp graph.cpp utils.cpp headers/graph.hpp headers/utils.hpp
	mpic++ test_construction.cpp graph.cpp utils.cpp -o test_construction

clean:
	rm test_construction 2>/dev/null >/dev/null
	
clear:
	rm latedays.qsub.*

