#include <iostream>
#include <fstream>
#include <vector>
#include "mpi.h"

#include "headers/utils.hpp"
#ifndef __GRAPH
  #include "headers/graph.hpp"
#endif

using namespace std;

// func declared in multipy.cu
void call_me_maybe(float *, float *, int, int);

int main ( int argc, char* argv[] )
{
  MPI_Init(&argc, &argv);
  ifstream fin("../data/as20000102.txt");
  distributed_graph<double> *test = create_from_edgelist_file<double>(fin);
  test -> construct_unnormalized_laplacian();
  test -> free_adjacency_matrix();

  // Get distributed Laplacian CSR submatrices
  double *data = test -> get_lap_A();
  int *row_ptr = test -> get_lap_row_ptr();
  int *col_idx = test -> get_lap_col_idx();

  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  if (myrank ==0) {
    int N = 5, M = 5;
    float *vecA, *vecB;

    vecA = new float[N]; 
    vecB = new float[N];

    for (int j = 0; j < N; j++) {
      vecA[j] = j;
      vecB[j] = 2;
    }

    call_me_maybe(vecA, vecB, N, M);
  }

  MPI_Finalize();
  return 0;
}
