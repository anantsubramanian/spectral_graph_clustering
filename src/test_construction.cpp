#include <iostream>
#include <fstream>
#include <vector>
#include "mpi.h"

#include "headers/utils.hpp"
#ifndef __GRAPH
  #include "headers/graph.hpp"
#endif

using namespace std;

int main ( int argc, char* argv[] )
{
  MPI_Init(&argc, &argv);

  // Test CSR distributed graph construction
  ifstream fin("../data/as20000102_row.txt");
  distributed_graph_csr<double> *test = create_csr_from_edgelist_file<double>(fin);
  test -> construct_unnormalized_laplacian();
  test -> free_adjacency_matrix();
  fin.close();

  // Get distributed Laplacian CSR submatrices
  double *data = test -> get_lap_A();
  int *row_ptr = test -> get_lap_row_ptr();
  int *col_idx = test -> get_lap_col_idx();

  // Test CSC distributed graph construction
  fin.open("../data/as20000102_col.txt");
  distributed_graph_csc<double> *test_csc = create_csc_from_edgelist_file<double>(fin);
  test_csc -> construct_unnormalized_laplacian();
  test_csc -> free_adjacency_matrix();
  fin.close();

  // Get distributed Laplacian CSC submatrices
  double *data_csc = test_csc -> get_lap_A();
  int *col_ptr = test_csc -> get_lap_col_ptr();
  int *row_idx = test_csc -> get_lap_row_idx();

  MPI_Finalize();
  return 0;
}

