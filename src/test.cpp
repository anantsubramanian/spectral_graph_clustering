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
  ifstream fin("../data/as20000102.txt");
  distributed_graph<double> *test = create_from_edgelist_file<double>(fin);
  test -> construct_unnormalized_laplacian();
  MPI_Finalize();
  return 0;
}
