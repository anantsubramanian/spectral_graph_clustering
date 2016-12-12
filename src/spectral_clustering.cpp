#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include "mpi.h"
#include "headers/utils.hpp"
#ifndef __GRAPH
  #include "headers/graph.hpp"
#endif
#include "headers/lanczos.hpp"

// Working with single precision arithmetic can be risky.
#define DATATYPE double
#define MPIDATATYPE MPI_DOUBLE
#define MASTER 0

using namespace std;

/**
 * Comparator function to sort the output eigen values
 */
bool abs_smaller ( const DATATYPE a, const DATATYPE b )
{
  return fabs(a) < fabs(b);
}

int main ( int argc, char *argv[] )
{
  MPI_Init(&argc, &argv);

  int rank, num_tasks;
  double start_time = MPI_Wtime();

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);

  // Load distributed graph from edgelist file, and constructed
  // the unnormalized Laplacian
  ifstream fin("../data/as20000102_row.txt");
  distributed_graph_csr<DATATYPE> *inputGraph =
    create_csr_from_edgelist_file<DATATYPE>(fin);
  inputGraph -> construct_unnormalized_laplacian();
  inputGraph -> free_adjacency_matrix();

  double data_load_time = MPI_Wtime();
  if ( rank == MASTER )
    cerr << "Data load time: " << data_load_time - start_time << "s\n";

  // Get distributed Laplacian CSR submatrices
  DATATYPE *data = inputGraph -> get_lap_A();
  int *row_ptr = inputGraph -> get_lap_row_ptr();
  int *col_idx = inputGraph -> get_lap_col_idx();
  int N = inputGraph -> get_N();
  int rows_in_node = inputGraph -> get_rows_in_node();
  int rows_per_node = inputGraph -> get_rows_per_node();
  int local_start_index = inputGraph -> get_row_start_index();

  // Number of eigen values requested
  int M = 50;

  DATATYPE *alpha, *beta;
  DATATYPE **v = new DATATYPE*[M];

  double lanczos_start_time = MPI_Wtime();
  // Replace with appropriate call if using different matrix distribution
  lanczos_csr <DATATYPE> ( data, row_ptr, col_idx, N, rows_in_node, rows_per_node,
                           local_start_index, M, MPIDATATYPE, &alpha, &beta, &v );
  double lanczos_end_time = MPI_Wtime();

  if ( rank == MASTER )
    cerr << "Lanczos run time: " << lanczos_end_time - lanczos_start_time << "s\n";

  double *eigen_values;
  double *eigen_vectors;
  int num_found;

  if ( rank == MASTER )
  {
    double start_lapack = MPI_Wtime();
    lapack_eigen ( alpha, beta, M, num_found, &eigen_values, &eigen_vectors );
    //qr_eigen ( alpha, beta, M );
    //sort ( alpha_2, alpha_2+M, abs_smaller );
    double end_lapack = MPI_Wtime();

    cerr << "LAPACK run time: " << end_lapack - start_lapack << "s\n\n";
    cerr << "Found " << num_found << " eigenvalues and eigenvectors.\n";
    for ( int i = 0; i < M; i++ )
      cout << eigen_values[i] << "\n";

    for ( int i = 0; i < M; i++ )
    {
      for ( int j = 0; j < M; j++ )
        cout << eigen_vectors[i * M + j] << " ";
      cout << "\n";
    }

    delete eigen_values;
    delete eigen_vectors;
  }

  delete alpha;
  delete beta;
  delete v;

  MPI_Finalize();
  return 0;
}

