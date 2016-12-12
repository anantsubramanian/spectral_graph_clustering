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

void run_lanczos_csr ( char *input_file, int M, DATATYPE **alpha_out,
                       DATATYPE **beta_out, DATATYPE ***v_out, int rank )
{
  double start_time = MPI_Wtime();

  // Load distributed graph from edgelist file, and construct
  // the unnormalized Laplacian
  ifstream fin(input_file);
  distributed_graph_csr<DATATYPE> *input_graph =
    create_csr_from_edgelist_file<DATATYPE>(fin);
  input_graph -> construct_unnormalized_laplacian();
  input_graph -> free_adjacency_matrix();

  double data_load_time = MPI_Wtime();
  if ( rank == MASTER )
    cerr << "Data load time: " << data_load_time - start_time << "s\n";

  // Get distributed Laplacian CSR submatrices
  DATATYPE *data = input_graph -> get_lap_A();
  int *row_ptr = input_graph -> get_lap_row_ptr();
  int *col_idx = input_graph -> get_lap_col_idx();
  int N = input_graph -> get_N();
  int rows_in_node = input_graph -> get_rows_in_node();
  int rows_per_node = input_graph -> get_rows_per_node();
  int local_start_index = input_graph -> get_row_start_index();

  DATATYPE *alpha, *beta;
  DATATYPE **v = new DATATYPE*[M];

  double lanczos_start_time = MPI_Wtime();
  // Replace with appropriate call if using different matrix distribution
  lanczos_csr <DATATYPE> ( data, row_ptr, col_idx, N, rows_in_node, rows_per_node,
                           local_start_index, M, MPIDATATYPE, &alpha, &beta, &v );
  double lanczos_end_time = MPI_Wtime();

  if ( rank == MASTER )
    cerr << "Lanczos run time: " << lanczos_end_time - lanczos_start_time << "s\n";

  *alpha_out = alpha;
  *beta_out = beta;
  *v_out = v;
}

void run_lanczos_csc ( char *input_file, int M, DATATYPE **alpha_out,
                       DATATYPE **beta_out, DATATYPE ***v_out, int rank )
{
  double start_time = MPI_Wtime();

  // Load distributed graph from edgelist file, and construct
  // the unnormalized Laplacian
  ifstream fin(input_file);
  distributed_graph_csc<DATATYPE> *input_graph =
    create_csc_from_edgelist_file<DATATYPE>(fin);
  input_graph -> construct_unnormalized_laplacian();
  input_graph -> free_adjacency_matrix();

  double data_load_time = MPI_Wtime();
  if ( rank == MASTER )
    cerr << "Data load time: " << data_load_time - start_time << "s\n";

  // Get distributed Laplacian CSR submatrices
  DATATYPE *data = input_graph -> get_lap_A();
  int *col_ptr = input_graph -> get_lap_col_ptr();
  int *row_idx = input_graph -> get_lap_row_idx();
  int N = input_graph -> get_N();
  int cols_in_node = input_graph -> get_cols_in_node();
  int cols_per_node = input_graph -> get_cols_per_node();
  int local_start_index = input_graph -> get_col_start_index();

  DATATYPE *alpha, *beta;
  DATATYPE **v = new DATATYPE*[M];

  double lanczos_start_time = MPI_Wtime();
  // Replace with appropriate call if using different matrix distribution
  lanczos_csr <DATATYPE> ( data, col_ptr, row_idx, N, cols_in_node, cols_per_node,
                           local_start_index, M, MPIDATATYPE, &alpha, &beta, &v );
  double lanczos_end_time = MPI_Wtime();

  if ( rank == MASTER )
    cerr << "Lanczos run time: " << lanczos_end_time - lanczos_start_time << "s\n";

  *alpha_out = alpha;
  *beta_out = beta;
  *v_out = v;
}


void print_usage()
{
  cerr << "Usage: mpirun <options> spectral_clustering <edgelistfile> <mode> <M>\n";
  cerr << "\n<mode>:\n\t0 MPI Only CSR\n";
  cerr << "\t1 MPI Only CSC\n";
  cerr << "\t2 MPI + CUDA CSR\n";
  cerr << "\t3 MPI + CUDA CSC\n\n";
  cerr << "<M>: Number of required eigenvectors\n\n";
}

int main ( int argc, char *argv[] )
{
  MPI_Init(&argc, &argv);

  int rank, num_tasks;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);

  if ( argc != 4 )
  {
    if ( rank == MASTER )
      print_usage();
    MPI_Finalize();
    return 1;
  }

  DATATYPE *alpha, *beta;
  DATATYPE **v;

  int M = atoi(argv[3]);

  switch ( argv[2][0] )
  {
    case '0': run_lanczos_csr ( argv[1], M, &alpha, &beta, &v, rank );
              break;
    case '1': run_lanczos_csc ( argv[1], M, &alpha, &beta, &v, rank );
              break;
    default: if ( rank == MASTER ) print_usage();
             MPI_Finalize();
             return 1;
  }

  if ( rank == MASTER )
  {
    double *eigen_values;
    double *eigen_vectors;
    int num_found;

    double start_lapack = MPI_Wtime();
    lapack_eigen ( alpha, beta, M, num_found, &eigen_values, &eigen_vectors );
    //qr_eigen ( alpha, beta, M );
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

