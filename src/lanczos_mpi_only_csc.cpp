// Implementation of the Lanczos algorithm with partial re-orthogonalization
// using the method described in H. D. Simon, The lanczos algorithm with
// partial reorthogonalization, Mathematics of Computation, 42(165):pp,
// 115-142, 1984.

#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "mpi.h"
#include "headers/utils.hpp"
#ifndef __GRAPH
  #include "headers/graph.hpp"
#endif

#define DATATYPE float
#define MPIDATATYPE MPI_FLOAT
#define MASTER 0

using namespace std;

/**
 * Generate a normal random number using the Box-Muller
 * transform.
 */
DATATYPE random_normal ( DATATYPE mu, DATATYPE sigma )
{
  DATATYPE U = rand() / (DATATYPE) RAND_MAX;
  DATATYPE V = rand() / (DATATYPE) RAND_MAX;

  DATATYPE X = (DATATYPE) (sqrt(-2.0 * log(U)) * cos(2 * M_PI * V));
  return mu + X * sigma;
}

int main ( int argc, char* argv[] )
{
  MPI_Init(&argc, &argv);
  
  int rank, num_tasks;

  double start_time = MPI_Wtime();

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);

  ifstream fin("../data/as20000102_col.txt");
  distributed_graph_csc<DATATYPE> *inputGraph =
    create_csc_from_edgelist_file<DATATYPE>(fin);
  inputGraph -> construct_unnormalized_laplacian();
  inputGraph -> free_adjacency_matrix();

  double data_load_time = MPI_Wtime();

  // Get distributed Laplacian CSC submatrices
  DATATYPE *data = inputGraph -> get_lap_A();
  int *col_ptr = inputGraph -> get_lap_col_ptr();
  int *row_idx = inputGraph -> get_lap_row_idx();
  int N = inputGraph -> get_N();
  int cols_in_node = inputGraph -> get_cols_in_node();
  int cols_per_node = inputGraph -> get_cols_per_node();
  int local_start_index = inputGraph -> get_col_start_index();

  // Determine the square root of the machine precision for re-orthogonalization
  DATATYPE epsilon = numeric_limits<DATATYPE>::epsilon();
  DATATYPE sqrteps = sqrt(epsilon);
  DATATYPE eta = pow(epsilon, 0.75);

  // The distributed intermediate vectors v
  int vsize = cols_per_node;
  DATATYPE *v_data = new DATATYPE[(N+2) * vsize];
  DATATYPE **v = new DATATYPE*[N+2];
  for ( int i = 0; i < N+2; i++ )
    v[i] = v_data + i*vsize;
  
  // The values in the tri-diagonal matrix, alphas and betas
  DATATYPE *alpha = new DATATYPE[N+1];
  DATATYPE *beta = new DATATYPE[N+1];

  unsigned int seedval;
  if ( rank == MASTER )
    seedval = time(NULL);
  MPI_Bcast(&seedval, 1, MPI_UNSIGNED, MASTER, MPI_COMM_WORLD);

  srand(seedval);

  // Generate the initial normalized vectors v1, and the 0 vector v0
  DATATYPE sum = 0.;
  DATATYPE sum_local = 0.;
  for ( int i = 0; i < cols_in_node; i++ )
  {
    v[0][i] = 0.;
    DATATYPE temp = rand() / (DATATYPE) RAND_MAX;
    v[1][i] = temp;
    sum_local += temp*temp;
  }
  MPI_Allreduce(&sum_local, &sum, 1, MPIDATATYPE, MPI_SUM, MPI_COMM_WORLD);
  sum = sqrt(sum);

  for ( int i = 0; i < cols_in_node; i++ )
    v[1][i] /= sum;

  beta[1] = 0;
  
  // Scratch array for partial results
  DATATYPE *scratch = new DATATYPE[N];
  DATATYPE *omega_data = new DATATYPE[(N+1) * (N+1)];
  DATATYPE **omega = new DATATYPE*[N+1];
  for ( int i = 0; i < N+1; i++ )
    omega[i] = omega_data + i * (N+1);

  omega[0][0] = 1.0;
  omega[1][1] = 1.0;

  vector<int> to_reorthogonalize;
  bool prev_reorthogonalized = false;

  double initialization_time = MPI_Wtime();

  // Re-seed pseudo random number generator, as the calls to random would have been
  // different for the last node if N % cols_per_node != 0
  if ( rank == MASTER )
    seedval = time(NULL);
  MPI_Bcast(&seedval, 1, MPI_UNSIGNED, MASTER, MPI_COMM_WORLD);
  srand(seedval);

  // Start main Lanczos loop to compute the tri-diagonal elements, alpha[i] and beta[i]
  for ( int j = 1; j < N; j++ )
  {
    // Compute the distributed vector v[j+1]
    sparse_csc_mdotv<DATATYPE>(data, col_ptr, row_idx, cols_in_node, N, v[j], N, scratch);
    
    // Reduce sum the corresponding parts of v[j+1] to each node
    for ( int k = 0; k < num_tasks; k++ )
      MPI_Reduce(scratch + k * cols_per_node, v[j+1], cols_per_node,
          MPIDATATYPE, MPI_SUM, k, MPI_COMM_WORLD);

    // Compute the alpha for this iteration
    // Reduce sum alphas of different nodes to obtain alpha
    DATATYPE alpha_local = dense_vdotv<DATATYPE>(v[j+1], cols_in_node, v[j]);
    MPI_Allreduce(&alpha_local, &alpha[j], 1, MPIDATATYPE, MPI_SUM, MPI_COMM_WORLD);

    // Orthogonalize against past 2 vectors v[j], v[j-1]
    daxpy<DATATYPE>(v[j+1], -alpha[j], v[j], v[j+1], cols_in_node);
    daxpy<DATATYPE>(v[j+1], -beta[j], v[j-1], v[j+1], cols_in_node);

    // Normalize v[j+1] and store normalization constant as beta[j+1]
    DATATYPE beta_local = 0.;
    for ( int k = 0; k < cols_in_node; k++ )
      beta_local += v[j+1][k]*v[j+1][k];
    MPI_Allreduce(&beta_local, &beta[j+1], 1, MPIDATATYPE, MPI_SUM, MPI_COMM_WORLD);
    beta[j+1] = sqrt(beta[j+1]);

    // Normalize local portion of the vector
    for ( int k = 0; k < cols_in_node; k++ )
      v[j+1][k] /= beta[j+1];

    // Check and perform necessary re-orthogonalization
    // psi is same for all k
    DATATYPE psi = epsilon * N * beta[2] / beta[j+1] * random_normal(0, 0.6);
    omega[j+1][j+1] = 1.0;
    omega[j+1][j] = psi;

    for ( int k = 1; k < j; k++ )
    {
      DATATYPE theta = epsilon * (beta[k+1] + beta[j+1]) * random_normal(0, 0.3);
      omega[j+1][k] = theta + (beta[k+1]*omega[j][k+1] + (alpha[k]-alpha[j])*omega[j][k]
          + beta[k]*omega[j][k-1] - beta[j]*omega[j-1][k]) / beta[j+1];
    }

    if ( !prev_reorthogonalized )
    {
      int k = 0;
      int low = 0, high = 0;
      while ( k <= j )
      {
        // Find an estimated omega[j+1][k] that exceeds threshold
        if ( fabs(omega[j+1][k]) > sqrteps )
        {
          to_reorthogonalize.push_back(k);
          low = k-1;
          // Search the eta neighbourhood of this omega
          while ( low >= high && fabs(omega[j+1][low]) > eta )
          {
            to_reorthogonalize.push_back(low);
            low--;
          }
          high = k+1;
          while ( high <= j && fabs(omega[j+1][high]) > eta )
          {
            to_reorthogonalize.push_back(high);
            high++;
          }
          k = high;
          prev_reorthogonalized = true;
        }
        k++;
      }
    }
    else
      prev_reorthogonalized = false;

    // Re-orthogonalize against these range of vectors, performing updates on
    // local parts only. Will synchronize at the end.
    for ( int k = 0; k < to_reorthogonalize.size(); k++ )
    {
      int index = to_reorthogonalize[k];
      DATATYPE temp;
      DATATYPE temp_local = dense_vdotv<DATATYPE>(v[j+1], cols_in_node, v[index]);
      MPI_Allreduce(&temp_local, &temp, 1, MPIDATATYPE, MPI_SUM, MPI_COMM_WORLD);
      daxpy<DATATYPE>(v[j+1], -temp, v[index], v[j+1], cols_in_node);
      // Update estimate after this vector has been re-normalized
      omega[j+1][index] = epsilon * random_normal(0, 1.5);
    }

    if ( to_reorthogonalize.size() > 0 )
    {
      // Need to re-normalize v[j+1] and store normalization constant as beta[j+1]
      beta_local = 0.;
      for ( int k = 0; k < cols_in_node; k++ )
        beta_local += v[j+1][k]*v[j+1][k];
      MPI_Allreduce(&beta_local, &beta[j+1], 1, MPIDATATYPE, MPI_SUM, MPI_COMM_WORLD);
      beta[j+1] = sqrt(beta[j+1]);

      // Normalize the local portion of the vector
      for ( int k = 0; k < cols_in_node; k++ )
        v[j+1][k] /= beta[j+1];
    }

    if ( !prev_reorthogonalized )
      to_reorthogonalize.clear();
    // End loop for j+1
  }

  // Compute the last remaining alpha[N]
  sparse_csc_mdotv<DATATYPE>(data, col_ptr, row_idx, cols_in_node, N, v[N], N, scratch);
  for ( int k = 0; k < num_tasks; k++ )
    MPI_Reduce(scratch + k * cols_per_node, v[N+1], cols_per_node,
        MPIDATATYPE, MPI_SUM, k, MPI_COMM_WORLD);

  DATATYPE alpha_local = dense_vdotv<DATATYPE>(v[N+1], cols_in_node, v[N]);
  MPI_Allreduce(&alpha_local, &alpha[N], 1, MPIDATATYPE, MPI_SUM, MPI_COMM_WORLD);

  double tri_diagonalization_time = MPI_Wtime();

  if ( rank == 0 )
    for ( int i = 1; i <= N; i++ )
      cout << alpha[i] << " " << beta[i] << "\n";

  cerr << "Data load time: " << data_load_time - start_time << "\n";
  cerr << "Initialization time: " << initialization_time - data_load_time << "\n";
  cerr << "Loop time: " << tri_diagonalization_time - initialization_time << "\n";

  delete v_data;
  delete v;
  delete alpha;
  delete beta;
  delete scratch;
  delete omega_data;
  delete omega;

  MPI_Finalize();
  return 0;
}

