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
#include <algorithm>
#include "mpi.h"
#include "headers/utils.hpp"
#include "cublas_v2.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "cusparse.h"
#include "headers/cu_utils.h"
#ifndef __GRAPH
  #include "headers/graph.hpp"
#endif

// Working with single precision arithmetic can be risky.
#define MASTER 0
#define EPS 1e-9

// Uncomment to print debug statements, true orthogonality of intermediate vectors
// and estimates through the recurrence relation.
//#define DEBUG

using namespace std;

/**
 * Generate a normal random number using the Box-Muller
 * transform.
 */
template <typename T>
T random_normal ( T mu, T sigma )
{
  T U = rand() / (T) RAND_MAX;
  T V = rand() / (T) RAND_MAX;

  T X = (T) (sqrt(-2.0 * log(U)) * cos(2 * M_PI * V));
  return mu + X * sigma;
}

/**
 * Update orthogonality estimate recurrence.
 */
template <typename T>
void update_estimate_recurrence ( T *beta, T *alpha, T **omega, int j, T *scratch,
                                  T epsilon, T sqrteps, T **v, int rows_in_node,
                                  int local_start_index, int N )
{
  int jplus1 = (j+1)%3;
  int jminus1 = (j-1)%3;
  int jmod3 = j%3;
  // psi is same for all k
  T psi = epsilon * N * (beta[2] / beta[j+1]) * random_normal(0.0, 0.6);
  omega[jplus1][j+1] = 1.0;
  omega[jplus1][j] = psi;
  omega[jplus1][0] = 0.;

#ifdef DEBUG
  cerr << "beta[" << j+1 << "] = " << beta[j+1] << "\n";
#endif

  for ( int k = 1; k < j; k++ )
  {
    T theta = epsilon * (beta[k+1] + beta[j+1]) * random_normal(0.0, 0.3);
    omega[jplus1][k] =
      theta + (beta[k+1]*omega[jmod3][k+1] + (alpha[k]-alpha[j])*omega[jmod3][k]
      + beta[k]*omega[jmod3][k-1] - beta[j]*omega[jminus1][k]) / beta[j+1];
  }

#ifdef DEBUG
  for ( int k = 1; k < j; k++ )
  {
    cerr << "omega[" << j+1 <<"][" << k <<"] = " << omega[jplus1][k] << " ";
    cerr << dense_vdotv<T>(scratch, rows_in_node, v[k]+local_start_index) << " ";
    cerr << (fabs(omega[jplus1][k]) > sqrteps ? "True" : "False") << "\n";
  }
  cerr << "omega[" << j+1 <<"][" << j <<"] = " << omega[jplus1][j] << " ";
  cerr << dense_vdotv<T>(scratch, rows_in_node, v[j]+local_start_index) << " ";
  cerr << (fabs(omega[jplus1][j]) > sqrteps ? "True" : "False") << "\n";
#endif
}

/**
 * Find offending indices to re-orthogonalize against. Returns true if any offending
 * indices were found.
 */
template <typename T>
bool find_offending_indices ( int j, T **omega, T sqrteps, T eta,
                              vector<int> &to_reorthogonalize )
{
  int jplus1 = (j+1)%3;
  bool offending_present = false;
  int k = 1;
  int low = 1, high = 1;
  while ( k <= j )
  {
    // Find an estimated omega[j+1][k] that exceeds threshold
    if ( fabs(omega[jplus1][k]) >= sqrteps )
    {
      to_reorthogonalize.push_back(k);
      low = k-1;
      // Search the eta neighbourhood of this omega
      while ( low >= high && fabs(omega[jplus1][low]) >= eta )
      {
        to_reorthogonalize.push_back(low);
        low--;
      }
      high = k+1;
      while ( high <= j && fabs(omega[jplus1][high]) >= eta )
      {
        to_reorthogonalize.push_back(high);
        high++;
      }
      k = high;
      offending_present = true;
    }
    k++;
  }

#ifdef DEBUG
    cerr << "Re-orthogonalizing against: " << to_reorthogonalize.size() << "\n";
#endif

  return offending_present;
}

/**
 * Re-orthogonalize against provided vectors using Gram Schmidt orthogonalization.
 */
template <typename T>
void re_orthogonalize ( T **v, vector<int> &to_reorthogonalize, int local_start_index,
                        int rows_in_node, T **omega, T *scratch, T epsilon, T sqrteps,
                        int j, MPI_Datatype mpi_datatype, T *beta, int N, T *dot_prods,
                        T *dot_prods_reduced )
{
  // Re-orthogonalize against these range of vectors, performing updates on
  // local parts only. Will synchronize at the end.
  for ( int k = local_start_index; k < local_start_index + rows_in_node; k++ )
     v[j+1][k] = scratch[k-local_start_index];

  for ( int k = 0; k < to_reorthogonalize.size(); k++ )
  {
    int index = to_reorthogonalize[k];
    T temp_local = dense_vdotv<T>(scratch, rows_in_node, v[index]+local_start_index);
    dot_prods[k] = temp_local;
    // Update estimate after this vector has been re-normalized
    omega[(j+1)%3][index] = epsilon * random_normal(0.0, 1.5);
  }

  MPI_Allreduce(dot_prods, dot_prods_reduced,
      to_reorthogonalize.size(), mpi_datatype, MPI_SUM, MPI_COMM_WORLD);
  for ( int k = 0; k < to_reorthogonalize.size(); k++ )
  {
    int index = to_reorthogonalize[k];
    daxpy<T>(
        v[j+1]+local_start_index, -dot_prods_reduced[k], v[index]+local_start_index,
        v[j+1]+local_start_index, rows_in_node);
  }

  if ( to_reorthogonalize.size() > 0 )
  {
    // Need to re-normalize v[j+1] and store normalization constant as beta[j+1]
    beta[j+1] = 0;
    for ( int k = local_start_index; k < local_start_index + rows_in_node; k++ )
      beta[j+1] += v[j+1][k]*v[j+1][k];
    T res;
    MPI_Allreduce(&beta[j+1], &res, 1, mpi_datatype, MPI_SUM, MPI_COMM_WORLD);
    beta[j+1] = sqrt(res);

#ifdef DEBUG
    cerr << "beta[" << j+1 << "] is now = " << beta[j+1] << "\n";
#endif
  }

#ifdef DEBUG
  cerr << "After re-orthogonalization:\n";
  if ( to_reorthogonalize.size() > 0 )
    for ( int k = 1; k <= j; k++ )
    {
      cerr << "omega[" << j+1 <<"][" << k <<"] = " << omega[(j+1)%3][k] << " ";
      cerr << dense_vdotv<T>(v[j+1], N, v[k]) << " ";
      cerr << (fabs(omega[(j+1)%3][k]) > sqrteps ? "True" : "False") << "\n";
    }
#endif
}


/**
 * Run the Lanczos algorithm with partial re-orthogonalization on a distributed CSR
 * matrix.
 *
 * Inputs:
 *   data - Data array of CSR matrix
 *   row_ptr - Row pointers for CSR matrix
 *   col_idx - Column indices corresponding to 'data'
 *   N - Total number of rows in the matrix (global)
 *   rows_in_node - Number of rows present in the local node
 *   rows_per_node - Ideal number of rows per node (if it was exactly divisible)
 *   local_start_index - Row that row 0 of this matrix partition corresponds to
 *   M - Number of iterations to run the Lanczos algorithm for (size of tri-diagonal
 *       matrix)
 *   mpi_datatype - The MPI datatype corresponding to T (eg. MPI_DOUBLE)
 *   eta - Neighbourhood to re-orthogonalize in (0 to 1, 0 = none, 1 = full reorth.)
 *         (optional)
 *
 * Outputs:
 *   alpha_out - The diagonal elements of the tri-diagonal matrix
 *   beta_out - The off-diagonal elements of the tri-diagonal matrix
 *   v_out - The produced intermediate orthonormal Lanczos vectors (MxN size)
 */
template <typename T>
void lanczos_csr_cuda ( T *data, int *row_ptr, int *col_idx, int nnz, int N, int rows_in_node,
                   int rows_per_node, int local_start_index, int M,
                   MPI_Datatype mpi_datatype, T **alpha_out, T **beta_out,
                   T ***v_out, T eta )
{
  int rank, num_tasks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);

  // Determine the square root of the machine precision for re-orthogonalization
  T epsilon = numeric_limits<T>::epsilon();
  T sqrteps = sqrt(epsilon);
  eta = pow(epsilon, eta);

#ifdef DEBUG
  cerr << "sqrteps = " << sqrteps << "\n";
  cerr << "eta = " << eta << "\n";
#endif

  // The intermediate vectors v
  int vsize = rows_per_node * num_tasks;
  T *v_data = new T[(M+2) * vsize];
  T **v = new T*[M+2];
  for ( int i = 0; i < M+2; ++i )
    v[i] = v_data + i*vsize;

  // The values in the tri-diagonal matrix, alphas and betas
  T *alpha = new T[M+2];
  T *beta = new T[M+2];

  // Synchronize the seed
  unsigned int seedval;
  if ( rank == MASTER )
    seedval = time(NULL);
  MPI_Bcast(&seedval, 1, MPI_UNSIGNED, MASTER, MPI_COMM_WORLD);

  srand(seedval);

  // Generate the initial normalized vectors v1, and the 0 vector v0
  // TODO: Check which is faster, generate locally on each machine, or generate in
  // parallel and perform AllGather?
  // Generating locally for now. All of them are using the same seed to the
  // pseudo-random number generator. WILL ONLY WORK if all machines are using the
  // same library version.
  T sum = 0;
  for ( int i = 0; i < N; ++i )
  {
    v[0][i] = 0;
    T temp = -0.5 + (rand() / (T) RAND_MAX);
    v[1][i] = temp;
    sum += temp*temp;
  }
  sum = sqrt(sum);

  for ( int i = 0; i < N; ++i )
    v[1][i] /= sum;

  beta[1] = sum;

  // Scratch array for partial results
  T *scratch = new T[rows_per_node];
  T *omega_data = new T[(M+1) * 3];
  T **omega = new T*[3];
  T *dot_prods = new T[M];
  T *dot_prods_reduced = new T[M];
  omega[0] = omega_data;
  omega[1] = omega_data + (M+1);
  omega[2] = omega_data + 2*(M+1);

  omega[0][0] = 1.0;
  omega[1][1] = 1.0;
  omega[1][0] = 0.0;

  vector<int> to_reorthogonalize;
  bool prev_reorthogonalized = false;

  cublasHandle_t handle;
  cublasCreate(&handle);


  T *devPtrX , *devPtrScratch, *devPtrNextBeta;
  T *devPtrTestScratch;
  T *devPtrCsrData;
  int *devPtrCsrRowPtr, *devPtrCsrColIdx;
  T *devPtrVj;

  T devAlpha;

  cudaMalloc( (void**) &devPtrX, rows_in_node * sizeof(T));	// vector A
  cudaMalloc( (void**) &devPtrNextBeta, sizeof(T));

  // cusparse init
  cusparseHandle_t cusparse_handle=0;
  cusparseMatDescr_t descr=0;
  cusparseCreate(&cusparse_handle);

  cusparseCreateMatDescr(&descr); 
  cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

  T one = 1;
  T zero = 0;

  // for CSR
  cudaMalloc( (void**) &devPtrCsrData, nnz * sizeof(T));
  cudaMalloc( (void**) &devPtrCsrRowPtr, (rows_per_node + 1) * sizeof(int));
  cudaMalloc( (void**) &devPtrCsrColIdx, nnz * sizeof(int));
  cudaMalloc( (void**) &devPtrVj, rows_per_node * sizeof(T));

  // for scratch vector
  cudaMalloc( (void**) &devPtrScratch, rows_per_node * sizeof(T));		// vector scratch

  // Start main Lanczos loop to compute the tri-diagonal elements, alpha[i] and beta[i]
  for ( int j = 1; j < M; j++ )
  {
    // Compute local alpha of the current iteration
    
    sparse_csr_mdotv<T>(data, row_ptr, col_idx, rows_in_node, v[j], N, scratch);
    //cudaMemcpy(devPtrCsrData, data, nnz * sizeof(T), cudaMemcpyHostToDevice);
    //cudaMemcpy(devPtrCsrRowPtr, row_ptr, (rows_per_node+1) * sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(devPtrCsrColIdx, col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(devPtrVj, v[j], rows_per_node * sizeof(int), cudaMemcpyHostToDevice);
    //cusparseDcsrmv(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, rows_per_node, N, nnz, &one, descr,
        //devPtrCsrData, devPtrCsrRowPtr, devPtrCsrColIdx, devPtrVj, &zero, devPtrScratch);
    cublasSetVector (rows_in_node, sizeof(T), scratch, 1, devPtrScratch, 1);

    
    //alpha[j] = dense_vdotv<T>(scratch, rows_in_node, v[j] + local_start_index);
    cublasTdot(handle, rows_in_node, devPtrScratch, 1, devPtrVj + local_start_index, 1, &alpha[j]);
    

    // Reduce sum alphas of different nodes to obtain alpha, then broadcast it back
    T res;
    MPI_Allreduce(&alpha[j], &res, 1, mpi_datatype, MPI_SUM, MPI_COMM_WORLD);
    alpha[j] = res;

    // Orthogonalize against past 2 vectors v[j], v[j-1]
    // Need to take care to subtract the right subpart of the arrays for each node

    //daxpy<T>(scratch, -alpha[j], v[j]+local_start_index, scratch, rows_in_node);
    devAlpha = -alpha[j];
    cublasSetVector( rows_in_node, sizeof(T), v[j] + local_start_index, 1, devPtrX, 1);
    cublasTaxpy(handle, rows_in_node, &devAlpha, devPtrX, 1, devPtrScratch, 1);

    //daxpy<T>(scratch, -beta[j], v[j-1]+local_start_index, scratch, rows_in_node);
    devAlpha = -beta[j];
    cublasSetVector( rows_in_node, sizeof(T), v[j-1] + local_start_index, 1, devPtrX, 1);
    cublasTaxpy(handle, rows_in_node, &devAlpha, devPtrX, 1, devPtrScratch, 1);

    // get scratch from GPU
    cublasGetVector (rows_in_node, sizeof(T), devPtrScratch, 1, scratch, 1);

    // Store normalization constant as beta[j+1]
    beta[j+1] = 0;
    for ( int k = 0; k < rows_in_node; k++ )
      beta[j+1] += scratch[k]*scratch[k];
    MPI_Allreduce(&beta[j+1], &res, 1, mpi_datatype, MPI_SUM, MPI_COMM_WORLD);
    beta[j+1] = sqrt(res);

    if ( fabs(beta[j+1] - 0) < EPS )
    {
      M = j+1;
      cerr << "Found an invariant subspace at " << j << "\n";
    }

    // Perform necessary re-orthogonalization
    update_estimate_recurrence<T> ( beta, alpha, omega, j, scratch, epsilon, sqrteps, v,
                                    rows_in_node, local_start_index, N );

    if ( !prev_reorthogonalized )
      prev_reorthogonalized =
        find_offending_indices<T> ( j, omega, sqrteps, eta, to_reorthogonalize );
    else
      prev_reorthogonalized = false;

    re_orthogonalize<T> ( v, to_reorthogonalize, local_start_index, rows_in_node,
                          omega, scratch, epsilon, sqrteps, j, mpi_datatype, beta, N,
                          dot_prods, dot_prods_reduced );

    // Normalize the vector
    for ( int k = local_start_index; k < local_start_index + rows_in_node; k++ )
      scratch[k-local_start_index] = v[j+1][k]/beta[j+1];

    // Gather and form the new array v[j+1] on each node
    MPI_Allgather ( scratch, rows_per_node, mpi_datatype, v[j+1], rows_per_node,
                    mpi_datatype, MPI_COMM_WORLD );

    // Wow! Invariant subspace was found
    if ( fabs(beta[j+1] - 0) < EPS )
      break;

    if ( !prev_reorthogonalized )
      to_reorthogonalize.clear();
    // End loop for j+1
  }

  // Compute the last remaining alpha[M]
  sparse_csr_mdotv<T>(data, row_ptr, col_idx, rows_in_node, v[M], N, v[M+1]);
  alpha[M] = dense_vdotv<T>(v[M+1], rows_in_node, v[M] + local_start_index);

  // Reduce sum alphas of different nodes to obtain alpha, then broadcast it back
  T res;
  MPI_Allreduce(&alpha[M], &res, 1, mpi_datatype, MPI_SUM, MPI_COMM_WORLD);
  alpha[M] = res;

#ifdef DEBUG
  for ( int j = 1; j < M; j++ )
    for ( int k = 1; k <= j; k++ )
      cerr << "orth(" << j << "," << k <<") = " << dense_vdotv<T>(v[j], N, v[k]) << "\n";
#endif

  *alpha_out = new T[M+1];
  *beta_out = new T[M+1];

  // Copy the alpha and beta values out
  memcpy ( *alpha_out, alpha + 1, M * sizeof(T) );
  memcpy ( *beta_out, beta + 2, M * sizeof(T) );

  // Copy the pointers to the intermediate vectors
  for ( int i = 1; i <= M; i++ )
    (*v_out)[i-1] = v[i];

  delete dot_prods;
  delete dot_prods_reduced;
  delete v;
  delete alpha;
  delete beta;
  delete scratch;
  delete omega_data;
  delete omega;
}

template void lanczos_csr_cuda (
    float *data, int *row_ptr, int *col_idx, int nnz, int N, int rows_in_node,
    int rows_per_node, int local_start_index, int M, MPI_Datatype mpi_datatype,
    float **alpha_out, float **beta_out, float ***v_out, float eta );

template void lanczos_csr_cuda (
    double *data, int *row_ptr, int *col_idx, int nnz, int N, int rows_in_node,
    int rows_per_node, int local_start_index, int M, MPI_Datatype mpi_datatype,
    double **alpha_out, double **beta_out, double ***v_out, double eta );


/**
 * Run the Lanczos algorithm with partial re-orthogonalization on a distributed CSC
 * matrix.
 *
 * Inputs:
 *   data - Data array of CSC matrix
 *   col_ptr - Column pointers for CSC matrix
 *   row_idx - Row indices corresponding to 'data'
 *   N - Total number of rows/columns in the matrix (global)
 *   columns_in_node - Number of columns present in the local node
 *   columns_per_node - Ideal number of columns per node (if it was exactly divisible)
 *   local_start_index - Column that column 0 of this matrix partition corresponds to
 *   M - Number of iterations to run the Lanczos algorithm for (size of tri-diagonal
 *       matrix)
 *   mpi_datatype - The MPI datatype corresponding to T (eg. MPI_DOUBLE)
 *   eta - Neighbourhood to re-orthogonalize in (0 to 1, 0 = none, 1 = full reorth.)
 *         (optional)
 *
 * Outputs:
 *   alpha_out - The diagonal elements of the tri-diagonal matrix
 *   beta_out - The off-diagonal elements of the tri-diagonal matrix
 *   v_out - The produced intermediate orthonormal Lanczos vectors (MxN size)
 */
template <typename T>
void lanczos_csc ( T *data, int *col_ptr, int *row_idx, int N, int cols_in_node,
                   int cols_per_node, int local_start_index, int M,
                   MPI_Datatype mpi_datatype, T **alpha_out, T **beta_out,
                   T ***v_out, T eta )
{
  int rank, num_tasks;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);

  // Determine the square root of the machine precision for re-orthogonalization
  T epsilon = numeric_limits<T>::epsilon();
  T sqrteps = sqrt(epsilon);
  eta = pow(epsilon, 0.75);

  // The distributed intermediate vectors v
  int vsize = cols_per_node;
  T *v_data = new T[(M+2) * vsize];
  T **v = new T*[M+2];
  for ( int i = 0; i < M+2; i++ )
    v[i] = v_data + i*vsize;

  // The values in the tri-diagonal matrix, alphas and betas
  T *alpha = new T[N+1];
  T *beta = new T[N+1];

  unsigned int seedval;
  if ( rank == MASTER )
    seedval = time(NULL);
  MPI_Bcast(&seedval, 1, MPI_UNSIGNED, MASTER, MPI_COMM_WORLD);

  srand(seedval);

  // Generate the initial normalized vectors v1, and the 0 vector v0
  T sum = 0.;
  T sum_local = 0.;
  for ( int i = 0; i < cols_in_node; i++ )
  {
    v[0][i] = 0.;
    T temp = -0.5 + (rand() / (T) RAND_MAX);
    v[1][i] = temp;
    sum_local += temp*temp;
  }
  MPI_Allreduce(&sum_local, &sum, 1, mpi_datatype, MPI_SUM, MPI_COMM_WORLD);
  sum = sqrt(sum);

  for ( int i = 0; i < cols_in_node; i++ )
    v[1][i] /= sum;

  beta[1] = 0;

  // Scratch array for partial results
  T *scratch = new T[N];
  T *dot_prods = new T[M];
  T *dot_prods_reduced = new T[M];
  T *omega_data = new T[(M+1) * 3];
  T **omega = new T*[3];
  omega[0] = omega_data;
  omega[1] = omega_data + (M+1);
  omega[2] = omega_data + 2*(M+1);

  omega[0][0] = 1.0;
  omega[1][1] = 1.0;
  omega[1][0] = 0.0;

  vector<int> to_reorthogonalize;
  bool prev_reorthogonalized = false;

  // Re-seed pseudo random number generator, as the calls to random would have been
  // different for the last node if N % cols_per_node != 0
  if ( rank == MASTER )
    seedval = time(NULL);
  MPI_Bcast(&seedval, 1, MPI_UNSIGNED, MASTER, MPI_COMM_WORLD);
  srand(seedval);

  // Start main Lanczos loop to compute the tri-diagonal elements, alpha[i] and beta[i]
  for ( int j = 1; j < M; j++ )
  {
    // Compute the distributed vector v[j+1]
    sparse_csc_mdotv<T>(data, col_ptr, row_idx, cols_in_node, N, v[j], N, scratch);

    // Reduce sum the corresponding parts of v[j+1] to each node
    for ( int k = 0; k < num_tasks; k++ )
      MPI_Reduce(scratch + k * cols_per_node, v[j+1], cols_per_node,
          mpi_datatype, MPI_SUM, k, MPI_COMM_WORLD);

    // Compute the alpha for this iteration
    // Reduce sum alphas of different nodes to obtain alpha
    T alpha_local = dense_vdotv<T>(v[j+1], cols_in_node, v[j]+local_start_index);
    MPI_Allreduce(&alpha_local, &alpha[j], 1, mpi_datatype, MPI_SUM, MPI_COMM_WORLD);

    // Orthogonalize against past 2 vectors v[j], v[j-1]
    daxpy<T>(v[j+1], -alpha[j], v[j], v[j+1], cols_in_node);
    daxpy<T>(v[j+1], -beta[j], v[j-1], v[j+1], cols_in_node);

    // Store normalization constant as beta[j+1]
    T beta_local = 0.;
    for ( int k = 0; k < cols_in_node; k++ )
      beta_local += v[j+1][k]*v[j+1][k];
    MPI_Allreduce(&beta_local, &beta[j+1], 1, mpi_datatype, MPI_SUM, MPI_COMM_WORLD);
    beta[j+1] = sqrt(beta[j+1]);

    if ( fabs(beta[j+1] - 0) < EPS )
    {
      M = j+1;
      cerr << "Found an invariant subspace at " << j << "\n";
    }

    // This will break if debug is enabled.
    // TODO: Fix this using a flag or separate it out to another function.
    update_estimate_recurrence<T> ( beta, alpha, omega, j, scratch, epsilon, sqrteps,
                                    v, cols_in_node, local_start_index, N );

    if ( !prev_reorthogonalized )
      prev_reorthogonalized =
        find_offending_indices<T> ( j, omega, sqrteps, eta, to_reorthogonalize );
    else
      prev_reorthogonalized = false;

    re_orthogonalize<T> ( v, to_reorthogonalize, 0, cols_in_node, omega, scratch,
                          epsilon, sqrteps, j, mpi_datatype, beta, N, dot_prods,
                          dot_prods_reduced );

    // Normalize the local portion of the vector
    for ( int k = 0; k < cols_in_node; k++ )
      v[j+1][k] /= beta[j+1];

    // Found an invariant subspace, so break
    if ( fabs(beta[j+1] - 0) < EPS )
      break;

    if ( !prev_reorthogonalized )
      to_reorthogonalize.clear();
    // End loop for j+1
  }

  // Compute the last remaining alpha[N]
  sparse_csc_mdotv<T>(data, col_ptr, row_idx, cols_in_node, N, v[N], N, scratch);
  for ( int k = 0; k < num_tasks; k++ )
    MPI_Reduce(scratch + k * cols_per_node, v[N+1], cols_per_node,
        mpi_datatype, MPI_SUM, k, MPI_COMM_WORLD);

  T alpha_local = dense_vdotv<T>(v[N+1], cols_in_node, v[N]);
  MPI_Allreduce(&alpha_local, &alpha[N], 1, mpi_datatype, MPI_SUM, MPI_COMM_WORLD);

  // Copy the alphas and betas into output variables
  *alpha_out = new T[M+1];
  *beta_out = new T[M+1];

  memcpy ( *alpha_out, alpha + 1, M * sizeof(T) );
  memcpy ( *beta_out, beta + 2, M * sizeof(T) );

  // Assign pointers to the intermediate vectors
  for ( int i = 1; i <= M; i++ )
    (*v_out)[i-1] = v[i];

  delete dot_prods;
  delete dot_prods_reduced;
  delete v;
  delete alpha;
  delete beta;
  delete scratch;
  delete omega_data;
  delete omega;
}

