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
#include "cublas_v2.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "cusparse.h"
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

  ifstream fin("../data/as20000102_row.txt");
  distributed_graph_csr<DATATYPE> *inputGraph =
    create_csr_from_edgelist_file<DATATYPE>(fin);
  inputGraph -> construct_unnormalized_laplacian();
  inputGraph -> free_adjacency_matrix();

  double data_load_time = MPI_Wtime();

  // Get distributed Laplacian CSR submatrices
  DATATYPE *data = inputGraph -> get_lap_A();
  int *row_ptr = inputGraph -> get_lap_row_ptr();
  int *col_idx = inputGraph -> get_lap_col_idx();
  int nnz = inputGraph -> get_lap_nnz_local();
  int N = inputGraph -> get_N();
  int rows_in_node = inputGraph -> get_rows_in_node();
  int rows_per_node = inputGraph -> get_rows_per_node();
  int local_start_index = inputGraph -> get_row_start_index();

  // Determine the square root of the machine precision for re-orthogonalization
  DATATYPE epsilon = numeric_limits<DATATYPE>::epsilon();
  DATATYPE sqrteps = sqrt(epsilon);
  DATATYPE eta = pow(epsilon, 0.75);

  // The intermediate vectors v
  int vsize = rows_per_node * num_tasks;
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
  // TODO: Check which is faster, generate locally on each machine, or generate in
  // parallel and perform AllGather?
  // Generating locally for now. All of them are using the same seed to the
  // pseudo-random number generator. WILL ONLY WORK if all machines are using the
  // same library version.
  DATATYPE sum = 0;
  for ( int i = 0; i < N; i++ )
  {
    v[0][i] = 0;
    DATATYPE temp = rand() / (DATATYPE) RAND_MAX;
    v[1][i] = temp;
    sum += temp*temp;
  }
  sum = sqrt(sum);

  for ( int i = 0; i < N; i++ )
    v[1][i] /= sum;

  beta[1] = 0;
  
  // Scratch array for partial results
  DATATYPE *scratch = new DATATYPE[rows_per_node];
  DATATYPE *test_scratch = new DATATYPE[rows_per_node];
  DATATYPE *nextBeta= new DATATYPE;
  DATATYPE *omega_data = new DATATYPE[(N+1) * (N+1)];
  DATATYPE **omega = new DATATYPE*[N+1];
  for ( int i = 0; i < N+1; i++ )
    omega[i] = omega_data + i * (N+1);

  omega[0][0] = 1.0;
  omega[1][1] = 1.0;

  vector<int> to_reorthogonalize;
  bool prev_reorthogonalized = false;

  double initialization_time = MPI_Wtime();

  cublasHandle_t handle;
  cublasCreate(&handle);


  float *devPtrX , *devPtrScratch, *devPtrNextBeta;
  float *devPtrCsrData;
  int *devPtrCsrRowPtr, *devPtrCsrColIdx;
  float *devPtrVj;

  float devAlpha;

  cudaMalloc( (void**) &devPtrX, rows_in_node * sizeof(float));	// vector A
  cudaMalloc( (void**) &devPtrNextBeta, sizeof(float));

  // cusparse init
  cusparseHandle_t cusparse_handle=0;
  cusparseMatDescr_t descr=0;
  cusparseCreate(&cusparse_handle);

  cusparseCreateMatDescr(&descr); 
  cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

  float one = 1;
  float zero = 0;

  // for CSR
  cudaMalloc( (void**) &devPtrCsrData, nnz * sizeof(float));
  cudaMalloc( (void**) &devPtrCsrRowPtr, (rows_per_node + 1) * sizeof(int));
  cudaMalloc( (void**) &devPtrCsrColIdx, nnz * sizeof(int));
  cudaMalloc( (void**) &devPtrVj, rows_per_node * sizeof(float));

  // for scratch vector
  cudaMalloc( (void**) &devPtrScratch, rows_per_node * sizeof(float));		// vector scratch


  // Start main Lanczos loop to compute the tri-diagonal elements, alpha[i] and beta[i]
  for ( int j = 1; j < N; j++ )
  {
    // Compute local alpha of the current iteration
    
    cudaMemcpy(devPtrCsrData, data, nnz * sizeof(DATATYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(devPtrCsrRowPtr, row_ptr, (rows_per_node+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(devPtrCsrColIdx, col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(devPtrVj, v[j], rows_per_node * sizeof(int), cudaMemcpyHostToDevice);

    cusparseScsrmv(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, rows_per_node, N, nnz, &one, descr,
        devPtrCsrData, devPtrCsrRowPtr, devPtrCsrColIdx, devPtrVj, &zero, devPtrScratch);

    //cudaMemcpy(scratch, devPtrScratch, rows_per_node * sizeof(float), cudaMemcpyDeviceToHost);

    //sparse_csr_mdotv<DATATYPE>(data, row_ptr, col_idx, rows_in_node, v[j], N, test_scratch);


    // for verification. cpu gpu error within single precision acc
    //float err_sum = 0;
    //for (int i = 0; i < rows_per_node; i++) {
      //if ( i%100 == 0) 
      //{
        //float err = scratch[i] - test_scratch[i];
        //err_sum += err;
        //if (err!= 0)
          //cout << "slight error" << err_sum << endl;
      //}

    //}
    //cout << "err_sum" << err_sum<< endl;

    float *devAlpha;
    cublasSdot(handle, rows_in_node, devPtrScratch, 1, devPtrVj, 1, &alpha[j]);
    //alpha[j] = dense_vdotv<DATATYPE>(scratch, rows_in_node, v[j]);
    
    // Reduce sum alphas of different nodes to obtain alpha, then broadcast it back
    DATATYPE res;
    MPI_Allreduce(&alpha[j], &res, 1, MPIDATATYPE, MPI_SUM, MPI_COMM_WORLD);
    alpha[j] = res;

    //cublasSetVector( rows_in_node, sizeof(float), scratch, 1, devPtrScratch, 1);

    // Orthogonalize against past 2 vectors v[j], v[j-1]
    // Need to take care to subtract the right subpart of the arrays for each node
    
    devAlpha = -alpha[j];
    cublasSetVector( rows_in_node, sizeof(float), v[j] + local_start_index, 1, devPtrX, 1);
    cublasSaxpy(handle, rows_in_node, &devAlpha, devPtrX, 1, devPtrScratch, 1);

    devAlpha = -beta[j];
    cublasSetVector( rows_in_node, sizeof(float), v[j-1] + local_start_index, 1, devPtrX, 1);
    cublasSaxpy(handle, rows_in_node, &devAlpha, devPtrX, 1, devPtrScratch, 1);

    // copy value of scratch to host 
    cublasGetVector (rows_in_node, sizeof(float), devPtrScratch, 1, scratch, 1);


    // Sanity check:
    // There is a very small difference between the computation in gpu and cpu in the order of e-08
    //
    //for (int i = 0; i < rows_in_node; i++)
      //if (cu_scratch[i] != scratch[i])
        //cerr << "Not equal!" << cu_scratch[i] - scratch[i]<<"\n";


    // Normalize v[j+1] and store normalization constant as beta[j+1]
    cublasSdot(handle, rows_in_node, devPtrScratch, 1, devPtrScratch, 1, nextBeta);
    beta[j+1] = *nextBeta;
    
    cudaThreadSynchronize(); // block until the device has completed

    MPI_Allreduce(&beta[j+1], &res, 1, MPIDATATYPE, MPI_SUM, MPI_COMM_WORLD);
    beta[j+1] = sqrt(res);

    // Normalize local portion of the vector
    a = 1/beta[j+1];
    cublasSscal(handle, rows_in_node, &a, devPtrScratch, 1);
    cublasGetVector (rows_in_node, sizeof(float), devPtrScratch, 1, scratch, 1);

    // Gather and form the new array v[j+1] on each node
    MPI_Allgather(scratch, rows_per_node, MPIDATATYPE, v[j+1], rows_per_node,
                  MPIDATATYPE, MPI_COMM_WORLD);

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
      DATATYPE temp_local =
        dense_vdotv<DATATYPE>(
            v[j+1]+local_start_index, rows_in_node, v[index]+local_start_index
        );
      MPI_Allreduce(&temp_local, &temp, 1, MPIDATATYPE, MPI_SUM, MPI_COMM_WORLD);
      daxpy<DATATYPE>(
          v[j+1]+local_start_index, -temp, v[index]+local_start_index,
          v[j+1]+local_start_index, rows_in_node);
      // Update estimate after this vector has been re-normalized
      omega[j+1][index] = epsilon * random_normal(0, 1.5);
    }

    if ( to_reorthogonalize.size() > 0 )
    {
      // Need to re-normalize v[j+1] and store normalization constant as beta[j+1]
      beta[j+1] = 0;
      for ( int k = local_start_index; k < local_start_index + rows_in_node; k++ )
        beta[j+1] += v[j+1][k]*v[j+1][k];
      MPI_Allreduce(&beta[j+1], &res, 1, MPIDATATYPE, MPI_SUM, MPI_COMM_WORLD);
      beta[j+1] = sqrt(res);

      // Normalize the vector
      for ( int k = local_start_index; k < local_start_index + rows_in_node; k++ )
        scratch[k-local_start_index] = v[j+1][k]/beta[j+1];

      // Gather and form the new array v[j+1] on each node
      MPI_Allgather(scratch, rows_per_node, MPIDATATYPE, v[j+1], rows_per_node,
                    MPIDATATYPE, MPI_COMM_WORLD);
    }

    if ( !prev_reorthogonalized )
      to_reorthogonalize.clear();
    // End loop for j+1
  }

  // Compute the last remaining alpha[N]
  sparse_csr_mdotv<DATATYPE>(data, row_ptr, col_idx, rows_in_node, v[N], N, v[N+1]);
  alpha[N] = dense_vdotv<DATATYPE>(v[N+1], rows_in_node, v[N]);

  // Reduce sum alphas of different nodes to obtain alpha, then broadcast it back
  DATATYPE res;
  MPI_Allreduce(&alpha[N], &res, 1, MPIDATATYPE, MPI_SUM, MPI_COMM_WORLD);
  alpha[N] = res;

  double tri_diagonalization_time = MPI_Wtime();

  //if ( rank == 0 )
    //for ( int i = 1; i <= N; i++ )
      //cout << alpha[i] << " " << beta[i] << "\n";

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

