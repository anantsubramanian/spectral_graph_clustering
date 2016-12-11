#include "headers/utils.hpp"
#include <vector>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <cmath>

using std::vector;

/**
 * Convert a given edgelist into CSR format.
 * vis = source vertices, vjs = destination vertices
 * row_count = total number of rows for the given edgelist
 * row_base = index of starting row
 * Outputs stored in A, row_ptr and col_idx.
 */
template <typename T>
void convert_edgelist_to_csr ( vector<T> &data, vector<int> &vis, vector<int> &vjs,
                               int row_count, int row_base, T** A, int **row_ptr,
                               int **col_idx )
{
  int nnz = data.size();
  *A = new T[nnz];
  *col_idx = new int[nnz];
  *row_ptr = new int[row_count+1];

  int cur_row_idx = 0;
  (*row_ptr)[0] = 0;

  for ( int data_idx = 0; data_idx < data.size(); data_idx++ )
  {
    while ( vis[data_idx] > cur_row_idx + row_base )
    {
      cur_row_idx++;
      (*row_ptr)[cur_row_idx] = data_idx;
    }
    (*col_idx)[data_idx] = vjs[data_idx];
    (*A)[data_idx] = data[data_idx];
  }

  cur_row_idx++;
  while (cur_row_idx < row_count + 1) (*row_ptr)[cur_row_idx++] = data.size();
}

/**
 * Multiplies a matrix stored in CSR format, with 'N' rows, with a dense vector
 * of size vec_size, and stores the result in 'result'
 */
template <typename T>
void sparse_csr_mdotv ( T *data, int *row_ptr, int *col_idx, int N, T *vec,
                        int vec_size, T *result )
{
  for ( int i = 0; i < N; i++ )
  {
    T res = (T) 0;
    for ( int j = row_ptr[i]; j < row_ptr[i+1]; j++ )
    {
      res += data[j] * vec[col_idx[j]];
    }
    result[i] = res;
  }
}

/**
 * Computes the dot product of two dense vectors
 */
template <typename T>
T dense_vdotv ( T *vec1, int vec_size, T *vec2 )
{
  T res = (T) 0;
  for ( int i = 0; i < vec_size; i++ )
    res += vec1[i] * vec2[i];
  return res;
}

/**
 * Computes a*x + y for two dense vectors x and y.
 */
template <typename T>
void daxpy ( T *result, T a, T *x, T *y, int vec_size )
{
  for ( int i = 0; i < vec_size; i++ )
    result[i] = a * x[i] + y[i];
}

/**
 * Convert a given edgelist into CSC format.
 * vis = source vertices, vjs = destination vertices
 * col_count = total number of cols for the given edgelist
 * col_base = index of starting column
 * Outputs stored in A, col_ptr and row_idx.
 */
template <typename T>
void convert_edgelist_to_csc ( vector<T> &data, vector<int> &vis, vector<int> &vjs,
                               int col_count, int col_base, T** A, int **col_ptr,
                               int **row_idx )
{
  int nnz = data.size();
  *A = new T[nnz];
  *row_idx = new int[nnz];
  *col_ptr = new int[col_count+1];

  int cur_col_idx = 0;
  (*col_ptr)[0] = 0;

  for ( int data_idx = 0; data_idx < data.size(); data_idx++ )
  {
    while ( vjs[data_idx] > cur_col_idx + col_base )
    {
      cur_col_idx++;
      (*col_ptr)[cur_col_idx] = data_idx;
    }
    (*row_idx)[data_idx] = vis[data_idx];
    (*A)[data_idx] = data[data_idx];
  }

  cur_col_idx++;
  while (cur_col_idx < col_count + 1) (*col_ptr)[cur_col_idx++] = data.size();
}

/**
 * Multiplies a matrix stored in CSC format, with 'N' columns and 'M' rows,
 * with a dense vector of size vec_size, and stores the result in 'result'
 */
template <typename T>
void sparse_csc_mdotv ( T *data, int *col_ptr, int *row_idx, int N, int M,
                        T *vec, int vec_size, T *result )
{
  memset(result, 0, M * sizeof(T));
  for ( int i = 0; i < N; i++ )
    for ( int j = col_ptr[i]; j < col_ptr[i+1]; j++ )
      result[row_idx[j]] += data[j] * vec[i];
}

/**
 * Computes the QR decomposition for a symmetric tri-diagonal matrix.
 * Implementation adapted from:
 * https://github.com/linhr/15618/blob/master/src/eigen.h
 *
 * If that file is not accessible, try here:
 * http://linhr.me/15618/final/
 *
 * Input: The alpha and beta values corresponding to the tri-diagonal
 * elements. alpha[i] = T[i][i], and beta[i] = T[i][i-1] = T[i-1][i]
 *
 * Output: alpha[i] replaced with eigen value i
 */
template <typename T>
void qr_eigen ( T *alpha, T *beta, int N, const T epsilon )
{
  for ( int i = 0; i < N-1; ++i )
    beta[i] *= beta[i];

  beta[N-1] = 0;
  bool converged = false;

  while (!converged)
  {
    T diff(0.);
    T u(0.);
    T ss2(0.), s2(0.);
    for ( int i = 0; i < N; ++i )
    {
      T gamma = alpha[i] - u;
      T p2 = T(std::abs(1 - s2)) < epsilon ?
        (1 - ss2) * beta[i-1] :
        gamma * gamma / (1 - s2);
      if ( i > 0 )
        beta[i-1] = s2 * (p2 + beta[i]);
      ss2 = s2;
      s2 = beta[i] / (p2 + beta[i]);
      if ( i < N-1 )
        u = s2 * (gamma + alpha[i+1]);
      else
        u = s2 * gamma;
      T old = alpha[i];
      alpha[i] = gamma + u;
      diff = std::max(diff, T(std::abs(old - alpha[i])));
    }

    if ( diff < epsilon )
      converged = true;
  }
}

template void convert_edgelist_to_csr (
    vector<float> &data, vector<int> &vis, vector<int> &vjs,
    int row_count, int row_base, float** A, int **row_ptr,
    int **col_idx );

template void convert_edgelist_to_csr (
    vector<double> &data, vector<int> &vis, vector<int> &vjs,
    int row_count, int row_base, double** A, int **row_ptr,
    int **col_idx );

template void sparse_csr_mdotv (
    float *data, int *row_ptr, int *col_idx, int N, float *vec, int vec_size,
    float *result );

template void sparse_csr_mdotv (
    double *data, int *row_ptr, int *col_idx, int N, double *vec, int vec_size,
    double *result );

template void convert_edgelist_to_csc (
    vector<float> &data, vector<int> &vis, vector<int> &vjs,
    int col_count, int col_base, float** A, int **col_ptr,
    int **row_idx );

template void convert_edgelist_to_csc (
    vector<double> &data, vector<int> &vis, vector<int> &vjs,
    int col_count, int col_base, double** A, int **col_ptr,
    int **row_idx );

template void sparse_csc_mdotv (
    float *data, int *col_ptr, int *row_idx, int N, int M, float *vec, int vec_size,
    float *result );

template void sparse_csc_mdotv (
    double *data, int *col_ptr, int *row_idx, int N, int M, double *vec, int vec_size,
    double *result );

template float dense_vdotv ( float *vec1, int vec_size, float *vec2 );

template double dense_vdotv ( double *vec1, int vec_size, double *vec2 );

template void daxpy ( float *result, float a, float *x, float *y, int vec_size );

template void daxpy ( double *result, double a, double *x, double *y, int vec_size );

template void qr_eigen ( float *alpha, float *beta, int N, const float epsilon );

template void qr_eigen ( double *alpha, double *beta, int N, const double epsilon );

