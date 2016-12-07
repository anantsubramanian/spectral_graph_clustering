#include "headers/utils.hpp"
#include <vector>

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
void sparse_mdotv ( T *data, int *row_ptr, int *col_idx, int N, T *vec, int vec_size,
                    T *result )
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

template void convert_edgelist_to_csr (
    vector<float> &data, vector<int> &vis, vector<int> &vjs,
    int row_count, int row_base, float** A, int **row_ptr,
    int **col_idx );

template void convert_edgelist_to_csr (
    vector<double> &data, vector<int> &vis, vector<int> &vjs,
    int row_count, int row_base, double** A, int **row_ptr,
    int **col_idx );

template void sparse_mdotv (
    float *data, int *row_ptr, int *col_idx, int N, float *vec, int vec_size,
    float *result );

template void sparse_mdotv (
    double *data, int *row_ptr, int *col_idx, int N, double *vec, int vec_size,
    double *result );

template float dense_vdotv ( float *vec1, int vec_size, float *vec2 );

template double dense_vdotv ( double *vec1, int vec_size, double *vec2 );

template void daxpy ( float *result, float a, float *x, float *y, int vec_size );

template void daxpy ( double *result, double a, double *x, double *y, int vec_size );

