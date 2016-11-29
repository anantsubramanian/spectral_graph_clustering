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

template void convert_edgelist_to_csr (
    vector<float> &data, vector<int> &vis, vector<int> &vjs,
    int row_count, int row_base, float** A, int **row_ptr,
    int **col_idx );

template void convert_edgelist_to_csr (
    vector<double> &data, vector<int> &vis, vector<int> &vjs,
    int row_count, int row_base, double** A, int **row_ptr,
    int **col_idx );

