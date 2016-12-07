#include "headers/graph.hpp"
#include "headers/utils.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include "mpi.h"

using std::vector;

/**
 * Reads and creates a distributed CSR graph of the required type 'T' from and edgelist
 * file.
 * ASSUMES THAT THE SAME FILE IS ACCESSIBLE FROM ALL NODES.
 * ASSUMES THAT THE GRAPH IS DIRECTED.
 * ASSUMES THAT THE EDGES ARE SORTED BY SOURCE VERTEX IDs.
 * Format of edge list file:
 * N M
 * V_1 V_2 W_1,2
 * ...
 * V_M-1 V_M W_M-1,M
 */
template <typename T>
distributed_graph_csr<T>* create_csr_from_edgelist_file(std::ifstream &fin)
{
  int N, M;
  int num_nodes;
  int rows_per_node, task_id;
  int rows_in_node;
  MPI_Comm_rank(MPI_COMM_WORLD, &task_id);
  MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);

  fin >> N >> M;

  rows_per_node = (N + num_nodes - 1) / num_nodes;

  if ( N % num_nodes != 0 )
  {
    if ( task_id != num_nodes - 1 )
      rows_in_node = rows_per_node;
    else
      rows_in_node = N % rows_per_node;
  }
  else
    rows_in_node = rows_per_node;

  distributed_graph_csr<T> *new_graph =
    new distributed_graph_csr<T>(N, M, rows_per_node, rows_in_node, task_id);

  int this_start_index = rows_per_node * task_id;

  T *A;
  int *row_ptr;
  int *col_idx;

  vector<int> vis, vjs;
  vector<T> data;

  // Read the rest of the edgelist file and populate the CSR arrays
  for ( int edge_no = 0; edge_no < M; edge_no++ )
  {
    int vi, vj;
    T wij;
    fin >> vi >> vj >> wij;
    if ( this_start_index <= vi && vi < this_start_index + rows_per_node )
    {
      vis.push_back(vi);
      vjs.push_back(vj);
      data.push_back(wij);
    }
  }

  convert_edgelist_to_csr <T> ( data, vis, vjs, rows_per_node, this_start_index,
                                &A, &row_ptr, &col_idx );

  new_graph->set_A(A);
  new_graph->set_row_ptr(row_ptr);
  new_graph->set_col_idx(col_idx);
  new_graph->set_nnz_local(data.size());

  return new_graph;
}

/**
 * Construct the unnormalized laplacian of the given distributed graph as
 * L = D - A. Note that A[i][j] = - 1 for i != j, and A[i][i] = Degree(i) - A[i][i].
 * Degree(i) is given by row_ptr(i+1) - row_ptr(i).
 */
template <typename T>
void distributed_graph_csr<T>::construct_unnormalized_laplacian()
{
  vector<int> vis;
  vector<int> vjs;
  vector<T> data;

  int selfloop_count = 0;
  for ( int row_idx = 0; row_idx < rows_per_node; row_idx++ )
  {
    for ( int data_idx = row_ptr[row_idx]; data_idx < row_ptr[row_idx+1]; data_idx++ )
    {
      // If self loop is missing, push it into the array with degree D (as A[i][i] = 0)
      if ( col_idx[data_idx] > row_idx + row_start_idx &&
           ( vis.size() == 0 || ( vis[vis.size()-1] != row_idx + row_start_idx ||
                                  vis[vis.size()-1] > vjs[vjs.size()-1] ) ) )
      {
        vis.push_back(row_idx + row_start_idx);
        vjs.push_back(row_idx + row_start_idx);
        data.push_back((T) row_ptr[row_idx+1] - row_ptr[row_idx]);
      }

      vis.push_back(row_idx + row_start_idx);
      vjs.push_back(col_idx[data_idx]);
      if ( col_idx[data_idx] == row_idx + row_start_idx )
      {
        selfloop_count++;
        // For A[i][i] != 0, D - A[i][i]
        data.push_back((T) (row_ptr[row_idx+1] - row_ptr[row_idx]) - A[data_idx]);
      }
      else
        data.push_back(-A[data_idx]);
    }

    // If the self loop is absent because we never got to i,i
    // (i.e. j was always less than i in A[i][j]), push the selfloop in
    if ( vis[vis.size()-1] > vjs[vjs.size()-1] )
    {
      vis.push_back(row_idx + row_start_idx);
      vjs.push_back(row_idx + row_start_idx);
      data.push_back((T) row_ptr[row_idx+1] - row_ptr[row_idx]);
    }
  }

  int num_tasks;
  MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);

  // Self loop count in the Laplacian should be of size rows_in_node
  // So we can estimate the size of nnz and use this to verify correctness.
  int lap_nnz = nnz_local + rows_in_node - selfloop_count;

  // The number of elements added should match up with our estimate of nnz
  assert(data.size() == lap_nnz);

  T* lap_A;
  int* lap_row_ptr;
  int* lap_col_idx;

  convert_edgelist_to_csr <T> ( data, vis, vjs, rows_per_node, row_start_idx,
                                &lap_A, &lap_row_ptr, &lap_col_idx);

  this -> set_lap_A ( lap_A );
  this -> set_lap_row_ptr ( lap_row_ptr );
  this -> set_lap_col_idx ( lap_col_idx );
  this -> set_lap_nnz_local ( lap_nnz );
}

/**
 * Free the memory associated with the adjacency matrix variables of this graph.
 * Useful if one is only working with the Laplacian.
 */
template <typename T>
void distributed_graph_csr<T>::free_adjacency_matrix()
{
  delete this -> A;
  delete this -> row_ptr;
  delete this -> col_idx;
  this -> A = NULL;
  this -> row_ptr = NULL;
  this -> col_idx = NULL;
}

/**
 * Reads and creates a distributed graph CSC of the required type 'T' from and edgelist
 * file.
 * ASSUMES THAT THE SAME FILE IS ACCESSIBLE FROM ALL NODES.
 * ASSUMES THAT THE GRAPH IS DIRECTED.
 * ASSUMES THAT THE EDGES ARE SORTED BY DESTINATION VERTEX IDs.
 * Format of edge list file:
 * N M
 * V_1 V_2 W_1,2
 * ...
 * V_M-1 V_M W_M-1,M
 */
template <typename T>
distributed_graph_csc<T>* create_csc_from_edgelist_file(std::ifstream &fin)
{
  int N, M;
  int num_nodes;
  int cols_per_node, task_id;
  int cols_in_node;
  MPI_Comm_rank(MPI_COMM_WORLD, &task_id);
  MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);

  fin >> N >> M;

  cols_per_node = (N + num_nodes - 1) / num_nodes;

  if ( N % num_nodes != 0 )
  {
    if ( task_id != num_nodes - 1 )
      cols_in_node = cols_per_node;
    else
      cols_in_node = N % cols_per_node;
  }
  else
    cols_in_node = cols_per_node;

  distributed_graph_csc<T> *new_graph =
    new distributed_graph_csc<T>(N, M, cols_per_node, cols_in_node, task_id);

  int this_start_index = cols_per_node * task_id;

  T *A;
  int *col_ptr;
  int *row_idx;

  vector<int> vis, vjs;
  vector<T> data;

  // Read the rest of the edgelist file and populate the CSC arrays
  for ( int edge_no = 0; edge_no < M; edge_no++ )
  {
    int vi, vj;
    T wij;
    fin >> vi >> vj >> wij;
    if ( this_start_index <= vj && vj < this_start_index + cols_per_node )
    {
      vis.push_back(vi);
      vjs.push_back(vj);
      data.push_back(wij);
    }
  }

  convert_edgelist_to_csc <T> ( data, vis, vjs, cols_per_node, this_start_index,
                                &A, &col_ptr, &row_idx );

  new_graph->set_A(A);
  new_graph->set_col_ptr(col_ptr);
  new_graph->set_row_idx(row_idx);
  new_graph->set_nnz_local(data.size());

  return new_graph;
}

/**
 * Construct the unnormalized laplacian of the given distributed CSC graph as
 * L = D - A. Note that A[i][j] = - 1 for i != j, and A[i][i] = Degree(i) - A[i][i].
 * ASSUMES THAT THE GRAPH IS UNDIRECTED.
 * Degree(i) is given by col_ptr(i+1) - col_ptr(i).
 */
template <typename T>
void distributed_graph_csc<T>::construct_unnormalized_laplacian()
{
  vector<int> vis;
  vector<int> vjs;
  vector<T> data;

  int selfloop_count = 0;
  for ( int col_idx = 0; col_idx < cols_per_node; col_idx++ )
  {
    for ( int data_idx = col_ptr[col_idx]; data_idx < col_ptr[col_idx+1]; data_idx++ )
    {
      // If self loop is missing, push it into the array with degree D (as A[i][i] = 0)
      if ( row_idx[data_idx] > col_idx + col_start_idx &&
           ( vjs.size() == 0 || ( vjs[vjs.size()-1] != col_idx + col_start_idx ||
                                  vjs[vjs.size()-1] > vis[vis.size()-1] ) ) )
      {
        vis.push_back(col_idx + col_start_idx);
        vjs.push_back(col_idx + col_start_idx);
        data.push_back((T) col_ptr[col_idx+1] - col_ptr[col_idx]);
      }

      vis.push_back(row_idx[data_idx]);
      vjs.push_back(col_idx + col_start_idx);
      if ( row_idx[data_idx] == col_idx + col_start_idx )
      {
        selfloop_count++;
        // For A[i][i] != 0, D - A[i][i]
        data.push_back((T) (col_ptr[col_idx+1] - col_ptr[col_idx]) - A[data_idx]);
      }
      else
        data.push_back(-A[data_idx]);
    }

    // If the self loop is absent because we never got to j,j
    // (i.e. i was always less than j in A[i][j]), push the selfloop in
    if ( vjs[vjs.size()-1] > vis[vis.size()-1] )
    {
      vis.push_back(col_idx + col_start_idx);
      vjs.push_back(col_idx + col_start_idx);
      data.push_back((T) col_ptr[col_idx+1] - col_ptr[col_idx]);
    }
  }

  int num_tasks;
  MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);

  // Self loop count in the Laplacian should be of size cols_in_node
  // So we can estimate the size of nnz and use this to verify correctness.
  int lap_nnz = nnz_local + cols_in_node - selfloop_count;

  // The number of elements added should match up with our estimate of nnz
  assert(data.size() == lap_nnz);

  T* lap_A;
  int* lap_col_ptr;
  int* lap_row_idx;

  convert_edgelist_to_csc <T> ( data, vis, vjs, cols_per_node, col_start_idx,
                                &lap_A, &lap_col_ptr, &lap_row_idx);

  this -> set_lap_A ( lap_A );
  this -> set_lap_col_ptr ( lap_col_ptr );
  this -> set_lap_row_idx ( lap_row_idx );
  this -> set_lap_nnz_local ( lap_nnz );
}

/**
 * Free the memory associated with the adjacency matrix variables of this graph.
 * Useful if one is only working with the Laplacian.
 */
template <typename T>
void distributed_graph_csc<T>::free_adjacency_matrix()
{
  delete this -> A;
  delete this -> col_ptr;
  delete this -> row_idx;
  this -> A = NULL;
  this -> col_ptr = NULL;
  this -> row_idx = NULL;
}

template class distributed_graph_csr<float>;
template class distributed_graph_csr<double>;

template distributed_graph_csr<float>* create_csr_from_edgelist_file(std::ifstream&);
template distributed_graph_csr<double>* create_csr_from_edgelist_file(std::ifstream&);

template class distributed_graph_csc<float>;
template class distributed_graph_csc<double>;

template distributed_graph_csc<float>* create_csc_from_edgelist_file(std::ifstream&);
template distributed_graph_csc<double>* create_csc_from_edgelist_file(std::ifstream&);

