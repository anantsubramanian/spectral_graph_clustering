#ifndef __GRAPH
  #define __GRAPH
#endif

#include <fstream>
#include <iostream>
#include <cassert>

/**
 * Class that stores the distributed representation of a graph,
 * distributed across nodes, and stored in CSR format.
 */
template <typename T>
class distributed_graph
{
  private:
    int N; // Number of nodes
    int M; // Number of edges (synonymous to NNZ for the whole matrix)

    int rows_per_node;
    int task_id;

    // The start index offset for this node in the matrix
    // i.e. row 0 of this matrix corresponds to row_start_idx
    // of the whole matrix.
    int row_start_idx;

    T *A;      // Data (size = nnz_local)
    int *row_ptr; // Row pointers (size ceil(N/rowsPerNode))
    int *col_idx; // Column indices of the values in A
    int nnz_local;  // The number of nonzero values in this submatrix

    // Similar variables, but for the Laplacian
    T* lap_A;
    int *lap_row_ptr;
    int *lap_col_idx;
    int lap_nnz_local;

  public:

    distributed_graph(int N, int M, int rows_per_node, int task_id)
    {
      this -> N = N;
      this -> M = M;
      this -> rows_per_node = rows_per_node;
      this -> task_id = task_id;

      this -> row_start_idx = rows_per_node * task_id;

      this -> A = NULL;
      this -> row_ptr = NULL;
      this -> col_idx = NULL;

      this -> lap_A = NULL;
      this -> lap_row_ptr = NULL;
      this -> lap_col_idx = NULL;
    }

    ~distributed_graph()
    {
      if (A) delete A;
      if (row_ptr) delete row_ptr;
      if (col_idx) delete col_idx;
      if (lap_A) delete lap_A;
      if (lap_row_ptr) delete lap_row_ptr;
      if (lap_col_idx) delete lap_col_idx;
    }

    void construct_unnormalized_laplacian();
    void free_adjacency_matrix();

    T* get_A() { return A; }
    int* get_row_ptr() { return row_ptr; }
    int* get_col_idx() { return col_idx; }
    int get_nnz_local() { return nnz_local; }

    void set_A(T *A)
    {
      assert(A != NULL);
      this -> A = A;
    }

    void set_row_ptr(int *row_ptr)
    {
      assert(row_ptr != NULL);
      this -> row_ptr = row_ptr;
    }

    void set_col_idx(int *col_idx)
    {
      assert(col_idx != NULL);
      this -> col_idx = col_idx;
    }

    void set_nnz_local(int nnz) { nnz_local = nnz; }

    T* get_lap_A() { return lap_A; }
    int* get_lap_row_ptr() { return lap_row_ptr; }
    int* get_lap_col_idx() { return lap_col_idx; }
    int get_lap_nnz_local() { return lap_nnz_local; }

    void set_lap_A(T *lap_A)
    {
      assert(lap_A != NULL);
      this -> lap_A = lap_A;
    }

    void set_lap_row_ptr(int *lap_row_ptr)
    {
      assert(lap_row_ptr != NULL);
      this -> lap_row_ptr = lap_row_ptr;
    }

    void set_lap_col_idx(int *lap_col_idx)
    {
      assert(lap_col_idx != NULL);
      this -> lap_col_idx = lap_col_idx;
    }

    void set_lap_nnz_local(int lap_nnz) { lap_nnz_local = lap_nnz; }

};

template <typename T>
extern distributed_graph<T>* create_from_edgelist_file(std::ifstream &fin);

