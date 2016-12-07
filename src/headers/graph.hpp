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
class distributed_graph_csr
{
  private:
    int N; // Number of nodes
    int M; // Number of edges (synonymous to NNZ for the whole matrix)

    int rows_per_node; // Ideal number of rows in each node (when exactly divisible)
    int rows_in_node;  // Actual number of rows in this node
    int task_id;

    // The start index offset for this node in the matrix
    // i.e. row 0 of this matrix corresponds to row_start_idx
    // of the whole matrix.
    int row_start_idx;

    T *A;      // Data (size = nnz_local)
    int *row_ptr; // Row pointers (size rows_in_node)
    int *col_idx; // Column indices of the values in A
    int nnz_local;  // The number of nonzero values in this submatrix

    // Similar variables, but for the Laplacian
    T* lap_A;
    int *lap_row_ptr;
    int *lap_col_idx;
    int lap_nnz_local;

  public:

    distributed_graph_csr(int N, int M, int rows_per_node, int rows_in_node, int task_id)
    {
      this -> N = N;
      this -> M = M;
      this -> rows_per_node = rows_per_node;
      this -> rows_in_node = rows_in_node;
      this -> task_id = task_id;

      this -> row_start_idx = rows_per_node * task_id;

      this -> A = NULL;
      this -> row_ptr = NULL;
      this -> col_idx = NULL;

      this -> lap_A = NULL;
      this -> lap_row_ptr = NULL;
      this -> lap_col_idx = NULL;
    }

    ~distributed_graph_csr()
    {
      if (A) delete A;
      if (row_ptr) delete row_ptr;
      if (col_idx) delete col_idx;
      if (lap_A) delete lap_A;
      if (lap_row_ptr) delete lap_row_ptr;
      if (lap_col_idx) delete lap_col_idx;
    }

    int get_N() { return N; }
    int get_M() { return M; }
    int get_rows_per_node() { return rows_per_node; }
    int get_rows_in_node() { return rows_in_node; }
    int get_row_start_index() { return row_start_idx; }

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

/**
 * Class that stores the distributed representation of a graph,
 * distributed across nodes, and stored in CSC format.
 */
template <typename T>
class distributed_graph_csc
{
  private:
    int N; // Number of nodes
    int M; // Number of edges (synonymous to NNZ for the whole matrix)

    int cols_per_node; // Ideal number of cols in each node (when exactly divisible)
    int cols_in_node;  // Actual number of cols in this node
    int task_id;

    // The start index offset for this node in the matrix
    // i.e. col 0 of this matrix corresponds to col_start_idx
    // of the whole matrix.
    int col_start_idx;

    T *A;      // Data (size = nnz_local)
    int *col_ptr; // Column pointers (size cols_in_node)
    int *row_idx; // Row indices of the values in A
    int nnz_local;  // The number of nonzero values in this submatrix

    // Similar variables, but for the Laplacian
    T* lap_A;
    int *lap_col_ptr;
    int *lap_row_idx;
    int lap_nnz_local;

  public:

    distributed_graph_csc(int N, int M, int cols_per_node, int cols_in_node, int task_id)
    {
      this -> N = N;
      this -> M = M;
      this -> cols_per_node = cols_per_node;
      this -> cols_in_node = cols_in_node;
      this -> task_id = task_id;

      this -> col_start_idx = cols_per_node * task_id;

      this -> A = NULL;
      this -> col_ptr = NULL;
      this -> row_idx = NULL;

      this -> lap_A = NULL;
      this -> lap_col_ptr = NULL;
      this -> lap_row_idx = NULL;
    }

    ~distributed_graph_csc()
    {
      if (A) delete A;
      if (col_ptr) delete col_ptr;
      if (row_idx) delete row_idx;
      if (lap_A) delete lap_A;
      if (lap_col_ptr) delete lap_col_ptr;
      if (lap_row_idx) delete lap_row_idx;
    }

    int get_N() { return N; }
    int get_M() { return M; }
    int get_cols_per_node() { return cols_per_node; }
    int get_cols_in_node() { return cols_in_node; }
    int get_col_start_index() { return col_start_idx; }

    void construct_unnormalized_laplacian();
    void free_adjacency_matrix();

    T* get_A() { return A; }
    int* get_col_ptr() { return col_ptr; }
    int* get_row_idx() { return row_idx; }
    int get_nnz_local() { return nnz_local; }

    void set_A(T *A)
    {
      assert(A != NULL);
      this -> A = A;
    }

    void set_col_ptr(int *col_ptr)
    {
      assert(col_ptr != NULL);
      this -> col_ptr = col_ptr;
    }

    void set_row_idx(int *row_idx)
    {
      assert(row_idx != NULL);
      this -> row_idx = row_idx;
    }

    void set_nnz_local(int nnz) { nnz_local = nnz; }

    T* get_lap_A() { return lap_A; }
    int* get_lap_col_ptr() { return lap_col_ptr; }
    int* get_lap_row_idx() { return lap_row_idx; }
    int get_lap_nnz_local() { return lap_nnz_local; }

    void set_lap_A(T *lap_A)
    {
      assert(lap_A != NULL);
      this -> lap_A = lap_A;
    }

    void set_lap_col_ptr(int *lap_col_ptr)
    {
      assert(lap_col_ptr != NULL);
      this -> lap_col_ptr = lap_col_ptr;
    }

    void set_lap_row_idx(int *lap_row_idx)
    {
      assert(lap_row_idx != NULL);
      this -> lap_row_idx = lap_row_idx;
    }

    void set_lap_nnz_local(int lap_nnz) { lap_nnz_local = lap_nnz; }

};

template <typename T>
extern distributed_graph_csr<T>* create_csr_from_edgelist_file(std::ifstream &fin);

template <typename T>
extern distributed_graph_csc<T>* create_csc_from_edgelist_file(std::ifstream &fin);

