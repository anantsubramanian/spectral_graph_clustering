// See the corresponding implementation file for details about the parameters.

template <typename T>
void lanczos_csr ( T *data, int *row_ptr, int *col_idx, int N, int rows_in_node,
                   int rows_per_node, int local_start_index, int M,
                   MPI_Datatype mpi_datatype, T **alpha_out, T **beta_out,
                   T ***v_out, T eta = 0.75 );

template <typename T>
void lanczos_csc ( T *data, int *col_ptr, int *row_idx, int N, int cols_in_node,
                   int cols_per_node, int local_start_index, int M,
                   MPI_Datatype mpi_datatype, T **alpha_out, T **beta_out,
                   T ***v_out, T eta = 0.75 );

