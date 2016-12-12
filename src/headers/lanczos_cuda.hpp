// See the corresponding implementation file for details about the parameters.

template <typename T>
void lanczos_csr_cuda ( T *data, int *row_ptr, int *col_idx, int nnz, int N, int rows_in_node,
                   int rows_per_node, int local_start_index, int M,
                   MPI_Datatype mpi_datatype, T **alpha_out, T **beta_out,
                   T ***v_out, T eta = 0.75 );

