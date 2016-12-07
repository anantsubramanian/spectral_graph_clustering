#include <vector>

using std::vector;

template <typename T>
void convert_edgelist_to_csr ( vector<T> &data, vector<int> &vis, vector<int> &vjs,
                               int row_count, int row_base, T** A, int **row_ptr,
                               int **col_idx );

template <typename T>
void convert_edgelist_to_csc ( vector<T> &data, vector<int> &vis, vector<int> &vjs,
                               int col_count, int col_base, T** A, int **col_ptr,
                               int **row_idx );

template <typename T>
void sparse_csr_mdotv ( T *data, int *row_ptr, int *col_idx, int N, T *vec, int vec_size,
                        T *result );

template <typename T>
void sparse_csc_mdotv ( T *data, int *col_ptr, int *row_idx, int N, T *vec, int vec_size,
                        T *result );

template <typename T>
T dense_vdotv ( T *vec1, int vec_sizes, T *vec2 );

template <typename T>
void daxpy ( T *result, T a, T *x, T *y, int vec_sizes );

extern void convert_edgelist_to_csr (
    vector<float> &data, vector<int> &vis, vector<int> &vjs,
    int row_count, int row_base, float** A, int **row_ptr,
    int **col_idx );

extern void convert_edgelist_to_csr (
    vector<double> &data, vector<int> &vis, vector<int> &vjs,
    int row_count, int row_base, double** A, int **row_ptr,
    int **col_idx );

extern void convert_edgelist_to_csc (
    vector<float> &data, vector<int> &vis, vector<int> &vjs,
    int col_count, int col_base, float** A, int **col_ptr,
    int **row_idx );

extern void convert_edgelist_to_csc (
    vector<double> &data, vector<int> &vis, vector<int> &vjs,
    int col_count, int col_base, double** A, int **col_ptr,
    int **row_idx );

extern void sparse_csr_mdotv (
    float *data, int *row_ptr, int *col_idx, int N, float *vec, int vec_size,
    float *result );

extern void sparse_csr_mdotv (
    double *data, int *row_ptr, int *col_idx, int N, double *vec, int vec_size,
    double *result );

extern void sparse_csc_mdotv (
    float *data, int *col_ptr, int *row_idx, int N, float *vec, int vec_size,
    float *result );

extern void sparse_csc_mdotv (
    double *data, int *col_ptr, int *row_idx, int N, double *vec, int vec_size,
    double *result );

extern float dense_vdotv ( float *vec1, int vec_sizes, float *vec2 );

extern double dense_vdotv ( double *vec1, int vec_sizes, double *vec2 );

extern void daxpy ( float *result, float a, float *x, float *y, int vec_sizes );

extern void daxpy ( double *result, double a, double *x, double *y, int vec_sizes );

