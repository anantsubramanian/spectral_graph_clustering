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
void sparse_csc_mdotv ( T *data, int *col_ptr, int *row_idx, int N, int M, T *vec,
                        int vec_size, T *result );

template <typename T>
T dense_vdotv ( T *vec1, int vec_sizes, T *vec2 );

template <typename T>
void daxpy ( T *result, T a, T *x, T *y, int vec_sizes );

template <typename T>
void qr_eigen ( T *alpha, T *beta, int N, const T epsilon = 1e-8 );

extern "C" void dstemr_ ( char *jobz, char *range, int *n, double *alpha, double *beta,
    double *vl, double *vu, int *il, int *iu, int *m, double *w, double *z, int *ldz,
    int *nzc, int *isuppz, int *tryrac, double *work, int *lwork, int *iwork,
    int *liwork, int *info );

extern void lapack_eigen (
    double *alpha, double *beta, int M, int &M_out,
    double **eigen_values, double **eigen_vectors );

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
    float *data, int *col_ptr, int *row_idx, int N, int M, float *vec, int vec_size,
    float *result );

extern void sparse_csc_mdotv (
    double *data, int *col_ptr, int *row_idx, int N, int M, double *vec, int vec_size,
    double *result );

extern float dense_vdotv ( float *vec1, int vec_sizes, float *vec2 );

extern double dense_vdotv ( double *vec1, int vec_sizes, double *vec2 );

extern void daxpy ( float *result, float a, float *x, float *y, int vec_sizes );

extern void daxpy ( double *result, double a, double *x, double *y, int vec_sizes );

extern void qr_eigen ( float *alpha, float *beta, int N, const float epsilon );

extern void qr_eigen ( double *alpha, double *beta, int N, const double epsilon );

