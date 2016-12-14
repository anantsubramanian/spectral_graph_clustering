#include <cstdio>
#include "cublas_v2.h"
#include "headers/cu_utils.h"
#include "cusparse.h"

using namespace std;

#define threadsPerBlock 1024 // max threads on Tesla K40 GPU

// CUDA scaling kernel

template <class T>
__global__ void cuda_Tscal_kernel (int vectorSize, T alpha, T *vector)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < vectorSize)
    vector[idx] = alpha * vector[idx];
}

// CUDA - Vector scaling wrapper
// vector = alpha * vector

template <class T>
void cudaTscal (int vectorSize, T *alpha, T *vector)
{
  int numBlocks = (vectorSize + threadsPerBlock - 1) / threadsPerBlock;

  cuda_Tscal_kernel<float><<< numBlocks, threadsPerBlock>>>( vectorSize, *alpha, vector);
  cudaThreadSynchronize();
}


// cuBLAS - Vector scaling wrappers
// x = alpha * x; incx - stride, n - number of elements

cublasStatus_t cublasTscal (
    cublasHandle_t handle, int n, const float *alpha, float *x,
    int incx )
{
  return cublasSscal(handle, n, alpha, x, incx);
}


cublasStatus_t cublasTscal (
    cublasHandle_t handle, int n, const double *alpha, double *x,
    int incx )
{
  return cublasDscal(handle, n, alpha, x, incx);
}


// cuBLAS - Dot product wrapper
// result = x.y; incx and incy - strides, n - number of elements

cublasStatus_t cublasTdot (
    cublasHandle_t handle, int n, const float *x, int incx, const float *y, int incy,
    float *result )
{
  return cublasSdot (handle, n, x, incx, y, incy, result);
}

cublasStatus_t cublasTdot (
    cublasHandle_t handle, int n, const double *x, int incx, const double *y, int incy,
    double *result )
{
  return cublasDdot (handle, n, x, incx, y, incy, result);
}


// cuBLAS - axpy wrapper
// y = alpha * x + y; incx and incy - strides, n - number of elements

cublasStatus_t cublasTaxpy (
    cublasHandle_t handle, int n, const float *alpha, const float *x, int incx, float *y,
    int incy )
{
  return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}

cublasStatus_t cublasTaxpy (
    cublasHandle_t handle, int n, const double *alpha, const double *x, int incx, double *y,
    int incy )
{
  return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}


// cuSparse - Sparse CSR Matrix Vector Multiplication wrapper
// y = alpha * matrixA * x + beta * y; m and n - dimensions, nnz - number of non-zero elements

cusparseStatus_t cusparseTcsrmv(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int nnz,
    const float *alpha, const cusparseMatDescr_t descrA, const float *csrValA,
    const int *csrRowPtrA, const int *csrColIndA, const float *x, const float *beta,
    float *y )
{
  return cusparseScsrmv(
      handle, transA, m, n, nnz, alpha, descrA, csrValA, csrRowPtrA,
      csrColIndA, x, beta, y);
}

cusparseStatus_t cusparseTcsrmv(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int nnz,
    const double *alpha, const cusparseMatDescr_t descrA, const double *csrValA,
    const int *csrRowPtrA, const int *csrColIndA, const double *x,
    const double *beta, double *y )
{
  return cusparseDcsrmv(
      handle, transA, m, n, nnz, alpha, descrA, csrValA, csrRowPtrA,
      csrColIndA, x, beta, y);
}


// cuSparse - Convert CSR/CSC matrix a CSC/CSR matrix resp.

cusparseStatus_t cusparseTcsr2csc(
    cusparseHandle_t handle, int m, int n, int nnz, const float *csrVal,
    const int *csrRowPtr, const int *csrColInd, float *cscVal, int *cscRowInd,
    int *cscColPtr, cusparseAction_t copyValues, cusparseIndexBase_t idxBase)
{
  return cusparseScsr2csc(
      handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscRowInd,
      cscColPtr, copyValues, idxBase);
}

cusparseStatus_t cusparseTcsr2csc(
    cusparseHandle_t handle, int m, int n, int nnz, const double *csrVal,
    const int *csrRowPtr, const int *csrColInd, double *cscVal, int *cscRowInd,
    int *cscColPtr, cusparseAction_t copyValues, cusparseIndexBase_t idxBase)
{
  return cusparseDcsr2csc(
      handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscRowInd,
      cscColPtr, copyValues, idxBase);
}

