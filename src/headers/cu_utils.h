#ifndef __CU_UTILS
  #define __CU_UTILS
#endif

#include <fstream>
#include <iostream>
#include <cassert>

#include "cusparse.h"

void cudaTscal (int vectorSize, float *alpha, float *vector);

cublasStatus_t cublasTscal (
    cublasHandle_t handle, int n, const float *alpha,
    float *x, int incx );

cublasStatus_t cublasTscal (
    cublasHandle_t handle, int n, const double *alpha, double *x, int incx);

cublasStatus_t cublasTdot (
    cublasHandle_t handle, int n, const float *x, int incx, const float *y,
    int incy, float *result );

cublasStatus_t cublasTdot (
    cublasHandle_t handle, int n, const double *x, int incx, const double *y,
    int incy, double *result );

cublasStatus_t cublasTaxpy (
    cublasHandle_t handle, int n, const float *alpha, const float *x, int incx,
    float *y, int incy );

cublasStatus_t cublasTaxpy (
    cublasHandle_t handle, int n, const double *alpha, const double *x, int incx,
    double *y, int incy );

cusparseStatus_t cusparseTcsrmv (
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int nnz,
    const float *alpha, const cusparseMatDescr_t descrA, const float *csrValA,
    const int *csrRowPtrA, const int *csrColIndA, const float *x, const float *beta,
    float *y );

cusparseStatus_t cusparseTcsrmv (
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int nnz,
    const double *alpha, const cusparseMatDescr_t descrA, const double *csrValA,
    const int *csrRowPtrA, const int *csrColIndA, const double *x, const double *beta,
    double *y );

cusparseStatus_t cusparseTcsr2csc(
    cusparseHandle_t handle, int m, int n, int nnz, const float *csrVal,
    const int *csrRowPtr, const int *csrColInd, float *cscVal, int *cscRowInd,
    int *cscColPtr, cusparseAction_t copyValues, cusparseIndexBase_t idxBase);

cusparseStatus_t cusparseTcsr2csc(
    cusparseHandle_t handle, int m, int n, int nnz, const double *csrVal,
    const int *csrRowPtr, const int *csrColInd, double *cscVal, int *cscRowInd,
    int *cscColPtr, cusparseAction_t copyValues, cusparseIndexBase_t idxBase);

