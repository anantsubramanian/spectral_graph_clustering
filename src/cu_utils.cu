#include <stdio.h>
#include "cublas_v2.h"
#include "headers/cu_utils.h"

using namespace std;

#define threadsPerBlock 1024

__global__ void cuda_Sdot_kernel(int vectorSize, float *vecA, float *vecB, float *vecResult)
{

  __shared__ float cache[threadsPerBlock];

  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  int cacheIdx = threadIdx.x;

  float local_sum = 0;

  while (idx < vectorSize)
  {
    local_sum += vecA[idx] * vecB[idx];
    idx += blockDim.x * gridDim.x;
  }

  cache[cacheIdx] = local_sum;
  __syncthreads();

  // reduce

  int i = blockDim.x/2;
  while (i != 0)
  {
    if (cacheIdx < i)
      cache[cacheIdx] += cache[cacheIdx + i];

    __syncthreads();
    i/= 2;
  }

  if (cacheIdx == 0)
    vecResult[blockIdx.x] = cache[0];
}

__global__ void dot_product_kernel(const int N, const float *x, const float *y,
    float *z) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float result[threadsPerBlock];

    if (index < N) {
        result[threadIdx.x] = x[index] * y[index];
    } else {
        result[threadIdx.x] = 0;
    }
    __syncthreads();

    int half = threadsPerBlock/ 2;
    while (half > 0) {
        if (threadIdx.x < half) {
            result[threadIdx.x] += result[threadIdx.x + half];
        }
        __syncthreads();
        half /= 2;
    }

    if (threadIdx.x == 0) {
        z[blockIdx.x] = result[0];
    }
}

void cudaSdot(int vectorSize, float *vecA, float skipA, float *vecB, float skipB, float *result)
{
   dim3 threads(threadsPerBlock);
   dim3 blocks(32);

   float *devVecResult;
   float *vecResult;
   vecResult = new float[vectorSize];
   cudaMalloc( (void **) &devVecResult, vectorSize * sizeof(float));

   /*cuda_Sdot_kernel<<< blocks, threads>>>( size, vecA, vecB, devVecResult );*/
   dot_product_kernel<<< blocks, threads>>>( vectorSize, vecA, vecB, devVecResult );

   cudaMemcpy(vecResult, devVecResult, sizeof(float), cudaMemcpyDeviceToHost);

   *result = 0;

   for (int i = 0; i < vectorSize; i ++)
     *result+= vecResult[i];


   cudaThreadSynchronize();
}

cublasStatus_t cublasTscal(cublasHandle_t handle, int n, const float *alpha, float *x, int incx)
{
    return cublasSscal(handle, n, alpha, x, incx);
}


cublasStatus_t cublasTscal(cublasHandle_t handle, int n, const double *alpha, double *x, int incx)
{
    return cublasDscal(handle, n, alpha, x, incx);
}

__global__ void cuda_Sscal_kernel(int vectorSize, float alpha, float *vector)
{

  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < vectorSize)
    vector[idx] = alpha * vector[idx];
}

void cudaSscal(int vectorSize, float *alpha, float *vector)
{
   int numBlocks = (vectorSize + threadsPerBlock - 1) / threadsPerBlock;

   cuda_Sscal_kernel<<< numBlocks, threadsPerBlock>>>( vectorSize, *alpha, vector);
   cudaThreadSynchronize();
}


cublasStatus_t cublasTdot (cublasHandle_t handle, int n, const float *x, int incx, const float *y, int incy, float *result)
{
  return cublasSdot (handle, n, x, incx, y, incy, result);
}

cublasStatus_t cublasTdot (cublasHandle_t handle, int n, const double *x, int incx, const double *y, int incy, double *result)
{
  return cublasDdot (handle, n, x, incx, y, incy, result);
}

cublasStatus_t cublasTaxpy(cublasHandle_t handle, int n, const float *alpha, const float *x, int incx, float *y, int incy)
{
  return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}

cublasStatus_t cublasTaxpy(cublasHandle_t handle, int n, const double *alpha, const double *x, int incx, double *y, int incy)
{
  return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}

