#include <stdio.h>
#include "cublas_v2.h"
/*#include <cublas.h>*/


/**
 * Computes a*x + y for two dense vectors x and y.
 */

extern void cu_daxpy( float *result, float a, float *x, float *y, int vec_size )
{
  cudaSetDevice(0);

  /*int i,j;*/
  float *devPtrX , *devPtrY;

  //timer stuff
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  /*printf ("\nvecX\n");*/
  /*for(i=0;i<vec_size;i++) printf("%f ",x[i]);*/
  /*printf ("\nvecY\n");*/
  /*for(i=0;i<vec_size;i++) printf("%f ",y[i]);*/

  /*printf ("\n\n");*/

  cudaMalloc( (void**) &devPtrX, vec_size * sizeof(float));	//matrix A
  cudaMalloc( (void**) &devPtrY, vec_size * sizeof(float));		//vector B

  // transfer host data to device
  cublasSetVector( vec_size, sizeof(float), x, 1, devPtrX, 1);
  cublasSetVector( vec_size, sizeof(float), y, 1, devPtrY, 1);

  //start timer 
  cudaEventRecord(start, 0);

  // do saxpy
  /*cublasStatus_t stat;*/
  cublasHandle_t handle;

  cublasCreate(&handle);

  cublasSaxpy(handle, vec_size, &a, devPtrX, 1, devPtrY, 1);

  /*// block until the device has completed*/
  cudaThreadSynchronize();

  //end timer
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);		

  cublasGetVector (vec_size, sizeof(float), devPtrY, 1, y, 1);

  /*for (j = 0; j < vec_size; j++) printf ("%7.0f", y[j]);//IDX2C(j,1,M)]);*/
  /*printf("\n\ntime = %f\n\n",elapsedTime);*/

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  result = y;

}
