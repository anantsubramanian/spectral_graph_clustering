#include <stdio.h>
#include <cublas.h>

extern void call_me_maybe( float *vecA, float *vecB, int N, int M)
{
  cudaSetDevice(0);

  int i,j;
  float *devPtrA , *devPtrB , *devPtrC;

  //timer stuff
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);


  printf ("\nvecA\n");
  for(i=0;i<N;i++) printf("%f ",vecA[i]);
  printf ("\nvecB\n");
  for(i=0;i<N;i++) printf("%f ",vecB[i]);

  printf ("\n\n");


  cublasInit(); // initilization of CUDA application
  cublasAlloc( N, sizeof(float), (void**) &devPtrA);	//matrix A
  cublasAlloc( N, sizeof(float), (void**) &devPtrB);		//vector B
  cublasAlloc( N, sizeof(float), (void**) &devPtrC);		//vector C


  // transfer host data to device
  cublasSetVector( N, sizeof(float), vecA, 1, devPtrA, 1);
  cublasSetVector( N, sizeof(float), vecB, 1, devPtrB, 1);

  // compute C = A*B in device
  float alpha = 1.0;
  float beta  = 0.0;

  //start timer 
  cudaEventRecord(start, 0);
  /*cublasSgemv('N', M, N, alpha, devPtrA, M, devPtrB, 1, beta, devPtrC, 1);*/
  /*cublasSdot(N, devPtrA, 1, devPtrB, 1);*/

    cublasHandle_t handle;
    int n = 5;
  /*cublasStatus_t cublasSdot (handle, n, devPtrA, 1, devPtrB, 1, devPtrC);*/

  // block until the device has completed
  cudaThreadSynchronize();

  //end timer
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);		

  cublasGetVector (N, sizeof(float), devPtrB, 1, vecB, 1);


  for (j = 0; j < N; j++) printf ("%7.0f", vecB[j]);//IDX2C(j,1,M)]);
  printf("\n\ntime = %f\n\n",elapsedTime);

  cublasShutdown(); 
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}
