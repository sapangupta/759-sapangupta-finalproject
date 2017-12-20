//
// Created by Nikhil S. Nakhate on 12/19/17.
//

/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.
 *
 * This software and the information contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a Non-Disclosure Agreement.  Any reproduction or
 * disclosure to any third party without the express written consent of
 * NVIDIA is prohibited.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.  This source code is a "commercial item" as
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer software" and "commercial computer software
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 */

/* Matrix multiplication: C = A * B.
 * Host code.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <fstream>
#include <ratio>
#include "stopwatch.hpp"
// includes, project
#include "2Dconvolution.h"

using namespace std;
////////////////////////////////////////////////////////////////////////////////
// declarations, forward

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int);

Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width,int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
bool CompareResults(float* A, float* B, int elements, float eps);
void FreeDeviceMatrix(Matrix* M);
void FreeMatrix(Matrix* M);

void ConvolutionOnDevice(const Matrix M, const Matrix N, Matrix P);

////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification
////////////////////////////////////////////////////////////////////////////////
__global__ void ConvolutionKernel(Matrix M, Matrix N, Matrix P)
{

    extern __shared__ float s[];

    int hB = N.height;
    int wB = N.width;

    if(threadIdx.x < 5 && threadIdx.y < 5) {
        s[threadIdx.y * 5 + threadIdx.x] = M.elements[threadIdx.y * 5 + threadIdx.x];
    }

    __syncthreads();

    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;

    double sum = 0;
    // check the start and end values of m and n to prevent overrunning the
    //  matrix edges
    unsigned int mbegin = (i < 1)? 1 - i : 0;
    unsigned int mend = (i > (hB - 2))?
                        hB - i + 1 : 3;
    unsigned int nbegin = (j < 1)? 1 - j : 0;
    unsigned int nend = (j > (wB - 3))?
                        (wB-j) + 1 : 3;
    // overlay A over B centered at element (i,j).  For each
    // overlapping element, multiply the two and accumulate
    for(unsigned int m = mbegin; m < mend; ++m) {
        for(unsigned int n = nbegin; n < nend; n++) {
            sum += s[m * 3 + n] *
                   N.elements[wB*(i + m - 1) + (j+n - 1)];
        }
    }
    // store the result
    P.elements[i*wB + j] = (float)sum;

}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

    Matrix  M;
    Matrix  N;
    Matrix  P;


    if(argc != 2)
    {
        printf("Usage %s Size\n",argv[0]);
        return 1;

    }

    int size = atoi(argv[1]);
    M  = AllocateMatrix(KERNEL_SIZE, KERNEL_SIZE,1);
    N  = AllocateMatrix(size, size,1);
    P  = AllocateMatrix(size, size,0);
    float cpuTime = 0.f,gpuTime=0.f;

    cudaEvent_t startEvent_inc, stopEvent_inc;
    cudaEventCreate(&startEvent_inc);
    cudaEventCreate(&stopEvent_inc);
    cudaEventRecord(startEvent_inc, 0); // starting timing for inclusive

    // M * N on the device
    ConvolutionOnDevice(M, N, P);

    cudaEventRecord(stopEvent_inc, 0);  //ending timing for inclusive
    cudaEventSynchronize(stopEvent_inc);
    cudaEventElapsedTime(&gpuTime, startEvent_inc, stopEvent_inc);

    // compute the matrix multiplication on the CPU for comparison
    Matrix reference = AllocateMatrix(P.height, P.width,0);

    stopwatch<std::milli, float> sw;
    // Start the timer
    sw.start();

    computeGold(reference.elements, M.elements, N.elements, N.height, N.width);

    // stop the timer
    sw.stop();
    cpuTime = sw.count();

    // in this case check if the result is equivalent to the expected soluion

    bool res = CompareResults(reference.elements, P.elements, P.width * P.height, 0.01f);;
    if(res==0)printf("Test Failed\n"); // This shouldnt pront for correct implementation
    printf("%f\n%f\n%f\n",P.elements[size*size-1],cpuTime,gpuTime);

    // Free matrices
    FreeMatrix(&M);
    FreeMatrix(&N);
    FreeMatrix(&P);
    return 0;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void ConvolutionOnDevice(const Matrix M, const Matrix N, Matrix P)
{
    // Load M and N to the device
    Matrix Md = AllocateDeviceMatrix(M);
    CopyToDeviceMatrix(Md, M);
    Matrix Nd = AllocateDeviceMatrix(N);
    CopyToDeviceMatrix(Nd, N);

    // Allocate P on the device
    Matrix Pd = AllocateDeviceMatrix(P);
    CopyToDeviceMatrix(Pd, P);

    // Setup the execution configuration
    dim3 dimGrid((N.height + 15) / 16, (N.height + 15) / 16);
    dim3 dimBlock(16, 16);


    // Launch the device computation threads!
    ConvolutionKernel << <dimGrid, dimBlock, sizeof(float) * 25>> > (Md, Nd, Pd);

    // Read P from the device
    CopyFromDeviceMatrix(P, Pd);

    // Free device matrices
    FreeDeviceMatrix(&Md);
    FreeDeviceMatrix(&Nd);
    FreeDeviceMatrix(&Pd);

}

// Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrix(const Matrix M)
{
    Matrix Mdevice = M;
    int size = M.width * M.height * sizeof(float);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}

Matrix AllocateMatrix(int height, int width,int init) // 1 is file read/ 0 is just allocation
{
    Matrix M;
    M.width = M.pitch = width;
    M.height = height;
    int size = M.width * M.height;
    M.elements = NULL;
    FILE *fp;
    fp = fopen("problem1.inp","r");
    // don't allocate memory on option 2

    M.elements = (float*) malloc(size*sizeof(float));
    if(init)
    {
        for(unsigned int i = 0; i < M.height * M.width; i++)
        {
            fscanf(fp,"%f",&M.elements[i]);
        }
    }
    return M;
}

// Copy a host matrix to a device matrix.
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.width * Mhost.height * sizeof(float);
    Mdevice.height = Mhost.height;
    Mdevice.width = Mhost.width;
    Mdevice.pitch = Mhost.pitch;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size,
               cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)
{
    int size = Mdevice.width * Mdevice.height * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size,
               cudaMemcpyDeviceToHost);
}

// Free a device matrix.
void FreeDeviceMatrix(Matrix* M)
{
    cudaFree(M->elements);
    M->elements = NULL;
}

// Free a host Matrix
void FreeMatrix(Matrix* M)
{
    free(M->elements);
    M->elements = NULL;
}

//compare the data stored in two arrays on the host
bool CompareResults(float* A, float* B, int elements, float eps)
{
    for(unsigned int i = 0; i < elements; i++){
        float error = A[i]-B[i];
        if(error>eps){
            return false;
        }
    }
    return true;
}