//
//  ising.c
//  gpuep
//
//  Created by David Hall on 4/24/12.
//  Copyright (c) 2012 UC Berkeley. All rights reserved.
//

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <OpenCL/opencl.h>
#include "hello.h"
#include "ising.h"
#include <math.h>
#include <assert.h>

void construct_ising(ising_t *ising, int rows, int cols) {
    ising->rows = rows;
    ising->cols = cols;
    ising->singleton = (float*)malloc(sizeof(float) * rows * cols);
    memset(ising->singleton, 0, sizeof(float) * rows * cols);
    ising->pair = (float*)malloc(sizeof(float) * rows * cols * 2);
    memset(ising->pair, 0, sizeof(float) * rows * cols * 2);
}

void destroy_ising(ising_t *ising) {
    free(ising->singleton);
    free(ising->pair);
}

void random_fill_ising(ising_t *ising, float lowerBound, float upperBound, unsigned* seed) {
    int size = ising->rows * ising->cols;
    for(int i = 0; i < size; ++i) {
        ising->singleton[i] = (float)rand_r(seed)/RAND_MAX * (upperBound - lowerBound) + lowerBound;
        ising->pair[i * 2] = (float)rand_r(seed)/RAND_MAX * (upperBound - lowerBound) + lowerBound;
        ising->pair[i * 2 + 1] = (float)rand_r(seed)/RAND_MAX * (upperBound - lowerBound) + lowerBound;
    }
    
}

int do_inference(ising_t* result, ising_t model, cl_context context, cl_device_id device_id, int numIter) {
    construct_ising(result, model.rows, model.cols);
    char* KernelSource=read_kernel("kernel.cl");
    
    int err;
    
    cl_command_queue commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
		printf("%i %i\n", CL_INVALID_VALUE, err);
        return EXIT_FAILURE;
    }
    
    
    cl_program program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
    if (!program)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }
    
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];
		
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }
    
    // Create the compute kernel in the program we wish to run
    //
    cl_kernel kernelInf = clCreateKernel(program, "updateFactor", &err);
    if (!kernelInf || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        return 1;
    }
    cl_kernel kernelMarg = clCreateKernel(program, "updateMarginals", &err);
    if (!kernelMarg || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel marg!\n");
        return 1;
    }
    
    unsigned count = model.rows * model.cols;
    
    cl_mem pair = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * count * 2, NULL, NULL);
    cl_mem single = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * count, NULL, NULL);
    cl_mem single_out = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * count, NULL, NULL);
    cl_mem message1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * count * 4, NULL, NULL);
    float* ising_message = malloc(sizeof(float) * model.rows * model.cols * 4);
    memset(ising_message, 0, sizeof(float) * model.rows * model.cols * 4);
    cl_mem message2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * count * 4, NULL, NULL);
    
    err = clEnqueueWriteBuffer(commands, pair, CL_TRUE, 0, sizeof(float) * count * 2, model.pair, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commands, single, CL_TRUE, 0, sizeof(float) * count, model.singleton, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commands, single_out, CL_TRUE, 0, sizeof(float) * count, model.singleton, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commands, message1, CL_TRUE, 0, sizeof(float) * count, ising_message, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commands, message2, CL_TRUE, 0, sizeof(float) * count, ising_message, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        return 1;
    }
    free(ising_message);
    
    size_t global[] = {model.rows, model.cols, 2};                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation
    
    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(kernelInf, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        return 1;
    }
    
    for(int iter = 0; iter < numIter; ++iter) {
        err  = clSetKernelArg(kernelInf, 0, sizeof(cl_mem), &pair);
        err |= clSetKernelArg(kernelInf, 1, sizeof(cl_mem), &single_out);
        err |= clSetKernelArg(kernelInf, 2, sizeof(cl_mem), &message1);
        err |= clSetKernelArg(kernelInf, 3, sizeof(cl_mem), &message2);
        err |= clSetKernelArg(kernelInf, 4, sizeof(int), &model.rows);
        err |= clSetKernelArg(kernelInf, 5, sizeof(int), &model.cols);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to set kernel arguments! %d\n", err);
            return 1;
        }
        
        cl_event inference_event;
        err = clEnqueueNDRangeKernel(commands, kernelInf, 3, NULL, global, NULL, 0, NULL, &inference_event);
        if (err)
        {
            printf("Error: Failed to execute kernel!\n");
            return EXIT_FAILURE;
        }
        
        err  = clSetKernelArg(kernelMarg, 0, sizeof(cl_mem), &single);
        err  = clSetKernelArg(kernelMarg, 1, sizeof(cl_mem), &message2);
        err |= clSetKernelArg(kernelMarg, 2, sizeof(cl_mem), &single_out);
        err |= clSetKernelArg(kernelMarg, 3, sizeof(int), &model.rows);
        err |= clSetKernelArg(kernelMarg, 4, sizeof(int), &model.cols);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to set kernel arguments! %d\n", err);
            return 1;
        }

        err = clEnqueueNDRangeKernel(commands, kernelMarg, 3, NULL, global, NULL, 1, &inference_event, NULL);
        if (err)
        {
            printf("Error: Failed to execute kernel!\n");
            return EXIT_FAILURE;
        }
        
        clFinish(commands);
        
        cl_mem temp = message2;
        message2 = message1;
        message1 = temp;
    }
    
    // Read back the results from the device to verify the output
    //
    err = clEnqueueReadBuffer( commands, single_out, CL_TRUE, 0, sizeof(float) * count, result->singleton, 0, NULL, NULL );  
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }
    
    
    return 0;
    
}

void ising_print(ising_t ising) {
    for(int r = 0; r < ising.rows; ++r) {
        for(int c = 0; c < ising.cols; ++c) {
            printf("(%d, %d) %.3f\n", r, c, get_ising_singleton(&ising, r, c));
        }
        printf("\n");
    }
    
    for(int r =0; r < ising.rows; ++r) {
        for(int c = 0; c < ising.cols; ++c) {
            for(int dir = 0; dir < 2; ++dir) {
                int nr = dir == IM_DOWN ? r + 1 : r;
                int nc = dir == IM_DOWN ? c : c + 1;    
                printf("(%d, %d) -> (%d, %d) %.3f\n", r, c, nr, nc, get_ising_pair(&ising, r, c, dir));
            }
        }
        printf("\n");
    }
}

int sequential_inference(ising_t* result, ising_t model, int numIter) {
    construct_ising(result, model.rows, model.cols);
    float* ising_single = result->singleton;
    memcpy(result->singleton, model.singleton, model.rows * model.cols * sizeof(float));
    
    float* ising_pair = model.pair;
    float* ising_message = malloc(sizeof(float) * model.rows * model.cols * 4);
    float* ising_message_out = malloc(sizeof(float) * model.rows * model.cols * 4);
    
    memset(ising_message, 0, sizeof(float) * model.rows * model.cols * 4);
    memset(ising_message_out, 0, sizeof(float) * model.rows * model.cols * 4);
    
    int cols = model.cols;
    int rows = model.rows;
    for(int iter = 0; iter < numIter; ++iter) {
        for(int r = 0; r < model.rows; ++r) {
            for(int c = 0; c < model.cols; ++c) {
                for(int dir = 0; dir < 2; ++dir) {
                    int nr = dir == IM_DOWN ? r + 1 : r;
                    int nc = dir == IM_DOWN ? c : c + 1;
                    int otherDir = dir == IM_DOWN ? IM_UP : IM_LEFT;
                    
                    if(nr < rows && nc < cols) {
                        float marginalWeightA = ising_single[r * cols + c];
                        float marginalWeightB = ising_single[nr * cols + nc];
                        float edgeWeight = ising_pair[(r * cols + c) * 2 + dir];
                        float mesgToA = ising_message[(r * cols + c) * 4 + dir];
                        float mesgToB = ising_message[(nr * cols + nc) * 4 + otherDir];
                        assert(iter == 0 || mesgToA == 0);
                        assert(iter == 0 || mesgToB == 0);
                        marginalWeightA -= mesgToA;
                        marginalWeightB -= mesgToB;
                        
                        float jointMarginal11 = expf(marginalWeightA + marginalWeightB + edgeWeight);
                        float jointMarginal10 = expf(marginalWeightA);
                        float jointMarginal01 = expf(marginalWeightB);
                        float sum = jointMarginal11 + jointMarginal10 + jointMarginal01 + 1;
                        
                        float newMargA = (jointMarginal11 + jointMarginal10)/sum;
                        float newMargB = (jointMarginal11 + jointMarginal01)/sum;
                        
                        // logit function
                        float newTargetA = logf(newMargA/(1-newMargA));
                        float newTargetB = logf(newMargB/(1-newMargB));
                        float adjA = newTargetA - marginalWeightA;
                        float adjB = newTargetB - marginalWeightB;
                        
                        ising_message_out[(r * cols + c) * 4 + dir] = adjA;
                        ising_message_out[(nr * cols + nc) * 4 + otherDir] = adjB;
                        assert(adjA < 1/0.0);
                        assert(adjB < 1/0.0);
                    }
                }
            }
        }
        
        float* ising_marginal_out = result->singleton;
        
        for(int r = 0; r < model.rows; ++r) {
            for(int c = 0; c < model.cols; ++c) {
                int offset = (r * cols + c) * 4;
                float result = model.singleton[offset/4];
                assert(result < 1/0.0);
                for(int i = 0; i < 4; ++i) {
                    result += ising_message_out[offset + i];
                    assert(result < 1/0.0);
                    assert(result == result);
                }
                ising_marginal_out[offset/4] = result;
            }
        }
    }
    free(ising_message);
    free(ising_message_out);
    return 0;
}