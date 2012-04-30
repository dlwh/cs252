/*
 *  ising_exact.c
 *  gpuep
 *
 *  Created by Alex K on 4/30/12.
 *  Copyright 2012 UC Berkeley. All rights reserved.
 *
 */

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <assert.h>

#include "hello.h"
#include "ising_exact.h"

void exact_marginals(ising_t* result, ising_t model){
	construct_ising(result, model.rows, model.cols);
	assert(model.rows * model.cols < 32);
	int N=pow(2, model.rows * model.cols);
	double* marginals0 = (double*) calloc(model.rows * model.cols, sizeof(double));
	double* marginals1 = (double*) calloc(model.rows * model.cols, sizeof(double));
	
	for(int i = 0; i < N; i++){
		float sum = 0;
		for(int c = 0; c < model.cols; c++){
			for(int r = 0; r < model.rows; r++){
				if((i >> r * model.cols + c) & 1){
					sum += model.singleton[r * model.cols + c];
					if(r + 1 < model.rows && (i >> ((r + 1) * model.cols + c) & 1)){
						sum+=model.pair[(r * model.cols + c) * 2];
					}
					if(c + 1 < model.cols && (i >> (r * model.cols + c + 1) & 1)){
						sum+=model.pair[(r * model.cols + c) * 2 + 1];
					}
				}
			}
		}
		double pot=expf(sum);
		
		for(int c = 0; c < model.cols; c++){
			for(int r = 0; r < model.rows; r++){
				if((i >> r * model.cols + c) & 1){
					marginals0[r * model.cols + c] += pot;
				}else{
					marginals1[r * model.cols + c] += pot;
				}
			}
		}
	}
	
	for(int i = 0; i < model.rows * model.cols; i++){
		result->singleton[i] = (float) log(marginals0[i] / marginals1[i]);
	}
	
	free(marginals0);
	free(marginals1);
}

int exact_marginals_parallel(ising_t* result, ising_t model, cl_context context, cl_device_id device_id){
	construct_ising(result, model.rows, model.cols);
	assert(model.rows * model.cols < 32);
	int N=pow(2, model.rows * model.cols);
	double* marginals0 = (double*) calloc(model.rows * model.cols, sizeof(double));
	double* marginals1 = (double*) calloc(model.rows * model.cols, sizeof(double));
	char* KernelSource=read_kernel("kernel_exact.cl");
	
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
    cl_kernel kernelIter = clCreateKernel(program, "one_configuration", &err);
    if (!kernelIter || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        return 1;
    }
	
	int count = model.rows * model.cols;
    
    cl_mem ising_pair = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * count * 2, NULL, NULL);
    cl_mem ising_single = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * count, NULL, NULL);
    cl_mem clmarginals0 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * count, NULL, NULL);
    cl_mem clmarginals1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * count, NULL, NULL);
	
	err = clEnqueueWriteBuffer(commands, ising_pair, CL_TRUE, 0, sizeof(float) * count * 2, model.pair, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commands, ising_single, CL_TRUE, 0, sizeof(float) * count, model.singleton, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commands, clmarginals0, CL_TRUE, 0, sizeof(float) * count, marginals0, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commands, clmarginals1, CL_TRUE, 0, sizeof(float) * count, marginals1, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        return 1;
    }
	
	size_t global[] = {N};
	size_t local;
	
	err = clGetKernelWorkGroupInfo(kernelIter, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        return 1;
    }
	
	err  = clSetKernelArg(kernelIter, 0, sizeof(cl_mem), &ising_pair);
	err |= clSetKernelArg(kernelIter, 1, sizeof(cl_mem), &ising_single);
	err |= clSetKernelArg(kernelIter, 2, sizeof(cl_mem), &clmarginals0);
	err |= clSetKernelArg(kernelIter, 3, sizeof(cl_mem), &clmarginals1);
	err |= clSetKernelArg(kernelIter, 4, sizeof(int), &model.rows);
	err |= clSetKernelArg(kernelIter, 5, sizeof(int), &model.cols);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to set kernel arguments! %d\n", err);
		return 1;
	}
	
	cl_event configuration_event;
	err = clEnqueueNDRangeKernel(commands, kernelIter, 1, NULL, global, NULL, 0, NULL, &configuration_event);
	if (err)
	{
		printf("Error: Failed to execute kernel!\n");
		return EXIT_FAILURE;
	}
	
	clFinish(commands);

	// Read back the results from the device to verify the output
	//
	err = clEnqueueReadBuffer(commands, clmarginals0, CL_TRUE, 0, sizeof(double) * count, marginals0, 0, NULL, NULL );
	err |= clEnqueueReadBuffer(commands, clmarginals1, CL_TRUE, 0, sizeof(double) * count, marginals1, 0, NULL, NULL ); 
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}
	
	
	for(int i = 0; i < model.rows * model.cols; i++){
		result->singleton[i] = (float) log(marginals0[i] / marginals1[i]);
	}
	
	free(marginals0);
	free(marginals1);
	
	return 0;
}