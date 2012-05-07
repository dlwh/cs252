/*
 *  measures.c
 *  gpuep
 *
 *  Created by Alex K on 5/3/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "measures.h"

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

#include <time.h>
#include <sys/time.h>

int measure_loop(int rows, int cols, int samples, int iterations, int nature_single_i, int nature_pair_i){
	printf("%d %d\n", nature_single_i, nature_pair_i);
	ising_t input;
    construct_ising(&input, rows, cols);
    ising_t output;
	float low_s, high_s, low_p, high_p;
	const char* nature_single = nature_single_map(&low_s, &high_s, nature_single_i);
	const char* nature_pair = nature_pair_map(&low_p, &high_p, nature_pair_i);
	
	int gpu = 1;
	/*------------------------------------------------------------------------------------------*/
	cl_device_id device_id;
	
	int err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
	if (err != CL_SUCCESS){
		printf("Error: Failed to create a device group!\n");
		return EXIT_FAILURE;
	}
	
	cl_context context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (!context){
		printf("Error: Failed to create a compute context!\n");
		return EXIT_FAILURE;
	}
	
	char* KernelSource=read_kernel("kernel.cl");
	
	
	cl_command_queue commands = clCreateCommandQueue(context, device_id, 0, &err);
	if (!commands){
		printf("Error: Failed to create a command commands!\n");
		printf("%i %i\n", CL_INVALID_VALUE, err);
		return EXIT_FAILURE;
	}
	
	
	cl_program program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
	if (!program){
		printf("Error: Failed to create compute program!\n");
		return EXIT_FAILURE;
	}
	
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS){
		size_t len;
		char buffer[2048];
		
		printf("Error: Failed to build program executable!\n");
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		return EXIT_FAILURE;
	}
	
	cl_kernel kernelInf = clCreateKernel(program, "updateFactor", &err);
	if (!kernelInf || err != CL_SUCCESS){
		printf("Error: Failed to create compute kernel!\n");
		return 1;
	}
	cl_kernel kernelMarg = clCreateKernel(program, "updateMarginals", &err);
	if (!kernelMarg || err != CL_SUCCESS){
		printf("Error: Failed to create compute kernel marg!\n");
		return 1;
	}
	/*---------------------------------------------------------------------------------------*/
	
	
	for(int k=0; k<samples; k++){
		unsigned seed = k;
		
		random_fill_ising(&input, low_s, high_s, low_p, high_p, &seed);
		
/*		printf("exact sequential log domain %d\n", k);
		ising_t exact;
		exact_marginals_log_domain(&exact, input);
		write_marginals_to_file(rows, cols, "exact", nature_single, nature_pair, k, 0, exact);
		destroy_ising(&exact);*/
		
//		printf("EP parallel %d\n", k);
//		do_inference_measure(kernelInf, kernelMarg, commands, context, device_id, &output, input, 1.5, iterations, nature_single, nature_pair, "parallelGPUPseudoConvex", k);
//		do_inference_measure(kernelInf, kernelMarg, commands, context, device_id, &output, input, 1, iterations, nature_single, nature_pair, "parallelGPU", k);
//		do_inference_measure(kernelInf, kernelMarg, commands, context, device_id, &output, input, 2, iterations, nature_single, nature_pair, "parallelGPUConvex", k);
//		printf("EP Sequential %d\n", k);
		sequential_inference_measure(&output, input, 1, iterations, nature_single, nature_pair, "sequential", k);
		
	}
	
	clReleaseProgram(program);
    clReleaseKernel(kernelInf);
	clReleaseKernel(kernelMarg);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
	
	return 0;
}

const char* nature_pair_map(float* low, float* high, int i){
	switch (i){
		case -2: *low=-3; *high=0; return "stronglyRepulsive";
		case -1: *low=-1; high=0; return "repulsive";
		case 0: *low=-1; *high=1; return "mixed";
		case 1: *low=-3; *high=3; return "stronglyMixed";
		case 2: *low=0; *high=1; return "attractive";
		case 3: *low=0; *high=3; return "stronglyAttractive";
	}
	printf("no corresponding mapping\n");
	exit(1);
}

const char* nature_single_map(float* low, float* high, int i){
	switch (i){
		case -1: *low=-1; *high=0; return "negative";
		case 0: *low=0; *high=0; return "zero";
		case 1: *low=-1; *high=1; return "mixed";
		case 2: *low=0; *high=1; return "positive";
	}
	printf("no corresponding mapping\n");
	exit(1);
}

int measure_time(cl_kernel kernelInf, cl_kernel kernelMarg, cl_command_queue commands, cl_context context, cl_device_id device_id, int rows, int cols, int iterations){
	ising_t input;
    construct_ising(&input, rows, cols);
    ising_t output;
	
	unsigned seed = 3;
	random_fill_ising(&input, -1, 1,-1, 1, &seed);
	
	//sequential_inference_measure(&output, input, 1, iterations, "", "", "sequential", 0);
	
	do_inference_measure(kernelInf, kernelMarg, commands, context, device_id, &output, input, 1, iterations, "", "", "", 0);
	
/*	struct timespec ts;
	clock_serv_t cclock;
	mach_timespec_t mts;
	host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
	clock_get_time(cclock, &mts);
	ts.tv_sec = mts.tv_sec;
	ts.tv_nsec = mts.tv_nsec;
	
	sequential_inference_measure(&output, input, 1, iterations, "", "", 0);
	
	clock_get_time(cclock, &mts);
	mach_port_deallocate(mach_task_self(), cclock);
	struct timespec ts_end;
	ts_end.tv_sec = mts.tv_sec;
	ts_end.tv_nsec = mts.tv_nsec;
	
	printf("%f\n",(ts_end.tv_sec * 1E9 + ts_end.tv_nsec - ts.tv_sec * 1E9 - ts.tv_nsec)/1E9);*/
	
	return 0;
}