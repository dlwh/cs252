#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "utils.h"
#include "ising.h"
#include "ising_exact.h"

#include "pairwise.h"
#include "measures.h"

int main (int argc, const char * argv[]) {
/*	for(int single=-1; single<3; single++){
		for(int pair=-2; pair<4; pair++){
			measure_loop(4, 4, 100, 300, single, pair);
		}
	}
*/	
/*	for(int k=3; k<500; k++){
		printf("%d\n", k);
		measure_loop(k, k, 10, 1000, 2, 3);
		measure_loop(k, k, 10, 1000, 2, -2);
		measure_loop(k, k, 10, 1000, -1, 1);
	}
*/	
//	measure_loop(630, 630, 10, 300, 2, -2);
	
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
	
	
	for(int i=3; i<500; i++){
		//int s = (int) 100*pow(10, i/25.);
		//int s=i;
		
		//printf("%d %d\n", i, s);
		//measure_loop(s, s, 10, 100, 0, 0);
		//measure_loop(s, s, 10, 100, 2, -2);
		
		printf("[%d, ", i);
		measure_time(kernelInf, kernelMarg, commands, context, device_id, i, i, 2+ceil(900/(i*i)));
	}
}

void instable(){
	ising_t input;
    construct_ising(&input, 2, 2);
    input.pair[0]=-5;
	input.pair[1]=-5;
	input.pair[2]=5;
	input.pair[5]=5;
}
