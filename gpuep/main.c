#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "ising.h"
#include "ising_exact.h"

void test(int a, int b);

int main (int argc, const char * argv[]) {
	ising_t input;
    construct_ising(&input, 5, 5);
    ising_t output;
    
    unsigned seed = 3;
    
    random_fill_ising(&input, -1, 1, &seed);
    
    int gpu = 1;
    
    cl_device_id device_id;             // compute device id 

    int err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }
    
    cl_context context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }
	
    
    ising_print(input);
    do_inference(&output, input, context, device_id, 400);
    ising_print(output);
    destroy_ising(&output);
    sequential_inference(&output, input, 400);
    ising_print(output);
	
	ising_t exact;
//	exact_marginals_parallel(&exact, input, context, device_id);
	exact_marginals(&exact, input);
	
	ising_print(exact);
	
    destroy_ising(&input);
    destroy_ising(&output);
	destroy_ising(&exact);
}
