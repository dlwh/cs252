/*
 *  measures.h
 *  gpuep
 *
 *  Created by Alex K on 5/3/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
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

#include "utils.h"
#include "ising.h"
#include "ising_exact.h"

#include "pairwise.h"

int measure_loop(int rows, int cols, int samples, int iterations, int nature_single_i, int nature_pair_i);
const char* nature_pair_map(float* low, float* high, int i);
const char* nature_single_map(float* low, float* high, int i);
int measure_time(cl_kernel kernelInf, cl_kernel kernelMarg, cl_command_queue commands, cl_context context, cl_device_id device_id, int rows, int cols, int iterations);