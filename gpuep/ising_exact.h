/*
 *  ising_exact.h
 *  gpuep
 *
 *  Created by Alex K on 4/30/12.
 *  Copyright 2012 UC Berkeley. All rights reserved.
 *
 */

#include "ising.h"

#ifndef gpuep_ising_exact_h
#define gpuep_ising_exact_h

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

float inline log_add(float a, float b){
	if(a>b){
		return a + log1p(exp(b-a));
	}else{
		return b + log1p(exp(a-b));
	}
}

extern void exact_marginals(ising_t* result, ising_t model);
extern void exact_marginals_log_domain(ising_t* result, ising_t model);
extern int exact_marginals_parallel(ising_t* result, ising_t model, cl_context context, cl_device_id device_id);

#endif