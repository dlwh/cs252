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

extern void exact_marginals(ising_t* result, ising_t model);
extern int exact_marginals_parallel(ising_t* result, ising_t model, cl_context context, cl_device_id device_id);

#endif