//
//  pairwise.h
//  gpuep
//
//  Created by David Hall on 5/3/12.
//  Copyright (c) 2012 UC Berkeley. All rights reserved.
//

#ifndef gpuep_pairwise_h
#define gpuep_pairwise_h

#include <OpenCL/opencl.h>

typedef struct pairwise_t {
    unsigned numVars;
    float* singleton;
    float* pair;
} pairwise_t;

extern void construct_pairwise(pairwise_t *pairwise, int numVars);
extern void destroy_pairwise(pairwise_t *pairwise);

extern void random_fill_pairwise(pairwise_t *pairwise, float lowerBound, float upperBound, float pairLowerBound, float pairUpperBound, unsigned* seed);
extern void pairwise_print(pairwise_t pairwise);
extern void pairwise_print_single(pairwise_t pairwise);
extern void pairwise_print_pair(pairwise_t pairwise);

/// Inference routines
extern int pair_do_inference(pairwise_t* result, pairwise_t model, cl_context context, cl_device_id device_id, float power, int numIter);
extern int pair_sequential_inference(pairwise_t* result, pairwise_t model, float power, int numIter);

extern void pairwise_exact_marginals_log_domain(pairwise_t* result, pairwise_t model);

inline static unsigned triangular_index(int r, int c) {
    return (c * (c+1) /2 + r);  
}

inline static unsigned triangular_size(int dim) {
    return dim * (dim+1) / 2;
}


static inline float get_pairwise_singleton(pairwise_t* pairwise, int var) {
    return pairwise->singleton[var];
}

static inline void set_pairwise_singleton(pairwise_t* pairwise, int var, float value) {
    pairwise->singleton[var] = value;
}

static inline float get_pairwise_pair(pairwise_t* pairwise, int v1, int v2) {
    return pairwise->pair[triangular_index(v1, v2)];
}

static inline void set_pairwise_pair(pairwise_t* pairwise, int v1, int v2, float value) {
    pairwise->pair[triangular_index(v1, v2)] = value;
}


#endif
