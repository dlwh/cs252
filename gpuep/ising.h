//
//  ising.h
//  gpuep
//
//  Created by David Hall on 4/24/12.
//  Copyright (c) 2012 UC Berkeley. All rights reserved.
//

#ifndef gpuep_ising_h
#define gpuep_ising_h

#include <OpenCL/opencl.h>

typedef struct {
    int rows, cols;
    // row * numCols + col
    float* singleton;
    // (row * numCols + col) * numDirs +  dir
    float* pair;
} ising_t;


extern void construct_ising(ising_t *ising, int rows, int cols);
extern void destroy_ising(ising_t *ising);
extern void random_fill_ising(ising_t *ising, float lowerBound, float upperBound, float pairLowerBound, float pairUpperBound, unsigned* seed);
extern void ising_print(ising_t ising);
extern void ising_print_single(ising_t ising);
extern void ising_print_pair(ising_t ising);

/// Inference routines
extern int do_inference(ising_t* result, ising_t model, cl_context context, cl_device_id device_id, float power, int numIter);
extern int sequential_inference(ising_t* result, ising_t model, float power, int numIter);

static inline float get_ising_singleton(ising_t* ising, int row, int col) {
    return ising->singleton[row * ising->cols + col];
}

static inline void set_ising_singleton(ising_t* ising, int row, int col, float value) {
    ising->singleton[row * ising->cols + col] = value;
}

typedef enum {
   IM_DOWN = 0,
   IM_RIGHT,
   IM_LEFT,
   IM_UP
} im_dir_t;

static inline float get_ising_pair(ising_t* ising, int row, int col, im_dir_t dir) {
    return ising->pair[(row * ising->cols + col)*2 + dir];
}

static inline void set_ising_pair(ising_t* ising, int row, int col, im_dir_t dir, float value) {
    ising->pair[(row * ising->cols + col)*2 + dir] = value;
}

#endif
