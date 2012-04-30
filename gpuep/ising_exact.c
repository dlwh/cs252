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

#include "ising_exact.h"

extern void exact_marginals(ising_t* result, ising_t model){
	construct_ising(result, model.rows, model.cols);
	assert(model.rows * model.cols <= 32);
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