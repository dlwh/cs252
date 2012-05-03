//
//  pairwise.c
//  gpuep
//
//  Created by David Hall on 5/3/12.
//  Copyright (c) 2012 UC Berkeley. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "math.h"
#include "pairwise.h"

float inline static log_add(float a, float b){
	if(a>b){
		return a + log1pf(expf(b-a));
	}else{
		return b + log1pf(expf(a-b));
	}
}

void construct_pairwise(pairwise_t *pairwise, int vars) {
    pairwise->numVars = vars;
    pairwise->singleton = (float*)malloc(sizeof(float) * vars);
    memset(pairwise->singleton, 0, sizeof(float) * vars);
    pairwise->pair = (float*)malloc(sizeof(float) * triangular_size(vars));
    memset(pairwise->pair, 0, sizeof(float) * triangular_size(vars));
}

void destroy_pairwise(pairwise_t *pairwise) {
    free(pairwise->singleton);
    free(pairwise->pair);
}

void random_fill_pairwise(pairwise_t *pairwise, float lowerBound, float upperBound, float pairLowerBound, float pairUpperBound, unsigned* seed) {
    for(int i = 0; i < pairwise->numVars; ++i) {
        pairwise->singleton[i] = (float)rand_r(seed)/RAND_MAX * (upperBound - lowerBound) + lowerBound;
        for(int j = i+1; j < pairwise->numVars; ++j) {
            pairwise->pair[triangular_index(i, j)] = (float)rand_r(seed)/RAND_MAX * (pairUpperBound - pairLowerBound) + pairLowerBound;
        }
    }
}

void pairwise_print(pairwise_t pairwise) {
    pairwise_print_single(pairwise);
    printf("\n");
    pairwise_print_pair(pairwise);
}

void pairwise_print_single(pairwise_t pairwise){
	for(int r = 0; r < pairwise.numVars; ++r) {
        printf("(%d) %.3f\n", r, get_pairwise_singleton(&pairwise, r));
    }
}

void pairwise_print_pair(pairwise_t pairwise){
	for(int r =0; r < pairwise.numVars; ++r) {
        for(int c = r+1; c < pairwise.numVars; ++c) {
            printf("(%d) -> (%d) %.3f\n", r, c, get_pairwise_pair(&pairwise, r, c));
        }
        printf("\n");
    }
}


int pair_sequential_inference(pairwise_t* result, pairwise_t model, float numEdges, int numIter) {
    construct_pairwise(result, model.numVars);
    float* pairwise_single = result->singleton;
    memcpy(result->singleton, model.singleton, model.numVars * sizeof(float));
    
    float* pairwise_pair = model.pair;
    float* pairwise_message = malloc(sizeof(float) * triangular_size(model.numVars) * 2);
    
    memset(pairwise_message, 0, sizeof(float) * triangular_size(model.numVars) * 2);
    
    for(int iter = 0; iter < numIter; ++iter) {
        for(int r = 0; r < model.numVars; ++r) {
            for(int c = r+1; c < model.numVars; ++c) {
                int index = triangular_index(r, c);
                float marginalWeightA = pairwise_single[r];
                float marginalWeightB = pairwise_single[c];
                float edgeWeight = pairwise_pair[index];
                float mesgToA = pairwise_message[index * 2 + 0];
                float mesgToB = pairwise_message[index * 2 + 1];
                marginalWeightA -= mesgToA * numEdges;
                marginalWeightB -= mesgToB * numEdges;
                
                float jointMarginal11 = marginalWeightA + marginalWeightB + edgeWeight * numEdges;
                float jointMarginal10 = marginalWeightA;
                float jointMarginal01 = marginalWeightB;
                
                
                float jointMarginalA1 = log_add(jointMarginal11,jointMarginal10);
                float jointMarginalB1 = log_add(jointMarginal11,jointMarginal01);
                float sum = log_add(log_add(jointMarginalA1, jointMarginal01),0.0);
                
                float newMargA = expf(jointMarginalA1 - sum);
                float newMargB = expf(jointMarginalB1 - sum);
                
                // logit function
                float newTargetA = logf(newMargA/(1-newMargA));
                float newTargetB = logf(newMargB/(1-newMargB));
                float adjA = newTargetA - marginalWeightA;
                float adjB = newTargetB - marginalWeightB;
                
                pairwise_message[index * 2 + 0] = adjA;
                pairwise_message[index * 2 + 1] = adjB;
                pairwise_single[r] = newTargetA;
                pairwise_single[c] = newTargetB;
            }
        }
    }
    free(pairwise_message);
    return 0;
}

void pairwise_exact_marginals_log_domain(pairwise_t* result, pairwise_t model){
	construct_pairwise(result, model.numVars);
	assert(model.numVars < 32);
	unsigned N=pow(2, model.numVars);
	float* marginals0 = (float*) calloc(model.numVars, sizeof(float));
	float* marginals1 = (float*) calloc(model.numVars, sizeof(float));
	for(int i=0; i< model.numVars; i++){
		marginals0[i] = -1.0/0.0;
		marginals1[i] = -1.0/0.0;
	}
    
	for(int i = 0; i < N; i++){
		float sum = 0;
        for(int r = 0; r < model.numVars; r++){
            if((i >> r) & 1){
                sum += model.singleton[r];
                for(int c = r+1; c < model.numVars; c++) {
                    if((i >> c) & 1) {
                        sum+=model.pair[triangular_index(r, c)];
                    }
                }
            }
        }
        
		for(int c = 0; c < model.numVars; c++){
            if ((i >> c) & 1) {
                marginals0[c] = log_add(sum, marginals0[c]);
            } else {
                marginals1[c] = log_add(sum, marginals1[c]);
            }
        }
    }
	
	for(int i = 0; i < model.numVars; ++i) {
		result->singleton[i] = marginals0[i] - marginals1[i];
	}
	
	free(marginals0);
	free(marginals1);
}
