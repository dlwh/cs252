/*
 *  utils.c
 *  gpuep
 *
 *  Created by Alex K on 5/3/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "utils.h"

char* read_kernel(const char* filename){
	FILE *fl=fopen(filename, "rt");
	if (fl==NULL){
		printf("Missing kernel file %s\n", filename);
		exit(1);
	}
	fseek(fl, 0, SEEK_END);
	int len= ftell(fl);
	rewind(fl);
	char *KernelSource = (char*) malloc(sizeof(char)*len + 1);
	KernelSource[len]='\0';
	fread(KernelSource, sizeof(char), len, fl);
	fclose(fl);
	
	return KernelSource;
}

void write_marginals_to_file(const char* prefix, int num, ising_t ising){
	char* filename = NULL;
	asprintf(&filename, "%s%d.txt", prefix, num);
	FILE* fl=fopen(filename, "wt");
	if (fl==NULL){
		printf("Can not open %s\n in W mode", filename);
		exit(1);
	}
	for(int r = 0; r < ising.rows; ++r) {
        for(int c = 0; c < ising.cols; ++c) {
            fprintf(fl, "%.8f\n", get_ising_singleton(&ising, r, c));
        }
    }
}