/*
 *  utils.h
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

#include "ising.h"

char* read_kernel(const char* filename);
void write_marginals_to_file(const char* filename, int suffix, ising_t ising);