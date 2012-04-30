#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void one_configuration(__global float* ising_pair,
								__global float* ising_single,
								__global float* marginals0,
								__global float* marginals1,
								int rows, int cols){
	int i = get_global_id(0);
	
	float sum = 0;
	for(int c = 0; c < cols; c++){
		for(int r = 0; r < rows; r++){
			if((i >> r * cols + c) & 1){
				sum += ising_single[r * cols + c];
				if(r + 1 < rows && (i >> ((r + 1) * cols + c) & 1)){
					sum += ising_pair[(r * cols + c) * 2];
				}
				if(c + 1 < cols && (i >> (r * cols + c + 1) & 1)){
					sum+=ising_pair[(r * cols + c) * 2 + 1];
				}
			}
		}
	}
	float pot = exp(sum);
	
	for(int c = 0; c < cols; c++){
		for(int r = 0; r < rows; r++){
			if((i >> r * cols + c) & 1){
				marginals0[r * cols + c] += pot;
			}else{
				marginals1[r * cols + c] += pot;
			}
		}
	}
}