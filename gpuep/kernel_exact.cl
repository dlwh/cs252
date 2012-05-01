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
	
	for(int c = 0; c < cols; c++){
		for(int r = 0; r < rows; r++){
			if((i >> r * cols + c) & 1){
				float m = marginals0[r * cols + c];
				if(m>sum){
					marginals0[r * cols + c] = m + log1p(exp(sum-m));
				}else{
					marginals0[r * cols + c] = sum + log1p(exp(m-sum));
				}
			}else{
				float m = marginals1[r * cols + c];
				if(m>sum){
					marginals1[r * cols + c] = m + log1p(exp(sum-m));
				}else{
					marginals1[r * cols + c] = sum + log1p(exp(m-sum));
				}
			}
		}
	}
}