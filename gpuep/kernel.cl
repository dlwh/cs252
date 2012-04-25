//#import <math.h>
typedef enum {
    IM_DOWN = 0,
    IM_RIGHT,
    IM_LEFT,
    IM_UP
} im_dir_t;

__kernel void updateFactor(__global float* ising_pair,
                           __global float* ising_single,
                           __global float* ising_message, 
                           __global float* ising_message_out, 
                           int rows, int cols) {
	int r = get_global_id(0);                                          
	int c = get_global_id(1);                                         
	im_dir_t dir = get_global_id(2);                                          
    int nr = dir == IM_DOWN ? r + 1 : r;
    int nc = dir == IM_DOWN ? c : c + 1;
    int otherDir = dir == IM_DOWN ? IM_UP : IM_LEFT;
    
	if(nr < rows && nc < cols) {
        float marginalWeightA = ising_single[r * cols + c];
        float marginalWeightB = ising_single[nr * cols + nc];
        float edgeWeight = ising_pair[(r * cols + c) * 2 + dir];
        float mesgToA = ising_message[(r * cols + c) * 4 + dir];
        float mesgToB = ising_message[(nr * cols + nc) * 4 + otherDir];
        marginalWeightA -= mesgToA;
        marginalWeightB -= mesgToB;
        
        float jointMarginal11 = exp(marginalWeightA + marginalWeightB + edgeWeight);
        float jointMarginal10 = exp(marginalWeightA);
        float jointMarginal01 = exp(marginalWeightB);
        float sum = jointMarginal11 + jointMarginal10 + jointMarginal01 + 1;
        
        float newMargA = (jointMarginal11 + jointMarginal10)/sum;
        float newMargB = (jointMarginal11 + jointMarginal01)/sum;
        
        // logit function
        float newTargetA = log(newMargA/(1-newMargA));
        float newTargetB = log(newMargB/(1-newMargB));
        float adjA = newTargetA - marginalWeightA;
        float adjB = newTargetB - marginalWeightB;
        
        ising_message_out[(r * cols + c) * 4 + dir] = adjA;
        ising_message_out[(nr * cols + nc) * 4 + otherDir] = adjB;
	}
}

__kernel void updateMarginals(__global float* ising_marginal_in,
                              __global float* ising_message, 
                              __global float* ising_marginal_out, 
                              int rows, int cols) {
	int r = get_global_id(0);                                          
	int c = get_global_id(1);                                         
    int offset = (r * cols + c) * 4;
    float result = ising_marginal_in[offset/4];
    for(int i = 0; i < 4; ++i) {
        result += ising_message[offset + i];
    }
    ising_marginal_out[offset/4] = result;
}

