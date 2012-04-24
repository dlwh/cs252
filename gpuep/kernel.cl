__kernel void square(__global float* input, 
					 __global float* output, 
					 const unsigned int count)                                           
{                                                                      
	int i = get_global_id(0);                                          
	if(i < count)                                                      
	{                                                                   
		float x=3.14159;
        for(int j=0; j<10000; j++){
		    for(int k=0; k<10000; k++){
				if(x>100000){
					x=x/((float) k+j);
				}else{
					x=x*((float) k+j);
				}
			}
		}
        output[i] = input[i] * input[i];                                
	}
}                                                                      
