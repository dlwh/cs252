#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <OpenCL/opencl.h>

int main(int argc, char** argv)
{
	// declare memory to hold the device id
	cl_device_id* device_id=(cl_device_id*) malloc(10*sizeof(cl_device_id));
	cl_int err;
	cl_platform_id* platforms = (cl_platform_id*) malloc(10*sizeof(cl_platform_id));
	cl_uint n_platforms;

	err = clGetPlatformIDs(10, platforms, &n_platforms);

	printf("Found %u platform\n", n_platforms);

	cl_uint device_num;
	err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, 10, device_id, &device_num);
	                                           
	if (err != CL_SUCCESS){
        	printf("Error: Failed to retrieve list of OpenCL devices.\n");
        	return EXIT_FAILURE;
	}

	printf("Found %d OpenCL devices on first platform\n", device_num);

	char* name = (char*) malloc(100*sizeof(char));
	cl_uint max_cu, max_freq, max_wid;
	size_t max_wgs;
	for(int i=0; i<device_num; i++){
		err = clGetDeviceInfo(device_id[i], CL_DEVICE_NAME, 100, name, NULL);
		err = clGetDeviceInfo(device_id[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &max_cu, NULL);
		err = clGetDeviceInfo(device_id[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &max_freq, NULL);
		err = clGetDeviceInfo(device_id[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &max_wid, NULL);
		err = clGetDeviceInfo(device_id[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_wgs, NULL);
		printf("%d %s %uMHz %u %u %lu \n", i, name, max_freq, max_cu, max_wid, max_wgs);
	}

	return 0;
}

