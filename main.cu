#include "includes/CompFab.h"
#include "math.h"
#include "curand.h"
#include "curand_kernel.h"
#include "includes/cuda_math.h"

#include <iostream>
#include <string>
#include <sstream>
#include "stdio.h"
#include <vector>

#include "includes/vox3D.cuh"



// check cuda calls for errors
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// set up random seed buffer
__global__ void setup_kernel ( curandState * state, unsigned long seed )
{
    int id = threadIdx.x;
    curand_init ( seed, id, 0, &state[id] );
} 

__global__ void voxelize_kernel_3D(vox3D voxelizer, curandState* globalState)
{
	// find the position of the voxel
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int zIndex = blockDim.z * blockIdx.z + threadIdx.z;

	int3 Index;
	Index.x = xIndex;
	Index.y = yIndex;
	Index.z = zIndex;

	if (voxelizer.samples > 0) {
		voxelizer.voxelize_kernel_open_mesh(globalState, Index);
	} else {
		voxelizer.voxelize_kernel(Index);
	}


}

// voxelize the given mesh with the given resolution and dimensions
void kernel_wrapper_3D(int samples, int w, int h, int d, CompFab::VoxelGrid *g_voxelGrid, std::vector<CompFab::Triangle> triangles, bool double_thick)
{
	
	int blocksInX = (w+8-1)/8;
	int blocksInY = (h+8-1)/8;
	int blocksInZ = (d+8-1)/8;

	dim3 Dg(blocksInX, blocksInY, blocksInZ);
	dim3 Db(8, 8, 8);

	curandState* devStates;
	if (samples > 0) {
		// set up random numbers
		dim3 tpb(RANDOM_SEEDS,1,1);
	    cudaMalloc ( &devStates, RANDOM_SEEDS*sizeof( curandState ) );
	    // setup seeds
	    setup_kernel <<< 1, tpb >>> ( devStates, time(NULL) );
	}
	
	// set up boolean array on the GPU
	bool *gpu_inside_array;
	gpuErrchk( cudaMalloc( (void **)&gpu_inside_array, sizeof(bool) * w * h * d ) );
	gpuErrchk( cudaMemcpy( gpu_inside_array, g_voxelGrid->m_insideArray, sizeof(bool) * w * h * d, cudaMemcpyHostToDevice ) );

	// set up triangle array on the GPU
	CompFab::Triangle* triangle_array = &triangles[0];
	CompFab::Triangle* gpu_triangle_array;
	gpuErrchk( cudaMalloc( (void **)&gpu_triangle_array, sizeof(CompFab::Triangle) * triangles.size() ) );
	gpuErrchk( cudaMemcpy( gpu_triangle_array, triangle_array, sizeof(CompFab::Triangle) * triangles.size(), cudaMemcpyHostToDevice ) );

	float3 lower_left = make_float3(g_voxelGrid->m_lowerLeft.m_x, g_voxelGrid->m_lowerLeft.m_y, g_voxelGrid->m_lowerLeft.m_z);
	
	vox3D voxelizer(gpu_inside_array, w, h, d, gpu_triangle_array, triangles.size(), (float) g_voxelGrid->m_spacing, double_thick, lower_left, samples);
	
	voxelize_kernel_3D <<< Dg, Db>>> (voxelizer, devStates);

	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	gpuErrchk( cudaMemcpy( g_voxelGrid->m_insideArray, gpu_inside_array, sizeof(bool) * w * h * d, cudaMemcpyDeviceToHost ) );

	gpuErrchk( cudaFree(gpu_inside_array) );
	gpuErrchk( cudaFree(gpu_triangle_array) );
	
}