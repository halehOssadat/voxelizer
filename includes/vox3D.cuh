#ifndef VOX3D_H
#define VOX3D_H

#include "curand.h"
#include "curand_kernel.h"
#include "includes/cuda_math.h"

#define RANDOM_SEEDS 1000
#define EPSILONF 0.000001
#define E_PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062

class vox3D
{
    // number of voxels
    int w;
    int h;
    int d;

    // triangles of the mesh being voxelized
    CompFab::Triangle* triangles;
    int numTriangles;
    bool* R;

    // information about how large the samples are and where they begin
    float spacing;
    float3 bottom_left;

    // sampling information for multiple intersection rays
    bool double_thick;
    
    public:
    int samples;

    vox3D(bool* R_,int w_, int h_,int d_, CompFab::Triangle* triangle_, int numTriangle, float sp, bool double_thick_, float3 bottom_left_, int samples_);

    __device__ bool intersects(CompFab::Triangle &triangle, float3 dir, float3 pos);
    
    __device__ void voxelize_kernel(int3 Index);
    
    __device__ void voxelize_kernel_open_mesh(curandState* globalState,int3 Index);

    __device__ float generate(curandState* globalState , int ind);

    __device__ bool inside(unsigned int numIntersections, bool double_thick);
      
};

#endif