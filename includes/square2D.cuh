#ifndef SQUARE2D_H
#define SQUARE2D_H

#include "curand.h"
#include "curand_kernel.h"
#include "includes/cuda_math.h"

#define RANDOM_SEEDS 1000
#define EPSILONF 0.000001
#define E_PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062

class square2D
{
    // number of squares
    int w;
    int h;

    // triangles of the mesh being square grid
    CompFab::Triangle* triangles;
    int numTriangles;
    bool* R;

    // information about how large the samples are and where they begin
    float spacing;
    float2 bottom_left;

    // sampling information for multiple intersection rays
    bool double_thick;
    
    public:
    int samples;

    square2D(bool* R_,int w_, int h_, CompFab::Triangle* triangle_, int numTriangle, float sp, bool double_thick_, float2 bottom_left_, int samples_);

    __device__ bool edgeFunction(float3 V1, float3 V2, float2 P);

    __device__ void voxelize_kernel(int2 Index);
    
};

#endif