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


vox3D::vox3D(bool* R_, int w_, int h_, int d_, CompFab::Triangle* triang_, int numTriangles_, float sp, bool double_thick_, float3 bottom_left_, int samples_)
{
	w = w_;
	h = h_;
	d = d_;

	triangles = triang_;
	numTriangles = numTriangles_;
	spacing = sp;

	bottom_left = bottom_left_;
	double_thick = double_thick_;

	R = R_;

	samples = samples_;

}

// adapted from: https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
__device__ bool vox3D::intersects(CompFab::Triangle &triangle, float3 dir, float3 pos) {
	float3 V1 = {triangle.m_v1.m_x, triangle.m_v1.m_y, triangle.m_v1.m_z};
	float3 V2 = {triangle.m_v2.m_x, triangle.m_v2.m_y, triangle.m_v2.m_z};
	float3 V3 = {triangle.m_v3.m_x, triangle.m_v3.m_y, triangle.m_v3.m_z};

	//Find vectors for two edges sharing V1
	float3 e1 = V2 - V1;
	float3 e2 = V3 - V1;
	
	// //Begin calculating determinant - also used to calculate u parameter
	float3 P = cross(dir, e2);

	//if determinant is near zero, ray lies in plane of triangle
	float det = dot(e1, P);
	
	//NOT CULLING
	if(det > -EPSILONF && det < EPSILONF) return false;
	float inv_det = 1.f / det;

	// calculate distance from V1 to ray origin
	float3 T = pos - V1;
	//Calculate u parameter and test bound
	float u = dot(T, P) * inv_det;
	//The intersection lies outside of the triangle
	if(u < 0.f || u > 1.f) return false;

	//Prepare to test v parameter
	float3 Q = cross(T, e1);
	//Calculate V parameter and test bound
	float v = dot(dir, Q) * inv_det;
	//The intersection lies outside of the triangle
	if(v < 0.f || u + v  > 1.f) return false;

	float t = dot(e2, Q) * inv_det;

	if(t > EPSILONF) { // ray intersection
		return true;
	}

	// No hit, no win
	return false;
}

// Decides whether or not each voxel is within the given mesh
__device__ void vox3D::voxelize_kernel(int3 Index)
{
	unsigned int xIndex = Index.x;
    unsigned int yIndex = Index.y;
    unsigned int zIndex = Index.z;

	// pick an arbitrary sampling direction
	float3 dir = make_float3(1.0, 0.0, 0.0);

	if ( (xIndex < w) && (yIndex < h) && (zIndex < d) )
	{
		// find linearlized index in final boolean array
		unsigned int index_out = zIndex*(w*h)+yIndex*h + xIndex;
		
		// find world space position of the voxel
		float3 pos = make_float3(bottom_left.x + spacing*xIndex,bottom_left.y + spacing*yIndex,bottom_left.z + spacing*zIndex);

		// check if the voxel is inside of the mesh. 
		// if it is inside, then there should be an odd number of 
		// intersections with the surrounding mesh
		unsigned int intersections = 0;
		for (int i = 0; i < numTriangles; ++i)
			if (intersects(triangles[i], dir, pos))
				intersections += 1;

		// store answer
		R[index_out] = inside(intersections, double_thick);
	}
}


// Decides whether or not each voxel is within the given partially un-closed mesh
// checks a variety of directions and picks most common belief
__device__ void vox3D::voxelize_kernel_open_mesh(curandState* globalState,int3 Index)
{
	unsigned int xIndex = Index.x;
    unsigned int yIndex = Index.y;
    unsigned int zIndex = Index.z;

	if ( (xIndex < w) && (yIndex < h) && (zIndex < d) )
	{
		// find linearlized index in final boolean array
		unsigned int index_out = zIndex*(w*h)+yIndex*h + xIndex;
		// find world space position of the voxel
		float3 pos = make_float3(bottom_left.x + spacing*xIndex,bottom_left.y + spacing*yIndex,bottom_left.z + spacing*zIndex);
		float3 dir;

		// we will randomly sample 3D space by sending rays in randomized directions
		int votes = 0;
		float theta;
		float z;

		for (int j = 0; j < samples; ++j)
		{
			// compute the random direction. Convert from polar to euclidean to get an even distribution
			theta = generate(globalState, index_out % RANDOM_SEEDS) * 2.f * E_PI;
			z = generate(globalState, index_out % RANDOM_SEEDS) * 2.f - 1.f;

			dir.x = sqrt(1-z*z) * cosf(theta);
			dir.y = sqrt(1-z*z) * sinf(theta);
			dir.z = sqrt(1-z*z) * cosf(theta);

			// check if the voxel is inside of the mesh. 
			// if it is inside, then there should be an odd number of 
			// intersections with the surrounding mesh
			unsigned int intersections = 0;
			for (int i = 0; i < numTriangles; ++i)
				if (intersects(triangles[i], dir, pos)) 
					intersections += 1;
			if (inside(intersections, double_thick)) votes += 1;
		}
		// choose the most popular answer from all of the randomized samples
		R[index_out] = votes > (samples / 2.f);
	}
}


// generates a random float between 0 and 1
__device__ float vox3D::generate( curandState* globalState , int ind) 
{
    curandState localState = globalState[ind];
    float RANDOM = curand_uniform( &localState );
    globalState[ind] = localState; 
    return RANDOM;
}

__device__ bool vox3D::inside(unsigned int numIntersections, bool double_thick) {
	// if (double_thick && numIntersections % 2 == 0) return (numIntersections / 2) % 2 == 1;
	if (double_thick) return (numIntersections / 2) % 2 == 1;
	return numIntersections % 2 == 1;
}
