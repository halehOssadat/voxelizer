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

#include "includes/square2D.cuh"


square2D::square2D(bool* R_, int w_, int h_, CompFab::Triangle* triang_, int numTriangles_, float sp, bool double_thick_, float2 bottom_left_, int samples_)
{
	w = w_;
	h = h_;

	triangles = triang_;
	numTriangles = numTriangles_;
	spacing = sp;

	bottom_left = bottom_left_;
	double_thick = double_thick_;

	R = R_;

	samples = samples_;

}

__device__ bool square2D::edgeFunction(float3 V1, float3 V2, float2 P)
{
	return ((P.x - V1.x) * (V2.y - V1.y) - (P.y - V1.y) * (V2.x - V1.x) <= 0); 
}

// Decides whether or not each pixel is within the given mesh
__device__ void square2D::voxelize_kernel(int2 Index)
{
	unsigned int xIndex = Index.x;
    unsigned int yIndex = Index.y;

	if ( (xIndex < w) && (yIndex < h) )
	{
		// find linearlized index in final boolean array
		unsigned int index_out = yIndex*h + xIndex;
		
		// find raster space position of the pixel (center of pixel)
		// center of pixel
		float2 pixelSample = make_float2(bottom_left.x + spacing*xIndex + spacing/2,bottom_left.y + spacing*yIndex + spacing/2);
		
		for (int i = 0; i < numTriangles; ++i)
		{
			float3 V0 = {triangles[i].m_v1.m_x, triangles[i].m_v1.m_y, triangles[i].m_v1.m_z};
			float3 V1 = {triangles[i].m_v2.m_x, triangles[i].m_v2.m_y, triangles[i].m_v2.m_z};
			float3 V2 = {triangles[i].m_v3.m_x, triangles[i].m_v3.m_y, triangles[i].m_v3.m_z};

			bool inside = true; 
			inside &= edgeFunction(V0, V1, pixelSample); 
			inside &= edgeFunction(V1, V2, pixelSample); 
			inside &= edgeFunction(V2, V0, pixelSample); 
 
			R[index_out] = inside;
		}
		
	}
}

