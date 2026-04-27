#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "matmul_metal.h"

void metal_matmul(const float* A, const float* B, float* C, uint32_t M, uint32_t N, uint32_t K)
{
    // get the GPU
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

	// load the kernel
    id<MTLLibrary> library = [device newDefaultLibrary];
    id<MTLFunction> function = [library newFunctionWithName:@"matmul"];
    
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];

	// create buffers (no copying on Apple Silicon — shared memory)
    NSUInteger bytesA = M * K * sizeof(float);
    NSUInteger bytesB = K * N * sizeof(float);
    NSUInteger bytesC = M * N * sizeof(float);

    id<MTLBuffer> bufA = [device newBufferWithBytes:A length:bytesA options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [device newBufferWithBytes:B length:bytesB options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufC = [device newBufferWithLength:bytesC options:MTLResourceStorageModeShared];

	// create command buffer and encoder
    id<MTLCommandQueue>   queue   = [device newCommandQueue];
    id<MTLCommandBuffer>  cmdBuf  = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

	// set pipeline and bind buffers
    [encoder setComputePipelineState:pipeline];

    [encoder setBuffer:bufA offset:0 atIndex:0];
    [encoder setBuffer:bufB offset:0 atIndex:1];
    [encoder setBuffer:bufC offset:0 atIndex:2];

    [encoder setBytes:&M length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&N length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&K length:sizeof(uint32_t) atIndex:5];

	// dispatch threads
    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
    MTLSize gridSize = MTLSizeMake(
        (N + 15) / 16,  // ceil(N/16) threadgroups in x
        (M + 15) / 16,  // ceil(M/16) threadgroups in y
        1
    );

    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

	// commit and wait
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    // read result back (no copy needed — shared memory)
    float* result = (float*)[bufC contents];
    memcpy(C, result, bytesC);
}

