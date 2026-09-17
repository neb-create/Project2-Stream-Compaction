#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernNaiveScan(int n, int d, const int *idata, int *odata) {

            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) return;

            if (index < (1 << d)) {
                odata[index] = idata[index];
            }
            else {
                odata[index] = idata[index] + idata[index - (1 << d)];
            }
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {

            // Setup Memory to GPU
            int* dev_bufA, * dev_bufB;
            cudaMalloc((void**)&dev_bufA, n * sizeof(int));
            cudaMalloc((void**)&dev_bufB, n * sizeof(int));
            cudaMemset(dev_bufA, 0, sizeof(int));

            // copy data, offset by 1 to the right for exclusive scan
            cudaMemcpy(dev_bufA + 1, idata, (n - 1) * sizeof(int), cudaMemcpyHostToDevice);

            // Algorithm
            timer().startGpuTimer();

            // compute the iterations needed
            int iterations = ilog2ceil(n);

            // Set up block grid size
            int blockSize = BLOCKSIZE;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            for (int d = 0; d < iterations; d++) {

                kernNaiveScan << <fullBlocksPerGrid, blockSize >> > (n, d, dev_bufA, dev_bufB);
                std::swap(dev_bufA, dev_bufB);
            }

            timer().endGpuTimer();

            // Copy and Setup Memory to CPU and free GPU memory
            cudaMemcpy(odata, dev_bufA, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_bufA);
			cudaFree(dev_bufB);

        }
    }
}
