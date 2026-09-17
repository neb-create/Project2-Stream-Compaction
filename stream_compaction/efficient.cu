#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int d, int* data) {

			// thread index
            int t_index = threadIdx.x + (blockIdx.x * blockDim.x);

            // data index
            int index = (t_index + 1) * (1 << (d + 1)) - 1;
            if (index >= n) return;

            // perform operation
			int index_add = index - (1 << d);
			data[index] += data[index_add];

        }

        __global__ void setLastZero(int n, int* data) {

            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index == 0) {
                data[n - 1] = 0;
            }

		}

        __global__ void kernDownSweep(int n, int d, int* data) {

            // thread index
            int t_index = threadIdx.x + (blockIdx.x * blockDim.x);

            // data index
            int index = (t_index + 1) * (1 << (d + 1)) - 1;
            if (index >= n) return;

            // perform operation
			int index_add = index - (1 << d);

			data[index] += data[index_add];
			data[index_add] = data[index] - data[index_add];

        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            // Setup Memory for data ( in-place ) 
            int max_d = ilog2ceil(n);
            int paddedN = 1 << max_d;
            int* dev_data;
            cudaMalloc((void**)&dev_data, paddedN * sizeof(int));
            cudaMemset(dev_data, 0, paddedN * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // Algorithm
            timer().startGpuTimer();

			// Setup Blocksize
            int blockSize = BLOCKSIZE;

            // UpSweep
            for (int d = 0; d < max_d; d++) {

				int totalThreads = 1 << (max_d - d - 1);

                dim3 fullBlocksPerGrid((totalThreads + blockSize - 1) / blockSize);
				kernUpSweep <<<fullBlocksPerGrid, blockSize >>> (paddedN, d, dev_data);

			}

			// DownSweep
			setLastZero <<<1, 1 >>> (paddedN, dev_data);
            for (int d = max_d - 1; d >= 0; d--) {

				int totalThreads = 1 << (max_d - d - 1);

				dim3 fullBlocksPerGrid((totalThreads + blockSize - 1) / blockSize);
				kernDownSweep <<<fullBlocksPerGrid, blockSize >>> (paddedN, d, dev_data);

            }

            timer().endGpuTimer();

            // Cleanup
            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_data);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {

			// Setup Memoery for mapped and scanned arrays
			int* dev_idata, * dev_odata, * dev_bools, * dev_indices;

			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			cudaMalloc((void**)&dev_bools, n * sizeof(int));
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int max_d = ilog2ceil(n);
            int paddedN = 1 << max_d;
            cudaMalloc((void**)&dev_indices, paddedN * sizeof(int));
            cudaMemset(dev_indices, 0, paddedN * sizeof(int));

            // Algorithm
            timer().startGpuTimer();

            // Setup Blocksize
			int blockSize = BLOCKSIZE;
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			// Map (outputs to dev_bools)
			StreamCompaction::Common::kernMapToBoolean <<<fullBlocksPerGrid, blockSize >>> (n, dev_bools, dev_idata);

			// Scan (outputs to dev_indices)
            cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
            // Scan UpSweep
            for (int d = 0; d < max_d; d++) {
                int totalThreads = 1 << (max_d - d - 1);
                dim3 fullBlocksPerGrid((totalThreads + blockSize - 1) / blockSize);
                kernUpSweep << <fullBlocksPerGrid, blockSize >> > (paddedN, d, dev_indices);
            }
            // Scan DownSweep
            setLastZero << <1, 1 >> > (paddedN, dev_indices);
            for (int d = max_d - 1; d >= 0; d--) {
                int totalThreads = 1 << (max_d - d - 1);
                dim3 fullBlocksPerGrid((totalThreads + blockSize - 1) / blockSize);
                kernDownSweep << <fullBlocksPerGrid, blockSize >> > (paddedN, d, dev_indices);
            }

            // Scatter
			StreamCompaction::Common::kernScatter <<<fullBlocksPerGrid, blockSize>>> (n, dev_odata, dev_idata, dev_bools, dev_indices);

            // Complete Algorithm
            timer().endGpuTimer();

			// Cleanup
            int last_index, last_bool;
            cudaMemcpy(&last_index, dev_indices + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&last_bool, dev_bools + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            int num_elements = last_index + last_bool;
			cudaMemcpy(odata, dev_odata, num_elements * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(dev_idata);
			cudaFree(dev_odata);
			cudaFree(dev_bools);
			cudaFree(dev_indices);

            return num_elements;
        }
    }
}
