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

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
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
			cudaMalloc((void**)&dev_indices, n * sizeof(int));
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // Algorithm
            timer().startGpuTimer();

            // Setup Blocksize
			int blockSize = 128;
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			// Map (outputs to dev_bools)
			StreamCompaction::Common::kernMapToBoolean <<<fullBlocksPerGrid, blockSize >>> (n, dev_bools, dev_idata);

			// Scan (outputs to dev_indices)
            // PLACEHOLDER: CPU IMPLEMENTATION
			int* mapped = new int[n];
			int* scanned = new int[n];
			cudaMemcpy(mapped, dev_bools, n * sizeof(int), cudaMemcpyDeviceToHost);
            int sum = 0;
            for (int i = 0; i < n; i++) {
                scanned[i] = sum;
                sum += mapped[i];
            }
			cudaMemcpy(dev_indices, scanned, n * sizeof(int), cudaMemcpyHostToDevice);\
			delete[] mapped;
			delete[] scanned;

            // scatter
			StreamCompaction::Common::kernScatter <<<fullBlocksPerGrid, blockSize>>> (n, dev_odata, dev_idata, dev_bools, dev_indices);

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
