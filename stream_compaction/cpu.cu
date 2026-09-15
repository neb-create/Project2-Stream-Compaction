#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            int sum = 0;
            for (int i = 0; i < n; i++) {
                odata[i] = sum;
                sum += idata[i];
			}

            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            int i = 0;
            for (int j = 0; j < n; j++) {
                if (idata[j] != 0) {
                    odata[i] = idata[j];
                    i++;
				}
            }

            timer().endCpuTimer();
            return i;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {

            int* mapped = new int[n];
            int* scanned = new int[n];

            timer().startCpuTimer();

            // map
            for (int i = 0; i < n; i++) {
				mapped[i] = idata[i] != 0 ? 1 : 0;
            }

            // scan
            int sum = 0;
            for (int i = 0; i < n; i++) {
                scanned[i] = sum;
                sum += mapped[i];
            }

			// scatter
            int count = 0;
            for (int i = 0; i < n; i++) {
                if (mapped[i] == 1) {
                    count++;
                    odata[scanned[i]] = idata[i];
				}
			}

            timer().endCpuTimer();

            // free memory
            delete[] mapped;
            delete[] scanned;

            return count;
        }
    }
}
