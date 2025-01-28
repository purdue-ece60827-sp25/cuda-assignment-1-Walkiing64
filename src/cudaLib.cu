
#include "cudaLib.cuh"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here

	//Get the index of this thread
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	//Only do the calculation if we are in the thread bounds
	if(i < size) {
		y[i] += scale * x[i];
	}
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here

	// Use the occupancy API to determine optimal block size
	int blockSize;

	cudaError_t err;
	#ifndef DETECT_BLOCKSIZE
		int minGridSize;
		err = cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, saxpy_gpu);

		dbprintf("Minimum Grid Size: %d\n", minGridSize);
		dbprintf("Optimal Block Size: %d\n", blockSize);
	#else
		blockSize = 1024; //Manually set optimal blocksize to save API calls
	#endif

	// Generate the vectors and the scale factor
	float* a = (float*) malloc(vectorSize * sizeof(*a));
	float* b = (float*) malloc(vectorSize * sizeof(*b));
	float* c = (float*) malloc(vectorSize * sizeof(*c));

	if(a == nullptr || b == nullptr || c == nullptr) {
		std::cout << "Unable to allocate vector memory! Exiting..." << std::endl;
	}

	vectorInit(a, vectorSize);
	vectorInit(b, vectorSize);

	// Make scale a random number between 0 and 100
	float scale = (float) (rand() % 100);

	#ifndef DEBUG_PRINT_DISABLE 
		printf("\n Adding vectors : \n");
		printf(" scale = %f\n", scale);
		printf(" a = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", a[i]);
		}
		printf(" ... }\n");
		printf(" b = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", b[i]);
		}
		printf(" ... }\n");
	#endif

	// Put the vectors on the GPU (only a and b are moved to device)
	float* a_d, * b_d;
	err = cudaMalloc((void**) &a_d, vectorSize * sizeof(*a_d));
	if(err != cudaSuccess) {
		std::cout << "Unable to allocate GPU vector memory! Exiting..." << std::endl;
	}
	cudaMemcpy(a_d, a, vectorSize * sizeof(*a_d), cudaMemcpyHostToDevice);

	err = cudaMalloc((void**) &b_d, vectorSize * sizeof(*b_d));
	if(err != cudaSuccess) {
		std::cout << "Unable to allocate GPU vector memory! Exiting..." << std::endl;
	}
	cudaMemcpy(b_d, b, vectorSize * sizeof(*b_d), cudaMemcpyHostToDevice);

	// Setup the kernel launch
	dim3 dimGrid(vectorSize/blockSize, 1, 1);
	if (vectorSize % blockSize != 0) {
		// Take the ceiling of the blocks per grid
		dimGrid.x++;
	}
	dim3 dimBlock(blockSize, 1, 1);

	saxpy_gpu<<<dimGrid, dimBlock>>>(a_d, b_d, scale, vectorSize);

	//Verify the results
	cudaMemcpy(c, b_d, vectorSize * sizeof(*b_d), cudaMemcpyDeviceToHost);
	
	#ifndef DEBUG_PRINT_DISABLE 
		printf(" c = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", c[i]);
		}
		printf(" ... }\n");
	#endif

	int errorCount = verifyVector(a, b, c, scale, vectorSize);
	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";

	// Free CPU and GPU vectors
	free(a);
	free(b);
	free(c);

	cudaFree(a_d);
	cudaFree(b_d);

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;

	//      Insert code here
	std::cout << "Sneaky, you are ...\n";
	std::cout << "Compute pi, you must!\n";
	return approxPi;
}
