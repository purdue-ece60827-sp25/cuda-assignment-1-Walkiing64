
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
	#ifdef DETECT_BLOCKSIZE
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

	//Get the index of this thread
	uint64_t i = threadIdx.x + blockDim.x * blockIdx.x;

	//This thread should only do stuff if there is a spot in the pSums for the output
	if(i < pSumSize) {
		//Initialize the hit count to be the number of samples (this is the maximum number of hits possible)
		uint64_t hitCount = sampleSize;

		//Setup the RNG
		curandState_t rng;
		curand_init(clock64(), i, 0, &rng);	

		//Run the generate code based on the sample size
		for(uint64_t j = 0; j < sampleSize; j++) {
			//Get the x and y values
			float x = curand_uniform(&rng);
			float y = curand_uniform(&rng);

			// Calculate the distance from the origin, and cast it to an int.
			// Since 0 < dist < 2^0.5, this value will be 0 (hit) or 1 (miss)
			int miss = __float2int_rd(x * x + y * y);

			// Since hitCount was initialized to the maximum number of hits, we subtract if a miss occured
			hitCount -= miss;
		}

		//The hit count should be complete now, so it can be written to the output vector
		pSums[i] = hitCount;
	}
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here

	// Get the index of the thread
	uint64_t i = threadIdx.x + blockDim.x * blockIdx.x; 

	// For some reason, the function header names "reduceSize" as the size of the totals array,
	// so we still need to calculate how much each thread must reduce:
	// If anyone is reading this, please change the variable name in the header of this function in the starter code.
	// I am unsure if I am allowed to change it, and it makes no sense for "reduceSize" to be the number of values
	// each thread reduces in one function, then for the same name to mean the size of an array in another.
	uint64_t reduce = pSumSize / reduceSize;

	// Dont operate on any values outside of totals, which has size = reduceSize (WHY)
	if(i < reduceSize) {
		// Initialize the partial total
		uint64_t pTotal = 0;

		// Loop through the number of values this thread is supposed to reduce
		for(uint64_t j = 0; j < reduce; j++) {
			// Dont try and add a value that is outside of the pSum vector
		
			// This has each thread operate on consecutive values, so consecutive threads do not see consecutive
			// values (probably bad)
			// if(i * reduce + j < pSumSize) {
			// 	pTotal += pSums[i * reduceSize + j];
			// }

			// This has each thread operate on values separated by reduceSize, so consecutive threads do see consecutive
			// values (probably better)
			if(i + j * reduceSize < pSumSize) {
				pTotal += pSums[i + reduceSize * j];
			}
		}

		//Write the partial total to the output
		totals[i] = pTotal;
	}
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

	// Use the occupancy API to determine optimal block size
	int genBlockSize;
	int redBlockSize;

	cudaError_t err;
	#ifdef DETECT_BLOCKSIZE
		int minGridSize;
		err = cudaOccupancyMaxPotentialBlockSize(&minGridSize, &genBlockSize, generatePoints);

		dbprintf("Minimum Generate Grid Size: %d\n", minGridSize);
		dbprintf("Optimal Generate Block Size: %d\n", genBlockSize);

		if(err != cudaSuccess) {
			dbprintf("Detecting the gen blocksize produced error: %s\n", cudaGetErrorString(err));
		}

		err = cudaOccupancyMaxPotentialBlockSize(&minGridSize, &redBlockSize, reduceCounts);
		
		dbprintf("Minimum Reduce Grid Size: %d\n", minGridSize);
		dbprintf("Optimal Reduce Block Size: %d\n", redBlockSize);

		if(err != cudaSuccess) {
			dbprintf("Detecting the red blocksize produced error: %s\n", cudaGetErrorString(err));
		}
	#else
		genBlockSize = 1024; //Manually set optimal blocksize to save API calls
		redBlockSize = 768; //768
	#endif

	//Allocate space for the vectors on the device
	uint64_t* pSum_d, * totals_d;

	//The size of pSum is equal to the number of generate threads
	dbprintf("Allocating memory for pSum_d and totals_d!\n");
	err = cudaMalloc((void**) &pSum_d, generateThreadCount * sizeof(*pSum_d));
	if(err != cudaSuccess) {
		std::cout << "Unable to allocate GPU vector memory! Exiting..." << std::endl;
	}
	dbprintf("Finished allocating memory for pSum_d!\n");

	// The size of totals is equal to the number of reduce threads
	// Note that if generateThreads % reduceThreads != 0, there will be values left over
	// at the end of pSums that must be handled by the CPU.

	err = cudaMalloc((void**) &totals_d, reduceThreadCount * sizeof(*totals_d));
	if(err != cudaSuccess) {
		std::cout << "Unable to allocate GPU vector memory! Exiting..." << std::endl;
	}
	dbprintf("Finished allocating memory for totals_d!\n");

	// Setup the kernel function call to generatePoints
	dim3 genDimGrid(generateThreadCount/genBlockSize, 1, 1);
	if (generateThreadCount % genBlockSize != 0) {
		// Take the ceiling of the blocks per grid
		genDimGrid.x++;
	}
	dim3 genDimBlock(genBlockSize, 1, 1);

	dbprintf("Generating partial sums!\n");

	generatePoints<<<genDimGrid, genDimBlock>>>(pSum_d, generateThreadCount, sampleSize);

	dbprintf("Finished generating sums!\n");

	// Setup the kernel launch for reduceCounts
	dim3 redDimGrid(reduceThreadCount/redBlockSize, 1, 1);
	if(reduceThreadCount % redBlockSize != 0) {
		redDimGrid.x++;
	}
	dim3 redDimBlock(redBlockSize, 1, 1);

	dbprintf("Reducing sums!\n");

	reduceCounts<<<redDimGrid, redDimBlock>>>(pSum_d, totals_d, generateThreadCount, reduceThreadCount);

	dbprintf("Finished reducing sums!\n");

	// Get totals_d off the device
	uint64_t* totals = (uint64_t*) malloc(reduceThreadCount * sizeof(*totals));
	cudaMemcpy(totals, totals_d, reduceThreadCount * sizeof(*totals_d), cudaMemcpyDeviceToHost);

	#ifndef DEBUG_PRINT_DISABLE 
		printf(" totals = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%ld, ", totals[i]);
		}
		printf(" ... }\n");
	#endif

	// Now all of the hits must be summed. If generateThreads % reduceThreadCount != 0, we must get
	// some values at the end of pSums off the device
	uint64_t total = 0;
	uint64_t leftovers = generateThreadCount % reduceThreadCount;
	dbprintf("Number of leftover values: %ld\n", leftovers);
	if(leftovers != 0) {
		//Allocate space for the leftover values
		uint64_t* left_arr = (uint64_t*) malloc(leftovers * sizeof(*left_arr));
		if(left_arr == nullptr) {
			std::cout << "Unable to allocate leftover array! Exiting..." << std::endl;
		}

		//Determine the offset from the begining of pSums
		uint64_t offset = generateThreadCount - leftovers;

		dbprintf("pSums offset: %d\n", offset);

		//Copy values from pSum_d to left_arr
		cudaMemcpy(left_arr, pSum_d + offset, leftovers * sizeof(*pSum_d), cudaMemcpyDeviceToHost);

		//Accumulate these values into totals
		for(int i = 0; i < leftovers; i++) {
			total += left_arr[i];
		}

		//Free left_arr
		free(left_arr);
	}

	// Now sum up all the values in totals
	for(int i = 0; i < reduceThreadCount; i++) {
		total += totals[i];
	}

	dbprintf("The total is: %ld\n", total);

	// The approximate pi is now just the average # of hits times 4
	approxPi = ((double) total / (generateThreadCount * sampleSize)) * 4.0f;

	//Free the host and device arrays
	free(totals);
	cudaFree(totals_d);
	cudaFree(pSum_d);

	return approxPi;
}
