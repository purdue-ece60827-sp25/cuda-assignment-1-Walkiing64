/**
 * @file lab1.cuh
 * @author Clayton Walker (cpwalker@purdue.edu)
 * @brief 
 * @version 0.1
 * @date 2025-01-28
 * 
 * 
 */

#pragma once

#ifndef LAB1_CUH
#define LAB1_CUH

	#define DEBUG_PRINT_DISABLE
		
	#define VECTOR_SIZE (5050000) //1 << 15

	#define MC_SAMPLE_SIZE		1e7 //1e6
	#define MC_ITER_COUNT		32

	#define WARP_SIZE			32
	#define SAMPLE_SIZE			MC_SAMPLE_SIZE
	#define GENERATE_BLOCKS		50000 //1024
	#define REDUCE_SIZE			50000 //32
	#define REDUCE_BLOCKS		(GENERATE_BLOCKS / REDUCE_SIZE)

#endif