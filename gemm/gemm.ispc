task void gemm_ispc_task(uniform int m, uniform int n, uniform int k,
	 uniform double A[], uniform double B[], uniform double C[], uniform double alpha,
	 uniform double beta, uniform int nPerBlock, uniform int iBlockStart, uniform int iBlockEnd,
	 uniform int jBlockStart, uniform int jBlockEnd) {
	// YOUR IMPLEMENTATION HERE

	// 1 block is splited into 8 parts executed by 8 threads (2 rows x 4 cols)
	uniform int nRowsPerPart = (iBlockEnd - iBlockStart)/2; // n Rows of elements of each part in each block for each thread
	uniform int nColsPerPart = (jBlockEnd - jBlockStart)/4; // n Cols of elements of each part in each block for each thread
	uniform int iRowPart = taskIndex / 4; // current row of each part
	uniform int iStart = iBlockStart + iRowPart * nRowsPerPart; // starting row of each part
	uniform int iEnd = (iStart == iBlockStart? iBlockStart + nRowsPerPart : iBlockEnd); // final row of each part
	uniform int jStart = jBlockStart + nColsPerPart * (taskIndex % 4); // starting col of each part
	uniform int jEnd = (taskIndex % 4 == 3? jBlockEnd : jStart + nColsPerPart); // final col of each part

	// SIMD with 4 ALUS
	foreach (i = iStart ... iEnd){
		for(uniform int j=jStart; j<jEnd; j++){
			double inner_prod = 0;
			for(uniform int kk=0; kk<k; kk++){
				inner_prod += A[i*k+kk] * B[kk*n+j];
			}
			C[i*n+j] = alpha * inner_prod + beta * C[i*n+j];
		}
	}
}

export void gemm_ispc(uniform int m, uniform int n, uniform int k,
	 uniform double A[], uniform double B[], uniform double C[], uniform double alpha, 
	 uniform double beta) {
	// YOUR IMPLEMENTATION HERE
	uniform int nThreads = 8;
	uniform int nBlock = m*m / 1000000 * 3 + 1;
	uniform int nPerBlock = m / nBlock;
	for(uniform int iBlock = 0; iBlock < nBlock; iBlock++){
		for(uniform int jBlock = 0; jBlock < nBlock; jBlock++){
			uniform int iBlockStart = iBlock * nPerBlock; // starting element in row of each block
			uniform int iBlockEnd = (iBlock < nBlock - 1? iBlockStart + nPerBlock : m); // final element in row of each block
			uniform int jBlockStart = jBlock * nPerBlock; // starting element in column of each block
			uniform int jBlockEnd = (jBlock < nBlock - 1? jBlockStart + nPerBlock : m); // end element in column of each block

			// Save matrix in L3 cache
			for(uniform int i = iBlockStart; i < iBlockEnd; i++){ 
				foreach (kk = 0 ... k){ 
					prefetch_l3(&A[i*k+kk]);
				}
			}

			for (uniform int kk = 0; kk < k; kk++){ 
				foreach(j = jBlockStart ... jBlockEnd){
					prefetch_l3(&B[kk*n+j]);
				}
			}

			for (uniform int i = iBlockStart; i < iBlockEnd; i++){ 
				foreach(j = jBlockStart ... jBlockEnd){ 
					prefetch_l3(&C[i*m+j]);
				}
			}
			
			launch[nThreads] gemm_ispc_task(m, n, k, A, B, C, alpha, beta, nPerBlock, iBlockStart,
											iBlockEnd, jBlockStart, nBlockEnd);
		}
	}
}