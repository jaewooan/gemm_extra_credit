uniform int nThreads = 8;
uniform int SIMD_WIDTH = 8;
uniform int L2_BLOCKSIZE = 256;
uniform int L1_BLOCKSIZE = 16;

task void gemm_ispc_task(uniform int m, uniform int n, uniform int k,
	 uniform double A[], uniform double B[], uniform double C[], uniform double alpha,
	 uniform double beta, uniform int nPerTask, uniform int iGroup) {
	// YOUR IMPLEMENTATION HERE
	uniform int nGroup = m / 256; // total number of groups
	uniform int iRow = iGroup / nGroup; // row group index
	uniform int iGroupStart = iRow * nPerTask; // starting element in row of each group
	uniform int iGroupEnd = iGroupStart + nPerTask; // final element in row of each group
	uniform int jGroupStart = nPerTask * (iGroup % nGroup); // starting element in column of each group
	uniform int jGroupEnd = jGroupStart + nPerTask; // end element in column of each group

	double inner_prod[8]; // 8 WIDE SPMD

	for (uniform int iblock2 = 0; iblock2 < m; iblock2 += L2_BLOCKSIZE){ // 8 Threads
		for (uniform int jblock2 = 0; jblock2 < n; jblock2 += L2_BLOCKSIZE){ // 8 Threads
			for (uniform int kblock2 = 0; kblock2 < k; kblock2 += L2_BLOCKSIZE){ // 8 Threads
				for (uniform int iblock1 = 0; iblock1 < L2_BLOCKSIZE; iblock1 += L1_BLOCKSIZE){ // 8 Threads
					for (uniform int jblock1 = 0; jblock1 < L2_BLOCKSIZE; jblock1 += L1_BLOCKSIZE){ // 8 Threads
						for (uniform int kblock1 = 0; kblock1 < L2_BLOCKSIZE; kblock1 += L1_BLOCKSIZE){ // 8 Threads
							for (uniform int i = 0; i<L1_BLOCKSIZE; i++){ // 8 Threads
								uniform int idx = iblock2 + iblock1 + i; //taskIndex;
								for(uniform int j = 0; j < L1_BLOCKSIZE; j+=SIMD_WIDTH){
									uniform int jdx = jblock2 + jblock1 + j + programIndex;
									inner_prod[programIndex] = 0;
									for(uniform int kk = 0; kk < L1_BLOCKSIZE; kk++){
										uniform int kdx = kblock2 + kblock1 + kk; //taskIndex;
										inner_prod[programIndex] += A[idx*k+kdx] * B[kdx*n+jdx];
									}
									C[idx*n+jdx] = alpha * inner_prod[programIndex] + beta * C[idx*n+jdx];
								}
							}
						}
					}
				}
			}
		}
	}

}

export void gemm_ispc(uniform int m, uniform int n, uniform int k,
	 uniform double A[], uniform double B[], uniform double C[], uniform double alpha, 
	 uniform double beta) {
	// YOUR IMPLEMENTATION HERE
	uniform int nGroup = m / 256;
	uniform int nPerTask = 256;
	for(uniform int iGroup = 0; iGroup < nGroup * nGroup; iGroup++){
		launch[nThreads] gemm_ispc_task(m, n, k, A, B, C, alpha, beta, nPerTask, iGroup);
	}
}
