uniform int nThreads = 8;
int SIMD_WIDTH = 8;
int L2_BLOCKSIZE = 256;
int L1_BLOCKSIZE = 16;

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

	for (int iblock2 = 0; iblock2 < m; iblock2 += L2_BLOCKSIZE){ // 8 Threads
		for (int jblock2 = 0; jblock2 < n; jblock2 += L2_BLOCKSIZE){ // 8 Threads
			for (int kblock2 = 0; kblock2 < k; kblock2 += L2_BLOCKSIZE){ // 8 Threads
				for (int iblock1 = 0; iblock1 < L2_BLOCKSIZE; iblock1 += L1_BLOCKSIZE*8){ // 8 Threads
					for (int jblock1 = 0; jblock1 < L2_BLOCKSIZE; jblock1 += L1_BLOCKSIZE){ // 8 Threads
						for (int kblock1 = 0; kblock1 < L2_BLOCKSIZE; kblock1 += L1_BLOCKSIZE){ // 8 Threads
							for (int i = 0; i<L1_BLOCKSIZE; i++){ // 8 Threads
								int idx = iblock2 + (iblock1 + taskIndex * L1_BLOCKSIZE) + i; //taskIndex;
								for(int j = 0; j < L1_BLOCKSIZE; j+=SIMD_WIDTH){
									int jdx = jblock2 + jblock1 + j + programIndex;
									inner_prod[programIndex] = 0;
									for(int kk = 0; kk < L1_BLOCKSIZE; kk++){
										int kdx = kblock2 + kblock1 + kk; //taskIndex;
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
