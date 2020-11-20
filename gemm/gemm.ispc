uniform int nThreads = 8;
int SIMD_WIDTH = 8;
int L2_BLOCKSIZE = 256;
int L1_BLOCKSIZE = 16;

task void gemm_ispc_task(uniform int m, uniform int n, uniform int k,
	 uniform double A_L2[256][256], uniform double B_L2[256][256], uniform double C[], uniform double alpha,
	 uniform double beta, int iblock2, int jblock2, int kblock2) {
	// YOUR IMPLEMENTATION HERE

	double inner_prod[8]; // 8 WIDE SPMD
	for (int iblock1 = 0; iblock1 < L2_BLOCKSIZE; iblock1 += L1_BLOCKSIZE*8){ // 8 Threads
		for (int jblock1 = 0; jblock1 < L2_BLOCKSIZE; jblock1 += L1_BLOCKSIZE){ // 8 Threads
			for (int kblock1 = 0; kblock1 < L2_BLOCKSIZE; kblock1 += L1_BLOCKSIZE){
				// Save matrix in L1 blocks
				double A_L1[16][16];
				double B_L1[16][16];
				for (int kk = 0; kk < L1_BLOCKSIZE; kk++){ 
					int kdx = kblock1 + kk;
					for (int i = 0; i < L1_BLOCKSIZE; i++){ 
						int idx = iblock1 + L1_BLOCKSIZE * taskIndex + i;
						A_L1[i][kk] = A_L2[idx][kdx];
					}
					for (int j = 0; j < L1_BLOCKSIZE; j++){ 
						int jdx = jblock1 + j;
						B_L1[kk][j] = B_L2[kdx][jdx];
					}
				}

				for (int i = 0; i < L1_BLOCKSIZE; i++){ // 8 Threads
					int idx = iblock2 + iblock1 + L1_BLOCKSIZE * taskIndex + i;
					for(int j = 0; j < L1_BLOCKSIZE; j+=SIMD_WIDTH){
						int jdx = jblock2 + jblock1 + j + programIndex;
						inner_prod[programIndex] = 0;
						for(int kk = 0; kk < L1_BLOCKSIZE; kk++){
							inner_prod[programIndex] += A_L1[i][kk] * B_L1[kk][j+ programIndex];
						}
						C[idx*n+jdx] = alpha * inner_prod[programIndex] + beta * C[idx*n+jdx];
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
	uniform double A_L2[256][256];
	uniform double B_L2[256][256];

	for (int iblock2 = 0; iblock2 < m; iblock2 += L2_BLOCKSIZE){ // 8 Threads
		for (int jblock2 = 0; jblock2 < n; jblock2 += L2_BLOCKSIZE){ // 8 Threads
			for (int kblock2 = 0; kblock2 < k; kblock2 += L2_BLOCKSIZE){ // 8 Threads
				// Save matrix in L2 blocks
				for (int kk = 0; kk < L2_BLOCKSIZE; kk++){ 
					int kdx = kblock2 + kk;
					for (int i = 0; i < L2_BLOCKSIZE; i++){ 
						int idx = iblock2 + i;
						A_L2[i][kk] = A[idx*k+kdx];
					}
					for (int j = 0; j < L2_BLOCKSIZE; j++){ 
						int jdx = jblock2 + j;
						B_L2[kk][j] = B[kdx*n+jdx];
					}
				}
				launch[nThreads] gemm_ispc_task(m, n, k, A_L2, B_L2, C, alpha, beta, iblock2, jblock2, kblock2);
			}
		}
	}	
}
