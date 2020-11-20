uniform int nThreads = 8;
int SIMD_WIDTH = 8;
uniform int L3_BLOCKSIZE = 256;
uniform int L2_BLOCKSIZE = 16;

task void gemm_ispc_task(uniform int m, uniform int n, uniform int k,
	 uniform double A_L3[], uniform double B_L3[], uniform double C[], uniform double alpha,
	 uniform double beta, uniform int iblock3, uniform int jblock3, uniform int kblock3) {
	// YOUR IMPLEMENTATION HERE
	uniform double inner_prod[8]; // 8 WIDE SPMD
	for (uniform int iblock2 = 0; iblock2 < L3_BLOCKSIZE; iblock2 += L2_BLOCKSIZE*8){ // 8 Threads
		for (uniform int jblock2 = 0; jblock2 < L3_BLOCKSIZE; jblock2 += L2_BLOCKSIZE){ // 8 Threads
			for (uniform int kblock2 = 0; kblock2 < L3_BLOCKSIZE; kblock2 += L2_BLOCKSIZE){
				// Save matrix in L2 Caches
				//double A_L1[16*16];
				for (uniform int i = 0; i < L2_BLOCKSIZE; i++){ 
					foreach (kk = 0 ... L2_BLOCKSIZE){
						prefetch_l2(A[(iblock3 + iblock2 + L2_BLOCKSIZE * taskIndex + i)*k+(kblock3 + kblock2 + kk)]);
					}
				}

				//double B_L1[16*16];
				for (uniform int kk = 0; kk < L2_BLOCKSIZE; kk++){ 
					foreach (j = 0 ... L2_BLOCKSIZE){ 
						prefetch_l2(B[(kblock3 + kblock2 + kk)*n+(jblock3 + jblock2 + j)]);
					}
				}

				for (uniform int i = 0; i < L2_BLOCKSIZE; i++){ // 8 Threads
					uniform int idx = iblock3 + iblock2 + L2_BLOCKSIZE * taskIndex + i;
					foreach(j = 0 ... L2_BLOCKSIZE){
						uniform int jdx = jblock3 + jblock2 + j;
						uniform int j_ = k % 8;
						inner_prod[j_] = 0;
						for(uniform int kk = 0; kk < L2_BLOCKSIZE; kk++){							
							inner_prod[j_] += A[idx*k+(kblock3 + kblock2 + kk)] * B[(kblock3 + kblock2 + kk)*n+jdx];
						}
						C[idx*n+jdx] = alpha * inner_prod[j_] + beta * C[idx*n+jdx];
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
	//uniform double A_L3[256*256];
	//uniform double B_L3[256*256];

	for (uniform int iblock3 = 0; iblock3 < m; iblock3 += L3_BLOCKSIZE){ // 8 Threads
		for (uniform int jblock3 = 0; jblock3 < n; jblock3 += L3_BLOCKSIZE){ // 8 Threads
			for (uniform int kblock3 = 0; kblock3 < k; kblock3 += L3_BLOCKSIZE){ // 8 Threads
				// Save matrix in L3 cache
				for(uniform int i = 0; i < L3_BLOCKSIZE; i++){ 
					foreach (kk = 0 ... L3_BLOCKSIZE){ 
						prefetch_l3(A[(iblock3 + i)*k+(kblock3 + kk)]);
					}
				}

				for (uniform int kk = 0; kk < L3_BLOCKSIZE; kk++){ 
					foreach(j = 0 ... L3_BLOCKSIZE){ 
						prefetch_l3(B[(kblock3 + kk)*n+(jblock3 + j)]);
					}
				}
				launch[nThreads] gemm_ispc_task(m, n, k, A_L3, B_L3, C, alpha, beta, iblock3, jblock3, kblock3);
			}
		}
	}	
}