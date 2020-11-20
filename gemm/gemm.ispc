uniform int nThreads = 8;
int SIMD_WIDTH = 8;
uniform int L3_BLOCKSIZE = 256;
uniform int L2_BLOCKSIZE_I = 32;
uniform int L2_BLOCKSIZE_J = 256;
uniform int L2_BLOCKSIZE_K = 256;

task void gemm_ispc_task(uniform int m, uniform int n, uniform int k,
	 uniform double A[], uniform double Btrans[], uniform double C[], uniform double alpha,
	 uniform double beta, uniform int iblock3, uniform int jblock3, uniform int kblock3) {
	// YOUR IMPLEMENTATION HERE
	//double inner_prod[8]; // 8 WIDE SPMD
	for (int iblock2 = 0; iblock2 < L2_BLOCKSIZE_I; iblock2++){
		int idx = iblock3 + taskIndex * L2_BLOCKSIZE_I + iblock2;
		for (int jblock2 = 0; jblock2 < L2_BLOCKSIZE_J; jblock2 ++){ 
			int jdx = jblock3 + jblock2;
			// Save C in L1 cache
			//prefetch_l1(&C[idx*m+jdx]);
			double C_scalar = 0;
			foreach (kblock2 = 0 ... L2_BLOCKSIZE_K){
				int kdx = kblock3 + kblock2; 
				//prefetch_l2(&A[idx*k+kdx]);
				//prefetch_l2(&Btrans[jdx*k+kdx]);
				C_scalar +=  A[idx*k+kdx] * Btrans[jdx*k+kdx];
			}
			C[idx*m+jdx] = alpha * C_scalar + beta * C[idx*m+jdx];
		}
	}

}

export void gemm_ispc(uniform int m, uniform int n, uniform int k,
	 uniform double A[], uniform double B[], uniform double C[], uniform double alpha, 
	 uniform double beta) {
	// YOUR IMPLEMENTATION HERE
	//uniform double A_L3[256*256];
	uniform double Btrans[256*256];

	for (uniform int iblock3 = 0; iblock3 < m; iblock3 += L3_BLOCKSIZE){ // 8 Threads
		for (uniform int jblock3 = 0; jblock3 < n; jblock3 += L3_BLOCKSIZE){ // 8 Threads
			for (uniform int kblock3 = 0; kblock3 < k; kblock3 += L3_BLOCKSIZE){ // 8 Threads
				// Save matrix in L3 cache
				/*for(uniform int i = 0; i < L3_BLOCKSIZE; i++){ 
					foreach (kk = 0 ... L3_BLOCKSIZE){ 
						prefetch_l3(&A[(iblock3 + i)*k+(kblock3 + kk)]);
					}
				}*/

				for (uniform int kk = 0; kk < L3_BLOCKSIZE; kk++){ 
					foreach(j = 0 ... L3_BLOCKSIZE){
						Btrans[(jblock3 + j)*k+(kblock3 + kk)] = B[(kblock3 + kk)*n+(jblock3 + j)]; 
				//		prefetch_l3(&Btrans[(jblock3 + j)*k+(kblock3 + kk)]);
					}
				}

				/*for (uniform int i = 0; i < L3_BLOCKSIZE; i++){ 
					foreach(j = 0 ... L3_BLOCKSIZE){ 
						prefetch_l3(&C[(iblock3 + i)*m+(jblock3 + j)]);
					}
				}*/
				launch[nThreads] gemm_ispc_task(m, n, k, A, Btrans, C, alpha, beta, iblock3, jblock3, kblock3);
			}
		}
	}	
}
