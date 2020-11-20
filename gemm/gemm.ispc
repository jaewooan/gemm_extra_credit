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
	for (uniform int i = iGroupStart; i<iGroupEnd; i += 8){ // 8 Threads
		uniform int idx = i + taskIndex;
		for(int j = jGroupStart; j < jGroupEnd; j += 8){
			int jdx = j + programIndex;
			inner_prod[programIndex] = 0;
			for(uniform int kk = 0; kk < k; kk++){
				inner_prod[programIndex] += A[idx*k+kk] * B[kk*n+jdx];
			}
			C[idx*n+jdx] = alpha * inner_prod[programIndex] + beta * C[idx*n+jdx];
		}
	}
}

export void gemm_ispc(uniform int m, uniform int n, uniform int k,
	 uniform double A[], uniform double B[], uniform double C[], uniform double alpha, 
	 uniform double beta) {
	// YOUR IMPLEMENTATION HERE
	uniform int nThreads = 8;
	uniform int nGroup = m / 256;
	uniform int nPerTask = 256;
	for(uniform int iGroup = 0; iGroup < nGroup * nGroup; iGroup++){
		launch[nThreads] gemm_ispc_task(m, n, k, A, B, C, alpha, beta, nPerTask, iGroup);
	}
}
