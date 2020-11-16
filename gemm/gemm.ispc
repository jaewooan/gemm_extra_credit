task void gemm_ispc_task(uniform int m, uniform int n, uniform int k,
	 uniform double A[], uniform double B[], uniform double C[], uniform double alpha,
	 uniform double beta, uniform int nRowsPerTask) {
	// YOUR IMPLEMENTATION HERE
	uniform int iStart = nRowsPerTask * taskIndex;
	uniform int iEnd = (taskIndex == 3? m : nRowsPerTask * (taskIndex + 1));
	foreach (i = iStart ... iEnd){
		for(uniform int j=0; j<n; j++){
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
	uniform int nThreads = 4;
	uniform int nRowsPerTask = m / nThreads;
	launch[nThreads] gemm_ispc_task(m, n, k, A, B, C, alpha, beta, nRowsPerTask);
}
