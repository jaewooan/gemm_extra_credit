task void gemm_ispc_task(uniform int m, uniform int n, uniform int k,
	 uniform double A[], uniform double B[], uniform double C[], uniform double alpha,
	 uniform double beta, uniform int nRowsPerTask) {
	// YOUR IMPLEMENTATION HERE
	// uniform int iStart = nRowsPerTask * taskIndex;
	// uniform int iEnd = (taskIndex == 3? m : nRowsPerTask * (taskIndex + 1));
	 uniform int iStart = (taskIndex % 2) * nRowsPerTask;
	 uniform int iEnd = (iStart == 0? nRowsPerTask : m);
	 uniform int jStart = (taskIndex < 2? 0 : nRowsPerTask);
	 uniform int jEnd = (jStart == 0? nRowsPerTask : n);
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
	uniform int nThreads = 4;
	uniform int nRowsPerTask = m / 2;
	launch[nThreads] gemm_ispc_task(m, n, k, A, B, C, alpha, beta, nRowsPerTask);
}
