task void gemm_ispc_task(uniform int m, uniform int n, uniform int k,
	 uniform double A[], uniform double B[], uniform double C[], uniform double alpha,
	 uniform double beta, uniform int nPerTask, uniform int iGroup) {
	// YOUR IMPLEMENTATION HERE
	// uniform int iStart = nRowsPerTask * taskIndex;
	// uniform int iEnd = (taskIndex == 8? m : nRowsPerTask * (taskIndex + 1));
	// uniform int iStart = (taskIndex % 2) * nRowsPerTask;
	// uniform int iEnd = (iStart == 0? nRowsPerTask : m);
	// uniform int jStart = (taskIndex < 2? 0 : nRowsPerTask);
	// uniform int jEnd = (jStart == 0? nRowsPerTask : n);
	uniform int nGroup = m*m / 1000000 * 3 + 1;
	uniform int iRow = iGroup / nGroup;
	uniform int iGroupStart = iRow * nPerTask;
	uniform int iGroupEnd = (iRow < nGroup - 1? iGroupStart + nPerTask : m);
	uniform int jGroupStart = nPerTask * (iGroup % nGroup);
	uniform int jGroupEnd = (iGroup % nGroup < nGroup - 1? jGroupStart + nPerTask : m);
	uniform int nRowsPerTask = (iGroupEnd - iGroupStart)/2;
	uniform int nColsPerTask = (jGroupEnd - jGroupStart)/4;

	uniform int iRowBlock = taskIndex / 4;
	uniform int iStart = iGroupStart + iRowBlock * nRowsPerTask;
	uniform int iEnd = (iStart == iGroupStart? iGroupStart + nRowsPerTask : iGroupEnd);
	uniform int jStart = jGroupStart + nColsPerTask * (taskIndex % 4);
	uniform int jEnd = (taskIndex % 4 == 3? jGroupEnd : jStart + nColsPerTask);
	foreach (i = iStart ... iEnd){
		for(uniform int j=jStart; j<jEnd; j++){
		//for(uniform int j=0; j<n; j++){
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
	uniform int nGroup = m*m / 1000000 * 3 + 1;
	uniform int nPerTask = m / nGroup;//8;
	for(uniform int iGroup = 0; iGroup < nGroup * nGroup; iGroup++){
		launch[nThreads] gemm_ispc_task(m, n, k, A, B, C, alpha, beta, nPerTask, iGroup);
	}
}
