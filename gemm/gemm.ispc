export void gemm_ispc(uniform int m, uniform int n, uniform int k,
	 uniform double A[], uniform double B[], uniform double C[], uniform double alpha, uniform double beta) {
	// YOUR IMPLEMENTATION HERE

	foreach (i = 0 ... m){
		for(uniform int j=0; j<n; j++){
			double inner_prod = 0;
			for(uniform int kk=0; kk<k; kk++){
				inner_prod += A[i*k+kk] * B[kk*n+j];
			}
			C[i*n+j] = alpha * inner_prod + beta * C[i*n+j];
		}
	}
}
