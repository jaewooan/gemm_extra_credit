uniform int rowGroup = 100; // num element = row*col + m*row + col * m << 1000000
uniform int colGroup = 4; // number of ALUs
uniform int nThreads = 8;
uniform int rowBlock = 25; //rowGroup / nThreads; // 25
uniform int colBlock = 4; //colGroup;     // 4

task void gemm_ispc_task(uniform int m, uniform int n, uniform int k,
	 uniform double A[], uniform double B[], uniform double C[], uniform double alpha,
	 uniform double beta, uniform int iStartGroup, uniform int intv_, uniform int iEndGroup, 
	 uniform int jStartBlock) {
	// YOUR IMPLEMENTATION HERE
	uniform double unit_block[25][4];
	uniform int iStartBlock = iStartGroup + taskIndex * intv_;  // 200+49 = 249 
	uniform int iEndBlock = (taskIndex == 7 ? iEndGroup : iStartBlock + intv_); //256 
	uniform int intvBlock = iEndBlock - iStartBlock;

	// Initialization
	//uniform double A_target[rowBlock][k];
	for(uniform int v = 0; v < rowBlock; v++){ //25
		foreach(h = 0 ... colBlock){ //4
			unit_block[v][h] = 0;
		}
		/*for(uniform kk = 0; kk < k; kk++){
			A_target[v][kk] = A[(iStartBlock + v) * k + kk];
		}*/
	}

	// Multication
	for(uniform int v = 0; v < intvBlock; v++){ //25
		for(uniform int kk = 0; kk < k; kk++){ // 256
			foreach(h = 0 ... colBlock){ //4
				unit_block[v][h] += A[(iStartBlock + v) * k + kk] * B[kk*n+jStartBlock+h];
			}
		}
	}

	// Save
	for(uniform int v = 0; v < intvBlock; v++){
		foreach(h = 0 ... colBlock){ //4
			C[(iStartBlock + v)*n+jStartBlock+h] = alpha * unit_block[v][h] + beta * C[(iStartBlock + v)*n+jStartBlock+h];
		}
	}
}

export void gemm_ispc(uniform int m, uniform int n, uniform int k,
	 uniform double A[], uniform double B[], uniform double C[], uniform double alpha, 
	 uniform double beta) {
	// YOUR IMPLEMENTATION HERE
	uniform int nRowGroup = (m - 1) / rowGroup + 1; //2
	uniform int nColGroup = (m - 1) / colGroup + 1; //64
	for(uniform int iGroup = 0; iGroup < nRowGroup * nColGroup; iGroup++){
		uniform int iRowGroup = iGroup / nColGroup; // 0~0, 1~1 -> 1
		uniform int jColGroup = iGroup % nColGroup; // 0~63, 0~63 -> 63
		uniform int iStartGroup = iRowGroup * rowGroup; // 200
		uniform int iEndGroup = (iRowGroup == nRowGroup - 1 ? m : iStartGroup + rowGroup); // 256
		uniform int jStart = jColGroup * colGroup; //252
		uniform int nRowElementGroup = iEndGroup - iStartGroup; //56
		uniform int intvBlock = nRowElementGroup / nThreads; // 56/8 = 7 
		launch[nThreads] gemm_ispc_task(m, n, k, A, B, C, alpha, beta, iStartGroup, intv_, iEndGroup, jStartBlock);
	}
}
