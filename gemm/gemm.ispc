uniform int rowGroup = 200; // num element = row*col + m*row + col * m << 1000000
uniform int colGroup = 4; // number of ALUs
uniform int nThreads = 8;
uniform int rowBlock = 25; //rowGroup / nThreads; // 25
uniform int colBlock = 4; //colGroup;     // 4

task void gemm_ispc_task(uniform int m, uniform int n, uniform int k,
	 uniform double A[], uniform double B[], uniform double C[], uniform double alpha,
	 uniform double beta, uniform int iGroup) {
	// YOUR IMPLEMENTATION HERE
	// 8 Unit blocks = 1 execution (big block)
	// uniform int iStart = nRowsPerTask * taskIndex;
	// uniform int iEnd = (taskIndex == 8? m : nRowsPerTask * (taskIndex + 1));
	// uniform int iStart = (taskIndex % 2) * nRowsPerTask;
	// uniform int iEnd = (iStart == 0? nRowsPerTask : m);
	// uniform int jStart = (taskIndex < 2? 0 : nRowsPerTask);
	// uniform int jEnd = (jStart == 0? nRowsPerTask : n);

	// taskIndex: 0~7 (8 threads)
	uniform double unit_block[25][4];

	uniform int nRowGroup = (m - 1) / rowGroup + 1;
	uniform int nColGroup = (m - 1) / colGroup + 1;
	uniform int iRowGroup = iGroup / nColGroup;
	uniform int jColGroup = iGroup % nColGroup;

	uniform int iStartGroup = iRowGroup * rowGroup;
	uniform int iEndGroup = (iRowGroup == nRowGroup - 1 ? m : iStart + rowBlock);
	uniform int jStart = jColGroup * colGroup;
	uniform int jEnd = (jColGroup == nColGroup - 1 ? m : jStart + colGroup);

	uniform int nRowElementGroup = iEndGroup - iStartGroup;
	uniform int invGroup = nRowElementGroup / nThreads;
	uniform int iStartBlock = iStartGroup + taskIndex * invGroup;  
	uniform int iEndBlock = (taskIndex == 7 ? iEndGroup : iStartBlock + invGroup);  

	// Initialization
	//uniform double A_target[rowBlock][k];
	for(uniform int v = 0; v < rowBlock; v++){
		for(uniform int h = 0; h < colBlock; h++){
			unit_block[v][h] = 0;
		}
		/*for(uniform kk = 0; kk < k; kk++){
			A_target[v][kk] = A[(iStartBlock + v) * k + kk];
		}*/
	}

	// Multication
	for(uniform v = 0; v < rowBlock; v++){
		for(uniform kk = 0; kk < k; kk++){
			foreach(h = 0 ... colBlock){
				int jdx = jStart + h;
				unit_block[v][h] += A[(iStartBlock + v) * k + kk] * B[kk*n+jStart+h];
			}
		}
	}

	// Save
	for(uniform int v = 0; i < rowBlock; i++){
		int idx = iStartBlock + v;
		for(uniform int h = 0; h < colBlock; h++){
			C[idx*n+jStart+j] = alpha * unit_block[v][h] + beta * C[idx*n+jStart+j];
		}
	}
}

export void gemm_ispc(uniform int m, uniform int n, uniform int k,
	 uniform double A[], uniform double B[], uniform double C[], uniform double alpha, 
	 uniform double beta) {
	// YOUR IMPLEMENTATION HERE
	uniform int nRowGroup = (m - 1) / rowGroup + 1;
	uniform int nColGroup = (m - 1) / colGroup + 1;
	for(uniform int iGroup = 0; iGroup < nRowGroup * nColGroup; iGroup++){
		launch[nThreads] gemm_ispc_task(m, n, k, A, B, C, alpha, beta, iGroup);
	}
}
