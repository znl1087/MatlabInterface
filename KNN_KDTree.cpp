#include "mex.h"
#include "flann\flann.hpp"
using namespace flann;

// 入口函数
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[] ) {
    
   	if ( nrhs != 3  ) {
        mexErrMsgTxt( "输入参数不合法……" );
    }

	if( nlhs != 2){
		mexErrMsgTxt("输出必须是两个矩阵");
	}
    
    double *dataset_d = mxGetPr(prhs[0]);
 	double *query_d = mxGetPr(prhs[1]);

	for(int i=0;i<6;i++)
		mexPrintf("%f\n",dataset_d[i]);
	int data_colums = mxGetM(prhs[0]);
	int data_rows = mxGetN(prhs[0]);
	int query_colums = mxGetM(prhs[1]);
	int query_rows = mxGetN(prhs[1]);
	int nn = (int)mxGetScalar(prhs[2]);
	//mexPrintf("%d %d %d %d %d\n",data_rows,data_colums,query_rows,query_colums,nn);
	flann::Matrix<double> points = flann::Matrix<double>(dataset_d, data_rows, data_colums);
	flann::Matrix<double> query = flann::Matrix<double>(query_d, query_rows, query_colums);

	flann::Matrix<double> dists(new double[query.rows*nn], query.rows, nn);
	flann::Matrix<int> indices(new int[query.rows*nn], query.rows, nn);

	flann::Index<flann::L2<double> > index(points, flann::KDTreeIndexParams(4));
	index.buildIndex();
	index.knnSearch(query, indices, dists, nn, flann::SearchParams(128));

	plhs[1] = mxCreateDoubleMatrix(nn,query_rows,mxREAL);
	plhs[0] = mxCreateNumericMatrix(nn,query_rows,mxINT32_CLASS,mxREAL);
	memcpy(mxGetPr(plhs[1]),dists.ptr(),mxGetNumberOfElements(plhs[1])*sizeof(double));
	memcpy(mxGetPr(plhs[0]),indices.ptr(),mxGetNumberOfElements(plhs[0])*sizeof(int));
	delete[] dists.ptr();
	delete[] indices.ptr();
}

