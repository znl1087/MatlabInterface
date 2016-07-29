#include "mex.h"
#include "flann\flann.hpp"
#include <algorithm>
#include <iostream>
#include <vector>
#include <iomanip>
#include <limits.h>
#include <map>
#include <random>
namespace MyFlann{
class MylshTable{
public:
	MylshTable(){
	}

	MylshTable(double w, int d, int kkk, flann::Matrix<double> features):w(w),d(d),kkk(kkk){
		bParams = GenUniform(kkk, w);
		r1 = GenUniform(kkk, Prime);
		r2 = GenUniform(kkk, Prime);
		for (int i = 0; i < kkk; i++){
			aParams.push_back(GenNormal(d));
		}
		for (int i = 0; i < features.rows; i++){
			add(features[i], i);
		}
	}
	std::vector<int> GetNeibours(const double * vec)const{
		auto ans = h(vec);
		std::vector<int> out;
		auto i = Table[ans.first].begin();
		for (; i != Table[ans.first].end(); i++){
			if (ans.second == i->first){
				out.push_back(i->second);
			}
		}
		return out;
	}
	//向哈希表中插入一个点
	void add(const double* vec, const int index){
		auto ans = h(vec);
		Table[ans.first].push_back(std::make_pair(ans.second, index));
	}
	std::pair<int, int> h(const double* vec)const{
		std::vector<int> hash1;
		for (int i = 0; i < kkk; i++){
			double sum = 0;
			for (int d = 0; d < d; d++){
				sum += vec[d] * aParams[i][d];
			}
			int ans = (int)std::ceil((sum + bParams[i]) / w);
			hash1.push_back(ans);
		}
		std::pair<int, int> ans = { 0, 0 };
		for (int i = 0; i < kkk; i++){
			ans.first = (ans.first + (hash1[i] % Prime) * (long long)(r1[i] % Prime)) % Prime;
			ans.second = (ans.second + (hash1[i] % Prime) * (long long)(r2[i] % Prime)) % Prime;
		}
		ans.first %= TableSize;
		return ans;
	}
	//利用normal_distribution生成满足标准正态分布的随机向量
	std::vector<double> GenNormal(int n){
		std::vector<double> ans;
		std::random_device rd;
		std::mt19937 gen(rd());
		std::normal_distribution<double> d(0, 1);
		for (int i = 0; i < n; i++){
			ans.push_back(d(gen));
		}
		return ans;
	}
	//生成均匀分布的随机小数b
	std::vector<double> GenUniform(int n, double w){
		std::vector<double> ans;
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<double> dis(0, w);

		for (int i = 0; i < n; i++){
			ans.push_back(dis(gen));
		}
		return ans;
	}
	std::vector<int> GenUniform(int n, int w){
		std::vector<int> ans;
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<int> dis(0, w);

		for (int i = 0; i < n; i++){
			ans.push_back(dis(gen));
		}
		return ans;
	}

	const int Prime = (int)1e9 + 7;

	const static int TableSize = 1000005;

	int d; //输入向量的维数

	int kkk; //哈希函数的个数

	std::vector<std::vector<double> > aParams; //哈希函数的随机参数a

	std::vector<double> bParams;			//哈希函数的随机参数b

	std::vector<int> r1;

	std::vector<int> r2;

	std::vector<std::pair<int, int> > Table[TableSize];	//哈希表

	double w; //用于分段的长度 
};


	class MyIndex{
	public:
		MyIndex(){}
		MyIndex(const flann::Matrix<double>& features){
			setDataset(features);
			table = new MylshTable(1.0, (int)features.rows, 10, features);
		}
		~MyIndex(){
			delete table;
		}
		void setDataset(const flann::Matrix<double>& dataset)
		{
			size_ = dataset.rows;
			veclen_ = dataset.cols;
			last_id_ = 0;

			ids_.clear();

			points_.resize(size_);
			for (size_t i = 0; i < size_; ++i) {
				points_[i] = dataset[i];
			}
		}

		virtual int knnSearch(const flann::Matrix<double>& queries,
			flann::Matrix<size_t>& indices,
			flann::Matrix<double>& dists,
			size_t knn) const
		{
			int count = 0;

			flann::KNNSimpleResultSet<double> resultSet(knn);
			for (int i = 0; i < (int)queries.rows; i++) {
				resultSet.clear();
				findNeighbors(resultSet, queries[i]);
				int n = (int)std::min(resultSet.size(), knn);
				resultSet.copy(indices[i], dists[i], n);
				count += n;
				mexPrintf("%d\n",i);
			}
			return count;
		}
		void findNeighbors(flann::ResultSet<double>& result, const double* vec)const{
			auto ans = table->GetNeibours(vec);
			flann::L2<double> distance;
			for (auto i = ans.begin(); i != ans.end(); i++){
				double dis = distance(vec, points_[*i], veclen_);
				result.addPoint(dis, *i);
			}
		}
	protected:
		/**
		* Each index point has an associated ID. IDs are assigned sequentially in
		* increasing order. This indicates the ID assigned to the last point added to the
		* index.
		*/
		size_t last_id_;
		/**
		* Number of points in the index (and database)
		*/
		size_t size_;


		/**
		* Size of one point in the index (and database)
		*/
		size_t veclen_;

		/**
		* Array of point IDs, returned by nearest-neighbour operations
		*/
		std::vector<size_t> ids_;

		/**
		* Point data
		*/
		std::vector<double*> points_;

		MylshTable *table;
	};
}
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
	int data_colums = (int)mxGetM(prhs[0]);
	int data_rows = (int)mxGetN(prhs[0]);
	int query_colums = (int)mxGetM(prhs[1]);
	int query_rows = (int)mxGetN(prhs[1]);
	int nn = (int)mxGetScalar(prhs[2]);
	//mexPrintf("%d %d %d %d %d\n",data_rows,data_colums,query_rows,query_colums,nn);
	flann::Matrix<double> points = flann::Matrix<double>(dataset_d, data_rows, data_colums);
	flann::Matrix<double> query = flann::Matrix<double>(query_d, query_rows, query_colums);

	flann::Matrix<double> dists(new double[query.rows*nn], query.rows, nn);
	flann::Matrix<size_t> indices(new size_t[query.rows*nn], query.rows, nn);

	MyFlann::MyIndex index = MyFlann::MyIndex(points);
	index.knnSearch(query, indices, dists, nn);

	plhs[1] = mxCreateDoubleMatrix(nn,query_rows,mxREAL);
	plhs[0] = mxCreateNumericMatrix(nn,query_rows,mxUINT64_CLASS,mxREAL);//若32位改成mxUINT32
	memcpy(mxGetPr(plhs[1]),dists.ptr(),mxGetNumberOfElements(plhs[1])*sizeof(double));
	memcpy(mxGetPr(plhs[0]),indices.ptr(),mxGetNumberOfElements(plhs[0])*sizeof(size_t));
	delete[] dists.ptr();
	delete[] indices.ptr();
}

