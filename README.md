# MatlabInterface
the Interface of Matlab about LSH and KDTree
# 代码说明
1. KNN_KDTree.cpp Flann库的KDTree实现Matlab接口
2. KNN_MyLsh.cpp 我自己实现的LSH算法的Matlab接口
3. TestTheLSH.m matlab文件，包括了对上述文件的编译及执行

#使用说明
1. 由于大数据运行速度较慢，我采用了小数据进行验证（这是我做的不足之处）
2. 在TestTheLSH.m中，将mex编译选项中的目录改为电脑的Flann库所在目录
3. 运行TestTheLSH.m，可以在Matlab中查看生成的结果。
