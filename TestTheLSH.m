%dataset = fvecs_read('sift\sift_base.fvecs');
%query = fvecs_read('sift\sift_query.fvecs');
%dataset = double(dataset);
%query = double(query);
dataset = (load('Iris90.txt'))';
query = dataset;
%mex -g KNN_KDTree.cpp -I'E:\FLANN\flann\include' -L'E:\FLANN\flann\lib'
%mex -g KNN_MyLsh.cpp -I'E:\FLANN\flann\include' -L'E:\FLANN\flann\lib'
[indices_lsh,dists_lsh] = KNN_MyLsh(dataset,query,5);
[indices_KDTree,dists_KDTree] = KNN_KDTree(dataset,query,5);