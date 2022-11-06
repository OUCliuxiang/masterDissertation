import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

maker = ["b.", "g.", "r.", "c.", "m.", "y.", "k.", 
         "b|", "g|", "r|", "c|", "m|", "y|", "k|", 
         "b_", "g_", "r_", "c_", "m_", "y_", "k_", 
         "b+", "g+", "r+", "c+", "m+", "y+", "k+",
         "bs", "gs", "rs", "cs", "ms", "ys", "ks",]

# names = ["pml_cls_trained", "pml_cls_untrain", "pml_trained", "pml_untrain", "cls_trained", "cls_untrain"]
names = ["pml_cls", "pml", "cls"]

plt.rcParams["axes.unicode_minus"] = False

for name in names:
    # 参数初始化
    inputfile = "feature_" + name + "_trained.csv" #销量及其他属性数据
    # outputfile = 'data_type.csv' #保存结果的文件名
    k = 12 if "trained" in name else 3 #聚类的类别
    iteration = 100 #聚类最大循环次数
    data = pd.read_csv(inputfile, index_col = 'cate') #读取数据
    data_zs = 1.0*(data - data.mean())/data.std() #数据标准化，std()表示求总体样本方差(除以n-1),numpy中std()是除以n
    print(data_zs)


    model = KMeans(n_clusters = k, max_iter = iteration) #分为k类
    #model = KMeans(n_clusters = k, n_jobs = 4, max_iter = iteration) #分为k类，并发数4
    model.fit(data_zs) #开始聚类

    #简单打印结果
    r1 = pd.Series(model.labels_).value_counts() #统计各个类别的数目
    r2 = pd.DataFrame(model.cluster_centers_) #找出聚类中心
    r = pd.concat([r2, r1], axis = 1) #横向连接（0是纵向），得到聚类中心对应的类别下的数目
    # print(r)
    r.columns = list(data.columns) + ["classes"] #重命名表头
    # print(r)

    #详细输出原始数据及其类别
    r = pd.concat([data, pd.Series(model.labels_, index = data.index)], axis = 1)  #详细输出每个样本对应的类别
    r.columns = list(data.columns) + ["clusterCate"] #重命名表头
    # r.to_csv(outputfile) #保存结果


    tsne = TSNE()
    tsne.fit_transform(data_zs)
    tsne = pd.DataFrame(tsne.embedding_, index = data_zs.index)


    for i in range(k):
        d = tsne[r["clusterCate"] == i]
        plt.plot(d[0], d[1], maker[5])

    

    inputfile = "feature_" + name + "_untrain.csv" #销量及其他属性数据
    # outputfile = 'data_type.csv' #保存结果的文件名
    k = 12 if "trained" in name else 3 #聚类的类别
    iteration = 100 #聚类最大循环次数
    data = pd.read_csv(inputfile, index_col = 'cate') #读取数据
    data_zs = 1.0*(data - data.mean())/data.std() #数据标准化，std()表示求总体样本方差(除以n-1),numpy中std()是除以n
    print(data_zs)

    model = KMeans(n_clusters = k, max_iter = iteration) #分为k类
    #model = KMeans(n_clusters = k, n_jobs = 4, max_iter = iteration) #分为k类，并发数4
    model.fit(data_zs) #开始聚类

    #简单打印结果
    r1 = pd.Series(model.labels_).value_counts() #统计各个类别的数目
    r2 = pd.DataFrame(model.cluster_centers_) #找出聚类中心
    r = pd.concat([r2, r1], axis = 1) #横向连接（0是纵向），得到聚类中心对应的类别下的数目
    # print(r)
    r.columns = list(data.columns) + ["classes"] #重命名表头
    # print(r)

    #详细输出原始数据及其类别
    r = pd.concat([data, pd.Series(model.labels_, index = data.index)], axis = 1)  #详细输出每个样本对应的类别
    r.columns = list(data.columns) + ["clusterCate"] #重命名表头
    # r.to_csv(outputfile) #保存结果

    tsne = TSNE()
    tsne.fit_transform(data_zs)
    tsne = pd.DataFrame(tsne.embedding_, index = data_zs.index)

    for i in range(k):
        d = tsne[r["clusterCate"] == i]
        plt.plot(d[0], d[1], maker[i % len(maker)])
    

    plt.savefig("cluster_" + name + "_trained_untrain.png")
    plt.close()
