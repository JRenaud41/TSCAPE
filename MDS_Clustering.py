import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
import sklearn.metrics as metrics
import hdbscan
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding
from sklearn_extra.cluster import KMedoids
df=pd.read_csv('DTW_MatrixSoft0.01.csv',sep=";") #Use the matrix create by the DTW
print(df) 
df = df.iloc[: , 1:]
name = df.iloc[: , 0]
df = df.iloc[: , 1:]
print(name)
print(df)
df = df.abs()
plt.rcParams.update({'font.size': 22})

print(df)

#########################################EMBEDDING/DIM REDUCE PART######################################### 
# loop for testing with more dim. This case is for dim=2.
Ndim = [1]   
for cpt2 in range(1,2):   
    res = []
    resTSNE = []
    resCHScore = []
    resCHScoreTSNE = []
    resDBScore = []
    resDBScoreTSNE = []
    
    Ndim.append(cpt2+1)
    #Run 10 times 
    for cpt in range(0,10):
        #Choose your embedding
        mds = manifold.MDS(n_components=cpt2+1, dissimilarity="precomputed", random_state=10)
        #spe = SpectralEmbedding(n_components=cpt2+1,affinity="precomputed")
        results = mds.fit(df)
        #results = spe.fit(df)
        
        print(Ndim)
        coords = results.embedding_
        coords = pd.DataFrame(coords, columns=Ndim)
        #coords = df
        plt.figure(figsize = (15,15))
        plt.scatter(coords.iloc[:,0], coords.iloc[:,1], s=50, linewidth=0, alpha=0.6)
        #plt.show()
        #########################################CLUSTERING PART#########################################

        ############################################################################
        #Use this to visualize the elbow and choose the number of cluster 
        '''sse = []
        k_list = range(2,22)
        for k in k_list:
            print(str(k)+'/'+(str(len(k_list))))
            #km =  hdbscan.HDBSCAN(min_cluster_size=k,min_samples=3,cluster_selection_method='leaf')
            km = SpectralClustering(n_clusters=k,assign_labels='discretize',affinity='nearest_neighbors',random_state=0)
            #km = KMeans(n_clusters=k, n_init=100, max_iter=400, init='k-means++', random_state=42)
            #km = KMedoids(n_clusters=k, max_iter=400, init='k-medoids++',random_state=42)
            km.fit(coords)
            sse.append([k, km.inertia_])
            #sse.append([k, metrics.silhouette_score(coords, km.labels_)])
            #sse.append([k,silhouette_score(coords, km.labels_, metric='euclidean')])
          
        plt.figure(figsize=(8,4))
        plt.plot(pd.DataFrame(sse)[0], pd.DataFrame(sse)[1], marker='o', linewidth='4')
        plt.title('Elbow method')
        plt.xlabel('Number of cluster')
        plt.ylabel('Inertia')
        #plt.show()'''

        ############################################################################

        #Choose your clustering algorithm 

        #clusterer =hdbscan.HDBSCAN(min_cluster_size=k,min_samples=3,cluster_selection_method='leaf').fit(coords)
        #clusterer = SpectralClustering(n_clusters=10,affinity='nearest_neighbors',random_state=0).fit(coords)
        #clusterer = KMeans(n_clusters=10, n_init=100, max_iter=400, init='k-means++', random_state=42).fit(coords)
        clusterer = KMedoids(n_clusters=8, max_iter=400, init='k-medoids++',random_state=42).fit(df)

        res.append(silhouette_score(coords, clusterer.labels_, metric='euclidean'))
        resCHScore.append(calinski_harabasz_score(coords, clusterer.labels_))
        resDBScore.append(davies_bouldin_score(coords, clusterer.labels_))
        tsne_coords = pd.DataFrame(coords, columns=['x', 'y'])
        clusters = pd.concat([coords,name, pd.DataFrame({'cluster':clusterer.labels_})], axis=1)
        clusters.to_csv('cluster.csv', sep=';', encoding='utf-8')
        #plot
        plt.figure(figsize = (15,15))
        plt.scatter(clusters.iloc[:,0],clusters.iloc[:,1], s=50, linewidth=0, c=clusterer.labels_, alpha=0.6)
        plt.legend()
        #plt.show()
        ####################################################################################################
        #Add t_SNE in our test
        #tsne
        tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=10000, learning_rate=200)
        tsne_coords = tsne.fit_transform(coords)
        plt.scatter(tsne_coords[:,0],tsne_coords[:,1], s=50, linewidth=0, alpha=0.6)
        #plt.show()
        #Clustering with t_sne
        clusterer = SpectralClustering(n_clusters=10 ,affinity='nearest_neighbors',random_state=0).fit(tsne_coords)
        #clusterer =hdbscan.HDBSCAN(min_cluster_size=k,min_samples=3,cluster_selection_method='leaf').fit(tsne_coords)
        #clusterer = KMeans(n_clusters=10, n_init=100, max_iter=400, init='k-means++', random_state=42).fit(tsne_coords)
        #clusterer = KMedoids(n_clusters=10, max_iter=400, init='k-medoids++',random_state=42).fit(tsne_coords)
        
        resTSNE.append(silhouette_score(tsne_coords, clusterer.labels_, metric='euclidean'))
        resCHScoreTSNE.append(calinski_harabasz_score(tsne_coords, clusterer.labels_))
        resDBScoreTSNE.append(davies_bouldin_score(tsne_coords, clusterer.labels_))
        tsne_coords = pd.DataFrame(tsne_coords, columns=['x', 'y'])
        clusters = pd.concat([tsne_coords,name, pd.DataFrame({'cluster':clusterer.labels_})], axis=1)
        clusters.to_csv('clusterTSNE.csv', sep=';', encoding='utf-8')

        #plot
        plt.figure(figsize = (15,15))
        plt.scatter(clusters.iloc[:,0],clusters.iloc[:,1], s=50, linewidth=0, c=clusterer.labels_, alpha=0.6)
        plt.legend()
        #plt.show()

        ####################################################################################################
    #Print result     
    print('--------------silhouette_score : ')
    print(res)
    print('Mean :',(sum(res)/len(res)))
    print('STD :',np.std(res))
    print('###############')
    print(resTSNE)
    print('Mean TSNE :',(sum(resTSNE)/len(resTSNE)))
    print('STD TSNE:',np.std(resTSNE))

    print('------------------------calinski_harabasz_score : ')
    print(resCHScore)
    print('Mean :',(sum(resCHScore)/len(resCHScore)))
    print('STD :',np.std(resCHScore))
    print('###############')
    print(resCHScoreTSNE)
    print('Mean TSNE :',(sum(resCHScoreTSNE)/len(resCHScoreTSNE)))
    print('STD TSNE :',np.std(resCHScoreTSNE))

    print('----------------------davies_bouldin_score : ')
    print(resDBScore)
    print('Mean :',(sum(resDBScore)/len(resDBScore)))
    print('STD :',np.std(resDBScore))
    print('###############')
    print(resDBScoreTSNE)
    print('Mean TSNE :',(sum(resDBScoreTSNE)/len(resDBScoreTSNE)))
    print('STD TSNE :',np.std(resDBScoreTSNE))

   

