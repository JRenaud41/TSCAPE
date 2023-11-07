import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import ruptures as rpt
from tslearn.metrics import dtw
from tslearn.metrics import soft_dtw
from tqdm import tqdm
import warnings
import os 
warnings.filterwarnings("ignore")

df=pd.read_csv('dataset.csv',sep=";")

#Dataset structure :  
# -> nbVente : value of time series 
# -> date : date of the day for the number of sales 
# -> nom : id of the time-series 

df =df.dropna()
df['date'] = pd.to_datetime(df['date'],dayfirst =True )
df['date'] = df['date'].dt.strftime('%d.%m.%Y')
#print(df)
#print (df.date.min())
#print (df.date.max())
nomCat = df.groupby(['nom']).agg({'nbVente':'sum'}).reset_index()
nomCat = nomCat.sort_values(by=['nbVente'],ascending=False)
nomCat = nomCat['nom']
distance = []
print(nomCat)
gamma = 0.005 #gamma parameter for SoftDTW version
cpt = nomCat

for i, nomI in enumerate(tqdm(cpt, desc ="global progression")):

	distanceLigne = []
	distanceLigne.append(nomI)
	for k in range(0,i) :
		distanceLigne.append(distance[k][i+1])
	distanceLigne.append(0)	
	df1= df.loc[df['nom'] == str(nomI)]
	if not df1.empty:
		df1['date'] = pd.to_datetime( df1['date'],dayfirst =True)

		df1 = df1.set_index(df1['date'])
		df1 = df1.resample("M").sum().ffill()
		cpt = cpt.iloc[1:]
	for j, nomJ in enumerate(tqdm(cpt, desc ="loop progression")):
		df2= df.loc[df['nom'] == str(nomJ)]
		if not df2.empty:
			df2['date'] = pd.to_datetime( df2['date'],dayfirst =True)
			df2 = df2.set_index(df2['date'])
			df2 = df2.resample("M").sum().ffill()
			min_max_scaler = MinMaxScaler()
			df2['nbVente'] = min_max_scaler.fit_transform(df2['nbVente'].values.reshape(-1, 1))
			df1['nbVente'] = min_max_scaler.fit_transform(df1['nbVente'].values.reshape(-1, 1))

			dtw_score = dtw(df2['nbVente'], df1['nbVente']) ### Use This line for DTW version
			#dtw_score = soft_dtw(df2['nbVente'], df1['nbVente'],gamma) ### Use This line for SoftDTW version 
			distanceLigne.append(dtw_score)
	distance.append(distanceLigne)
	os.system('cls')
nomCat = nomCat.values.tolist()	
nomCat.insert(0, "nomProduit")
print(len(nomCat))
print(np.array(distance))
dfDistance = pd.DataFrame(np.array(distance),columns=nomCat)
print(dfDistance)
dfDistance.to_csv('DTW_Matrix.csv', sep=';', encoding='utf-8') ### Use This line for DTW version
#dfDistance.to_csv('DTW_MatrixSoft'+str(gamma)+'.csv', sep=';', encoding='utf-8') ### Use This line for SoftDTW version
