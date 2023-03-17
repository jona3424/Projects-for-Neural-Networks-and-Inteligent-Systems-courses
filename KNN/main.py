from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

pd.set_option("display.width",None)

#citaj csv
data=pd.read_csv("cakes.csv")
#pisanje prvih 5
print(data.iloc[:5,:])

#moglo je i sa data.head(5)
#prikaz uz pomoc pandasa
print(data.info(),"\n\n",data.describe(),"\n\n",data.describe(include=object))

"""

ulaz = data.iloc[:, :6].to_numpy()
izlaz = data.type.to_numpy()

#razdvajanje na klase
klase = np.unique(izlaz)
K1 = ulaz[izlaz == klase[0], :]
K2 = ulaz[izlaz == klase[1], :]




#prikaz statistiki elemenata iz dataseta
print("Brasno: " ,end=" ")
print(np.mean(data.flour))
print("Jaja: " ,end=" ")
print(np.mean(data.eggs))
print("Secer: " ,end=" ")
print(np.mean(data.sugar))
print("Mlijeko: " ,end=" ")
print(np.mean(data.milk))
print("Puter: " ,end=" ")
print(np.mean(data.butter))
print("Prasak: " ,end=" ")
print(np.mean(data.baking_powder))
print(str(len(K1))+" klasa koje su cupcake, " +str(len(K2))+" klasa koje su mafini")

"""
#eliminisanje redova sa przanim val
data= data.dropna()

data['eggs']*=63

sns.heatmap(data.iloc[:,:6].corr(),annot=True,cmap='coolwarm',cbar=True)
plt.show()

for column in data.select_dtypes(include=['float','int']):
    sns.scatterplot(x=column, y='type', data=data, hue='type')
    plt.show()

#  posto nemamo kategoricke atribute mozemo eventualno prikazati count klasa na ovom dijagramu za stavku broj  6
# for column in data.select_dtypes(include=['object']):
#     sns.countplot(x=column, hue='type', data=data)
#     plt.show()
# plt.show()


#odabir atributa tako sto biramo  jako korelisane atribute
"""
deepdata=pd.read_csv("cakes.csv")
deepdata.dropna()
deepdata['type'] =deepdata['type'].map({'cupcake':0, 'muffin':1})
corr_matrix = deepdata.corr()
output_corr = corr_matrix['type']
high_corr_attributes = output_corr[(output_corr >  0.4) | (output_corr < - 0.4)].keys()

data = data.loc[:, high_corr_attributes]

ulaz = data.iloc[:, :len(high_corr_attributes)-1].to_numpy()
"""

ulaz = data.iloc[:, :6].to_numpy()
izlaz = data.type.to_numpy()

#razdvajanje na klase

izlaz = np.reshape(izlaz, (izlaz.size, 1))

scaler = MinMaxScaler()
scaler.fit(ulaz)
# normalizuj podatke
normalized_ulaz = scaler.transform(ulaz)


#splitovanje testova i treninga
ulazTrening, ulazTest, izlazTrening, izlazTest =train_test_split(normalized_ulaz,izlaz,random_state=42,test_size=0.1,shuffle=True)



def Knnmodel(x_train, y_train, x_Test, k):
    distances = []
    for i in range(x_train.shape[0]):
        # menhent distance = np.abs(x_Test - x_train[i]).sum()
        # cebiseva distance=np.max(np.abs(x_Test - x_train[i]))
        distance = np.linalg.norm(x_Test - x_train[i])
        distances.append((distance, y_train[i][0]))
    distances = sorted(distances, key=lambda x: x[0])
    k_neighbors=[]
    for i in range(k):
        k_neighbors.append(distances[i][1])
    counts = Counter(k_neighbors)
    most_common = counts.most_common(1)
    return most_common[0][0]

def calculateK(arr):
    return round(np.sqrt(len(arr))) if round(np.sqrt(len(arr)))%2!=0 else round(np.sqrt(len(arr)))+1

retvals=[]
for i in range(ulazTest.shape[0]):
    retvals.append(Knnmodel(ulazTrening, izlazTrening, ulazTest[i], calculateK(ulazTrening)))
accuracy = accuracy_score(izlazTest, retvals)
print("Tacnost mog modela je: ",accuracy," za k: " ,calculateK(ulazTrening))

knn=KNeighborsClassifier(calculateK(ulazTrening),metric='euclidean')
knn.fit(ulazTrening,izlazTrening)
y_predict=knn.predict(ulazTest)
accuracy = accuracy_score(izlazTest,y_predict)

print("Tacnost ugradjenog modela je: ",accuracy," za k: " ,calculateK(ulazTrening))


