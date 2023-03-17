import copy
from collections import Counter

import matplotlib.cm as cm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

#citaj csv
data=pd.read_csv("fuel_consumption.csv")

#eliminisanje redova sa przanim val
data= data.dropna(thresh = len(data.columns))

#pisanje prvih 5
print(data.iloc[:5,:])

#moglo je i sa data.head(5)
#prikaz uz pomoc pandasa
print(data.info(),"\n\n",data.describe(),"\n\n",data.describe(include=object))

"""
#prikaz statistiki elemenata iz dataseta rucno bruh
print(Counter(data.MODELYEAR))
print(Counter(data.MAKE))
print(Counter(data.MODEL))
print(Counter(data.VEHICLECLASS))
print("Velicina motora: ",np.mean(data.ENGINESIZE))
print("Broj cilindara: ",np.mean(data.CYLINDERS))
print(Counter(data.TRANSMISSION))
print(Counter(data.FUELTYPE))
print("Potrosnja u gradu: ",np.mean(data.FUELCONSUMPTION_CITY))
print("Potrosnja goriva na otvorenom: ",np.mean(data.FUELCONSUMPTION_HWY))
print("Potrosnja kombinovano: ",np.mean(data.FUELCONSUMPTION_COMB))
print("Potrosnja goriva druga mjerna jedinica: ",np.mean(data.FUELCONSUMPTION_COMB_MPG))
print("Emisija gasova: ",np.mean(data.CO2EMISSIONS))
"""
"""
#korelaciona matrica ?
sns.heatmap(data.loc[:,["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_CITY","FUELCONSUMPTION_HWY","FUELCONSUMPTION_COMB","FUELCONSUMPTION_COMB_MPG","CO2EMISSIONS"]].corr(),annot=True,cmap='coolwarm',cbar=True)
plt.show()

#rasijavanje tacaka
for column in data.select_dtypes(include=['float','int']):
    sns.scatterplot(x=column, y='CO2EMISSIONS', data=data, hue='CO2EMISSIONS')
    plt.show()

for i in ["MODELYEAR","MAKE","MODEL","VEHICLECLASS","TRANSMISSION","FUELTYPE"]:
    grouped_data = data.groupby(i)['CO2EMISSIONS'].mean()
    grouped_data.plot(kind='bar', cmap='rainbow')
    plt.xlabel(i)

    plt.ylabel('CO2 Emissions')
    plt.show()
"""

"""
#moglo se prikazati i sa funkcijom hist
plt.figure()
data['MODEL'].hist()
plt.xticks(rotation=90)
plt.show()

"""

#df = pd.get_dummies(df, columns=['FUELTYPE'])
 #ovo je funkcija koja npr model pretvara u model_audi,model_bmw
#za lakse racunanje korelacije jelte bez da odbacimo text param


for i in ["MODELYEAR","MAKE","MODEL","VEHICLECLASS","TRANSMISSION","FUELTYPE"]:
    model_dict = {model:i for i, model in enumerate(data[i].unique())}
    data[i] = data[i].map(model_dict)

output_corr=data.corr()["CO2EMISSIONS"]
high_corr_attributes = output_corr[(output_corr >  0.8) | (output_corr < - 0.8)].keys()
data = data.loc[:, high_corr_attributes]


ulaz = data.iloc[:, :len(high_corr_attributes)-1]
izlaz = data.CO2EMISSIONS

#izlaz = np.reshape(izlaz, (izlaz.size, 1))



#splitovanje testova i treninga
ulazTrening, ulazTest, izlazTrening, izlazTest =train_test_split(ulaz,izlaz,random_state=64,test_size=0.2,shuffle=True)


class LinearRegressionGradientDescent:
    def __init__(self):
        self.coeff = None
        self.features = None
        self.target = None
        self.mse_history = None

    def set_coefficients(self, *args):
        self.coeff = np.array(args).reshape(-1, 1)

    def cost(self):
        predicted = self.features.dot(self.coeff)
        s = pow(predicted - self.target, 2).sum()
        return (0.5 / len(self.features)) * s

    # Argument mora biti DataFrame
    def predict(self, features):
        features = features.copy(deep=True)
        features.insert(0, 'c0', np.ones((len(features), 1)))
        features = features.to_numpy()
        return features.dot(self.coeff).reshape(-1, 1).flatten()

    # Jedan korak u algoritmu gradijentnog spusta.
    def gradient_descent_step(self, learning_rate):
        predicted = self.features.dot(self.coeff)

        s = self.features.T.dot(predicted - self.target)
        gradient = (1. / len(self.features)) * s
        self.coeff = self.coeff - learning_rate * gradient
        return self.coeff, self.cost()

    def perform_gradient_descent(self, learning_rate, num_iterations=100):
        self.mse_history = []
        for i in range(num_iterations):
            _, curr_cost = self.gradient_descent_step(learning_rate)
        self.mse_history.append(curr_cost)
        return self.coeff, self.mse_history

    # features mora biti DataFrame
    def fit(self, features, target):
        self.features = copy.deepcopy(features)

        coeff_shape = len(features.columns)+1
        self.coeff = np.zeros(shape=coeff_shape).reshape(-1, 1)

        self.features.insert(0, 'c0', np.ones((len(features), 1)))
        self.features = self.features.to_numpy()
        self.target = target.to_numpy().reshape(-1, 1)



# Kreiranje i obucavanje modela
lrgd = LinearRegressionGradientDescent()
lrgd.fit(ulazTrening, izlazTrening)

learning_rates = np.array(0.001)
res_coeff, mse_history = lrgd.perform_gradient_descent(learning_rates, 1000000)

predicted=lrgd.predict(ulazTest)
mse = mean_squared_error(izlazTest, predicted)
print("Mean Sqared Error mog modela je: ",mse)
r2 = r2_score(izlazTest, predicted)
print("Preciznost naseg modela je: ",r2*100,"%")

# Kreiranje i obucavanje sklearn.LinearRegression modela
reg = LinearRegression().fit(ulazTrening, izlazTrening)
predicted=reg.predict(ulazTest)
print("Tacnost ugradjenog modela je: ", r2_score(izlazTest,predicted)*100,"%")
mse = mean_squared_error(izlazTest, predicted)
print("Mean Squared Error ugradjenog modela je: ", mse)



