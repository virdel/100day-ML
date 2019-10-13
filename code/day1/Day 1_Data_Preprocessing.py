import numpy as np
import pandas as pd
dataset=pd.read_csv("Data.csv")#读取csv文件为dataframe类型
x=dataset.iloc[:,:-1].values#所以返回的是series类型，values返回ndarray
y=dataset.iloc[:,-1].values

#处理丢失数据
from sklearn.preprocessing import Imputer
#在版本0.20之前，请使用Imputer 类，版本0.22(包含0.22)之后的，则使用SimpleImputer
#https://sklearn.apachecn.org/docs/0.21.3/41.html
imputer=Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer=imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])

#解析分类数据
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#将x第一列编码，形式为0，1，2，3，4
labelencoder_x=LabelEncoder()
x[:,0]=labelencoder_x.fit_transform(x[:,0])
#将x转化为onehot 编码
onehotencoder=OneHotEncoder(categorical_features=[0])
#x[:,0]=onehotencoder.fit_transform(x[:,0]).toarray()
x=onehotencoder.fit_transform(x).toarray()

labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)

#拆分数据集
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#特征量化(归一化）
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)
print(x_train)
print(x_test)



