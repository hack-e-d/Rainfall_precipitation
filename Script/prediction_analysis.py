# importing libraries 
import pandas as pd 
import numpy as np 
import sklearn as sk 
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt 

# read the cleaned data 
data = pd.read_csv("austin_final.csv") 

# the features or the 'x' values of the data 
# these columns are used to train the model 
# the last column, i.e, precipitation column 
# will serve as the label 
X = data.drop(['PrecipitationSumInches'], axis = 1) 

# the output or the label. 
Y = data['PrecipitationSumInches'] 
# reshaping it into a 2-D vector 
Y = Y.values.reshape(-1, 1) 

# consider a random day in the dataset 
# we shall plot a graph and observe this 
# day 
day_index = 798
days = [i for i in range(Y.size)] 

# initialize a linear regression classifier 
clf = LinearRegression() 
# train the classifier with our 
# input data. 
clf.fit(X, Y) 

# plot a graph of the precipitation levels 
# versus the total number of days. 
# one day, which is in red, is 
# tracked here. It has a precipitation 
# of approx. 2 inches. 
print("the precipitation trend graph: ") 
plt.scatter(days, Y, color = 'g') 
plt.scatter(days[day_index], Y[day_index], color ='r') 
plt.title("Precipitation level") 
plt.xlabel("Days") 
plt.ylabel("Precipitation in inches") 


plt.show() 
inpt=[]
inpt=input("enter the input").split()
for i in range(len(inpt)):
    inpt[i]=float(inpt[i])
print(inpt)
inpt=np.array(inpt)
inpt = inpt.reshape(1, -1) 
print('The precipitation in inches for the input is:', clf.predict(inpt)) 

