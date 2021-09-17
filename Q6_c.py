# IIT2019024 - Swaraj Bhosle

# In this code I have demonstrated the difference between a regularised hypothesis and a non regularised hypothesis using Batch GDA, Stochastic GDA and Minibatch GDA.

import numpy as np
import math
import pandas as pd


input_data =pd.read_csv('https://raw.githubusercontent.com/Cipher-vasu/Databases_for_ML/main/Housing%20Price%20data%20set.csv')


Price = input_data['price']
bedrooms_count = input_data['bedrooms']
bathrooms_count = input_data['bathrms']
Area_of_floor = input_data['lotsize']


# Performing feature scanning on Area_of_floor
Area_of_floor_Max = max(Area_of_floor)
Area_of_floor_Min = min(Area_of_floor)
Area_of_floor_Mean = np.mean(Area_of_floor)

Area_of_floor_Scaled = []     # taking empty list of scaled area of floor
for i in Area_of_floor:
	Area_of_floor_Scaled.append((i - Area_of_floor_Mean) / (Area_of_floor_Max - Area_of_floor_Min))

#segmenting the features
Training_Features = []
for i in range(383):
	Training_Features.append([1, Area_of_floor_Scaled[i], bedrooms_count[i], bathrooms_count[i]])
Training_Price = Price[:383]   # sliced the data from price to training_price list
Price_test = []
Features_test = []
for i in range(383, len(Price)):
	Features_test.append([1, Area_of_floor_Scaled[i], bedrooms_count[i], bathrooms_count[i]])
	Price_test.append(Price[i])
m = len(Training_Features)

# Function to calculate Slope to find the coefficients
def Slope(Coeffcnt, Training_Features, Training_Price, ind):
	Err = 0
	for i in range(len(Training_Features)):
		itr = 0
		for j in range(len(Coeffcnt)):
			itr = itr + Coeffcnt[j] * Training_Features[i][j]
		Err += (itr - Training_Price[i]) * Training_Features[i][ind]
	return Err

# Using scaled batch gradient without regularisation
print("Using scaled batch gradient without regularisation")
Rate_of_learning = 0.001
m = len(Training_Features)

Coeffcnt = [0, 0, 0, 0]
print("Initial coefficients: ")
print(Coeffcnt)
for i in range(5000):
	TempeCoeff = Coeffcnt.copy()
	for j in range(len(Coeffcnt)):
		TempCoeff[j] = TempCoeff[j] - ((Rate_of_learning / m) * (Slope(Coeffcnt, Training_Features, Training_Price, j)))
	Coeffcnt = TempCoeff.copy()
print("Final coefficients are:")
print(Coeffcnt)

# Finding Mean absolute percentage error.
Error = 0
for i in range(len(Features_test)):
	val_predicted = 0
	for j in range(len(Coeffcnt)):
	  	val_predicted = val_predicted + Coeffcnt[j] * Features_test[i][j]
	Error += abs(val_predicted - Price_test[i]) / Price_test[i]
Error = (Error / len(Features_test)) * 90
print("Mean absolute percentage error is : " + str(Error))
print()

# Using scaled batch gradient with regularisation
print("Using scaled batch gradient with regularisation")
Rate_of_learning = 0.001
LambdaParameter = -49
Coeffcnt = [0, 0, 0, 0]
print("Initial coefficients: ")
print(Coeffcnt)
for itr in range(5000):
	TempCoeff = Coeffcnt.copy()
	for j in range(len(Coeffcnt)):
		if (j == 0):
			TempCoeff[j] = TempCoeff[j] - ((Rate_of_learning / m) * (Slope(Coeffcnt, Training_Features, Training_Price, j)))
		else:
			TempCoeff[j] = (1 - Rate_of_learning * LambdaParameter / m) * TempCoeff[j] - ((Rate_of_learning / m) * (Slope(Coeffcnt, Training_Features, Training_Price, j)))
	Coeffcnt = TempCoeff.copy()
print("Final coefficients are:")
print(Coeffcnt)

# Finding Mean absolute percentage error.
Error = 0
for i in range(len(Features_test)):
	val_predicted = 0
	for j in range(len(Coeffcnt)):
	  	val_predicted = val_predicted + Coeffcnt[j] * Features_test[i][j]
	Error += abs(val_predicted - Price_test[i]) / Price_test[i]
Error = (Error / len(Features_test)) * 100
print("Mean absolute percentage error is : " + str(Error))
print()

def SlopeStoch(Coeffcnt,Training_Features,ActualVal,ind):
	itrn = 0
	for j in range(len(Coeffcnt)):
		itrn = itrn + Coeffcnt[j]*Training_Features[j]
	return (itrn - ActualVal) * Training_Features[ind]

# Using Scaled Stochastic gradient without regularisation.
print("Using Stochastic gradient without regularisation")

Rate_of_learning = 0.005
Coeffcnt = [0, 0, 0, 0]
print("Initial coefficients: ")
print(Coeffcnt)

for iter in range(10):
	for i in range(len(Training_Price)):
		TempCoeff = Coeffcnt.copy()
		for j in range(4):
			TempCoeff[j] = TempCoeff[j] - (Rate_of_learning * (SlopeStoch(Coeffcnt, Training_Features[i], Training_Price[i], j)))
		Coeffcnt = TempCoeff.copy()

print("Final coefficients are:")
print(Coeffcnt)

# Finding Mean absolute percentage error.
Error = 0
for i in range(len(Features_test)):
	val_predicted = 0
	for j in range(len(Coeffcnt)):
	  	val_predicted = val_predicted + Coeffcnt[j] * Features_test[i][j]
	Error += abs(val_predicted - Price_test[i]) / Price_test[i]
Error = (Error / len(Features_test)) * 100
print("Mean absolute percentage error is : " + str(Error))
print()

# Using Scaled Stochastic gradient with regularisation.
print("Using Stochastic gradient with regularisation")

Rate_of_learning = 0.005
LambdaParameter = 142000
Coeffcnt = [0, 0, 0, 0]
print("Initial coefficients: ")
print(Coeffcnt)

for iter in range(10):
	for i in range(len(Training_Price)):
		TempCoeff = Coeffcnt.copy()
		for j in range(4):
			if j == 0:
				TempCoeff[j] = TempCoeff[j] - (Rate_of_learning * (SlopeStoch(Coeffcnt, Training_Features[i], Training_Price[i], j)))
			else:
				TempCoeff[j] = (1 - Rate_of_learning * LambdaParameter) * TempCoeff[j] - (Rate_of_learning * (SlopeStoch(Coeffcnt, Training_Features[i], Training_Price[i], j)))
		Coeffcnt = TempCoeff.copy()

print("Final coefficients are:")
print(Coeffcnt)

# Finding Mean absolute percentage error.
Error = 0
for i in range(len(Features_test)):
	val_predicted = 0
	for j in range(len(Coeffcnt)):
	  	val_predicted = val_predicted + Coeffcnt[j] * Features_test[i][j]
	Error += abs(val_predicted - Price_test[i]) / Price_test[i]
Error = (Error / len(Features_test)) * 100
print("Mean absolute percentage error is : " + str(Error))
print()

# Using Scaled Minibatch gradient without regularisation for batch size = 20
print("Using Scaled Minibatch gradient without regularisation for batch size = 20")

size_of_batch = 20;
Rate_of_learning = 0.002
Coeffcnt = [0, 0, 0, 0]
batches_count = math.ceil(len(Training_Price) / size_of_batch)
equallyDiv = False
if (len(Training_Price) % size_of_batch == 0):
	equallyDiv = True;

for itr in range(30):
	for batch in range(batches_count):
		Summation = [0, 0, 0, 0]
		for j in range(len(Coeffcnt)):
			for i in range(size_of_batch):
				if (batch * size_of_batch + i == len(Training_Features)):
					break
				val_predictedValue = 0.0
				for pj in range(len(Coeffcnt)):
					val_predictedValue += Coeffcnt[pj] * Training_Features[batch * size_of_batch + i][pj]
				val_predictedValue -= Training_Price[batch * size_of_batch + i]
				val_predictedValue *= Training_Features[batch * size_of_batch + i][j]
				Summation[j] += val_predictedValue;

		if (not equallyDiv and batch == batches_count - 1):
			for j in range(len(Summation)):
				Coeffcnt[j] -= (Summation[j] / (len(Training_Price) % size_of_batch)) * Rate_of_learning
		else:
			for j in range(len(Summation)):
				Coeffcnt[j] -= (Summation[j] / size_of_batch) * Rate_of_learning
print("Final coefficients are:")
print(Coeffcnt)

# Finding Mean absolute percentage error.
Error = 0
for i in range(len(Features_test)):
	val_predicted = 0
	for j in range(len(Coeffcnt)):
	  	val_predicted = val_predicted + Coeffcnt[j] * Features_test[i][j]
	Error += abs(val_predicted - Price_test[i]) / Price_test[i]
Error = (Error / len(Features_test)) * 100
print("Mean absolute percentage error is : " + str(Error))
print()

# Using Scaled Minibatch gradient with regularisation for batch size = 20
print("Using Scaled Minibatch gradient with regularisation for batch size = 20")

size_of_batch = 20;
Rate_of_learning = 0.002
LambdaParameter = -372
Coeffcnt = [0, 0, 0, 0]
batches_count = math.ceil(len(Training_Price) / size_of_batch)
equallyDiv = False
if (len(Training_Price) % size_of_batch == 0):
	equallyDiv = True;

for epoch in range(30):
	for batch in range(batches_count):
		Summation = [0, 0, 0, 0]
		for j in range(len(Coeffcnt)):
			for i in range(size_of_batch):
				if (batch * size_of_batch + i == len(Training_Features)):
					break
				val_predictedValue = 0.0
				for pj in range(len(Coeffcnt)):
					val_predictedValue += Coeffcnt[pj] * Training_Features[batch * size_of_batch + i][pj]
				val_predictedValue -= Training_Price[batch * size_of_batch + i]
				val_predictedValue *= Training_Features[batch * size_of_batch + i][j]
				Summation[j] += val_predictedValue;

		if (not equallyDiv and batch == batches_count - 1):
			for j in range(len(Summation)):
				if j == 0:
					Coeffcnt[j] -= (Summation[j] / (len(Training_Price) % size_of_batch)) * Rate_of_learning
				else:
					Coeffcnt[j] = (1 - Rate_of_learning * LambdaParameter / m) * Coeffcnt[j] - (Summation[j] / (len(Training_Price) % size_of_batch)) * Rate_of_learning
		else:
			for j in range(len(Summation)):
				if j == 0:
					Coeffcnt[j] -= (Summation[j] / size_of_batch) * Rate_of_learning
				else:
					Coeffcnt[j] = (1 - Rate_of_learning * LambdaParameter / m) * Coeffcnt[j] - (Summation[j] / size_of_batch) * Rate_of_learning
print("Final coefficients are:")
print(Coeffcnt)

# Finding Mean absolute percentage error.
Error = 0
for i in range(len(Features_test)):
	val_predicted = 0
	for j in range(len(Coeffcnt)):
	  	val_predicted = val_predicted + Coeffcnt[j] * Features_test[i][j]
	Error += abs(val_predicted - Price_test[i]) / Price_test[i]
Error = (Error / len(Features_test)) * 100
print("Mean absolute percentage error is : " + str(Error))
print()