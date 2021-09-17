# IIT2019024 - Swaraj Bhosle

# In this code I have implemented locally weighted linear regression, feature Scaled and regularised batch gradient linear regression, feature Scaled and regularised Stochastic gradient linear regression and feature Scaled and regularised mini batch gradient linear regression. I have also compared the Mininum absulute percentage error of all of these.

import math
import numpy as np
import pandas as pd


def LocallyWeightedLR(X_matrx, xi, Y_matrx, Tau_hyperparameter):
    TransposeX_matrx = np.transpose(X_matrx)
    W = kernel(X_matrx, xi, Tau_hyperparameter)
    X_matrxTransposeW = TransposeX_matrx * W
    X_matrxTransposeWX = np.matmul(X_matrxTransposeW, X_matrx)
    InverseX_matrxTransposeWX = np.linalg.pinv(X_matrxTransposeWX)
    InverseX_matrxTransposeWXX_matrxTransposeW = np.matmul(InverseX_matrxTransposeWX, X_matrxTransposeW)
    InverseX_matrxTransposeWXX_matrxTransposeWY = np.matmul(InverseX_matrxTransposeWXX_matrxTransposeW, Y_matrx)
    InverseX_matrxTransposeWXX_matrxTransposeWYTranspose = np.transpose(InverseX_matrxTransposeWXX_matrxTransposeWY)
    return InverseX_matrxTransposeWXX_matrxTransposeWYTranspose.dot(xi)


def calculate_error(Y_matrx, pred_Y):
    error = 0
    for i in range(len(Y_matrx)):
        error += abs(Y_matrx[i] - pred_Y[i]) / Y_matrx[i]
    error = error / len(Y_matrx)
    return error * 100


def kernel(X_matrx, xi, Tau_hyperparameter):
    return np.exp(-np.sum((xi - X_matrx) ** 2, axis=1) / (2 * Tau_hyperparameter * Tau_hyperparameter))


input_data = pd.read_csv('Housing Price data set.csv', usecols=["price", "lotsize", "bedrooms", "bathrms"])
Area_of_floor = input_data['lotsize']
bedrooms_count = input_data['bedrooms']
bathrooms_count = input_data['bathrms']
Y_matrx = input_data['price']
Y_matrx = np.array(Y_matrx)
Y_matrx = Y_matrx.reshape(Y_matrx.shape[0], 1)

# Performing feature scanning on Area_of_floor
Area_of_floor_Mean = np.mean(Area_of_floor)
Area_of_floor_Max = max(Area_of_floor)
Area_of_floor_Min = min(Area_of_floor)
Area_of_floor_Scaled = []
for i in Area_of_floor:
    Area_of_floor_Scaled.append((i - Area_of_floor_Mean) / (Area_of_floor_Max - Area_of_floor_Min))

X_matrx = []
for i in range(len(Area_of_floor)):
    X_matrx.append([1, Area_of_floor_Scaled[i], bedrooms_count[i], bathrooms_count[i]])
X_matrx = np.array(X_matrx)

Tau_hyperparameter = 0.00005
print("Using Locally Weighted Linear Regression for Tau = " + str(Tau_hyperparameter))
pred = []
for i in range(X_matrx.shape[0]):
    pred_Y = LocallyWeightedLR(X_matrx, X_matrx[i], Y_matrx, Tau_hyperparameter)
    pred.append(pred_Y)
print("Mean absolute percentage error is : " + str(calculate_error(Y_matrx, pred)))
print()

Price = input_data['price']

# segmenting the features
Training_features = []
for i in range(383):
    Training_features.append([1, Area_of_floor_Scaled[i], bedrooms_count[i], bathrooms_count[i]])
Training_price = Price[:383]
PriceTest = []
FeaturesTest = []
for i in range(383, len(Price)):
    FeaturesTest.append([1, Area_of_floor_Scaled[i], bedrooms_count[i], bathrooms_count[i]])
    PriceTest.append(Price[i])
m = len(Training_features)


# Function to calculate Slope to find Coeffcnticients
def Slope(Coeffcnt, Training_features, Training_price, index):
    Err = 0
    for i in range(len(Training_features)):
        itrn = 0
        for j in range(len(Coeffcnt)):
            itrn = itrn + Coeffcnt[j] * Training_features[i][j]
        Err += (itrn - Training_price[i]) * Training_features[i][index]
    return Err


# Using scaled batch gradient with regularisation
print("Using scaled batch gradient with regularisation")
Rate_of_learning = 0.001
parameter_lambda = -49
Coeffcnt = [0, 0, 0, 0]
print("Initial Coeffcnticients: ")
print(Coeffcnt)
for itrs in range(5000):
    TempCoeffcnt = Coeffcnt.copy()
    for j in range(len(Coeffcnt)):
        if (j == 0):
            TempCoeffcnt[j] = TempCoeffcnt[j] - (
                        (Rate_of_learning / m) * (Slope(Coeffcnt, Training_features, Training_price, j)))
        else:
            TempCoeffcnt[j] = (1 - Rate_of_learning * parameter_lambda / m) * TempCoeffcnt[j] - (
                        (Rate_of_learning / m) * (Slope(Coeffcnt, Training_features, Training_price, j)))
    Coeffcnt = TempCoeffcnt.copy()
print("Final Coeffcnticients are:")
print(Coeffcnt)

# Finding Mean absolute percentage error.
Error = 0
for i in range(len(FeaturesTest)):
    predicted_val = 0
    for j in range(len(Coeffcnt)):
        predicted_val = predicted_val + Coeffcnt[j] * FeaturesTest[i][j]
    Error += abs(predicted_val - PriceTest[i]) / PriceTest[i]
Error = (Error / len(FeaturesTest)) * 100
print("Mean absolute percentage error is : " + str(Error))
print()


def SlopeStoch(Coeffcnt, Training_features, ActualVal, ind):
    itr = 0
    for j in range(len(Coeffcnt)):
        itr = itr + Coeffcnt[j] * Training_features[j]
    return (itr - ActualVal) * Training_features[ind]


# Using Scaled Stochastic gradient with regularisation.
print("Using Stochastic gradient with regularisation")

# I tried with different values of tau but found this to be the best.
Rate_of_learning = 0.004
parameter_lambda = 142000
Coeffcnt = [0, 0, 0, 0]
print("Initial Coeffcnticients: ")
print(Coeffcnt)

for iter in range(10):
    for i in range(len(Training_price)):
        TempCoeffcnt = Coeffcnt.copy()
        for j in range(4):
            if j == 0:
                TempCoeffcnt[j] = TempCoeffcnt[j] - (
                            Rate_of_learning * (SlopeStoch(Coeffcnt, Training_features[i], Training_price[i], j)))
            else:
                TempCoeffcnt[j] = (1 - Rate_of_learning * parameter_lambda / m) * TempCoeffcnt[j] - (
                            Rate_of_learning * (SlopeStoch(Coeffcnt, Training_features[i], Training_price[i], j)))
        Coeffcnt = TempCoeffcnt.copy()

print("Final Coeffecients are:")
print(Coeffcnt)

# Finding Mean absolute percentage error.
Error = 0
for i in range(len(FeaturesTest)):
    predicted = 0
    for j in range(len(Coeffcnt)):
        predicted = predicted + Coeffcnt[j] * FeaturesTest[i][j]
    Error += abs(predicted - PriceTest[i]) / PriceTest[i]
Error = (Error / len(FeaturesTest)) * 100
print("Mean absolute percentage error is : " + str(Error))
print()

# Using Scaled Minibatch gradient with regularisation for batch size = 20
print("Using Scaled Minibatch gradient with regularisation for batch size = 20")

Size_of_batch = 20;
Rate_of_learning = 0.002
parameter_lambda = -372
Coeffcnt = [0, 0, 0, 0]
batches_cou = math.ceil(len(Training_price) / Size_of_batch)
equallyDiv = False
if (len(Training_price) % Size_of_batch == 0):
    equallyDiv = True;

for itr in range(30):
    for batch in range(batches_cou):
        Summation = [0, 0, 0, 0]
        for j in range(len(Coeffcnt)):
            for i in range(Size_of_batch):
                if (batch * Size_of_batch + i == len(Training_features)):
                    break
                PredictedValue = 0.0
                for wj in range(len(Coeffcnt)):
                    PredictedValue += Coeffcnt[wj] * Training_features[batch * Size_of_batch + i][wj]
                PredictedValue -= Training_price[batch * Size_of_batch + i]
                PredictedValue *= Training_features[batch * Size_of_batch + i][j]
                Summation[j] += PredictedValue;

        if (not equallyDiv and batch == batches_cou - 1):
            for j in range(len(Summation)):
                if j == 0:
                    Coeffcnt[j] -= (Summation[j] / (len(Training_price) % Size_of_batch)) * Rate_of_learning
                else:
                    Coeffcnt[j] = (1 - Rate_of_learning * parameter_lambda / m) * Coeffcnt[j] - (
                                Summation[j] / (len(Training_price) % Size_of_batch)) * Rate_of_learning
        else:
            for j in range(len(Summation)):
                if j == 0:
                    Coeffcnt[j] -= (Summation[j] / Size_of_batch) * Rate_of_learning
                else:
                    Coeffcnt[j] = (1 - Rate_of_learning * parameter_lambda / m) * Coeffcnt[j] - (
                                Summation[j] / Size_of_batch) * Rate_of_learning
print("Final Coeffcnticients are:")
print(Coeffcnt)

# Finding Mean absolute percentage error.
Error = 0
for i in range(len(FeaturesTest)):
    val_predicted = 0
    for j in range(len(Coeffcnt)):
        val_predicted = val_predicted + Coeffcnt[j] * FeaturesTest[i][j]
    Error += abs(val_predicted - PriceTest[i]) / PriceTest[i]
Error = (Error / len(FeaturesTest)) * 100
print("Mean absolute percentage error is : " + str(Error))
print()