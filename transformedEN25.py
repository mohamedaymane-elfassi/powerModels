#######################
## IMPORTING MODULES ##
#######################
import numpy as np
import pandas as pd
import numpy
import seaborn as sns
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
import statsmodels.api as sm
import scipy.optimize as optimization
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from datetime import date
from datetime import datetime
from datetime import timedelta
import time
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from scipy import stats
import warnings
import math as mth
import holidays
#######################
#######################

#######################
## IGNORING WARNINGS ##
#######################
warnings.filterwarnings("ignore")
###################
###################

###################
## DEFINING FILE ##
###################
FILENAME = 'C:\\Users\\Aymane\\Documents\\Power markets\\Data\\EPEX_spot_DA_auction_hour_prices.csv'
###################
###################

###################
##  OPENING FILE ##
###################
file = open(FILENAME, 'r')
f = file.readlines()
###################
###################

###################
##  READING FILE ##
###################
list = []
for line in f:
        l = line.strip()
        list.append(l)
###################
###################

###################
##  CLOSING FILE ##
###################
file.close()
###################
###################

######################
##  REMOVING HEADER ##
######################
newList = []
for i in range(1,len(list)):
    newList.append(list[i].split(";"))
######################
######################

##########################
##  PREPARING DATAFRAME ##
##########################
dataset = pd.DataFrame(newList)
dataset = dataset.rename({0: 'DATE', 1: 'PRICE'}, axis='columns')
###########################
###########################

##################################
# PARSE STRINGS TO DATETIME TYPE #
##################################
dataset['DATE'] = pd.to_datetime(dataset['DATE'], infer_datetime_format=True)
dataset['HOUR'] = dataset['DATE'].dt.hour + 1
dataset['DOW'] = dataset['DATE'].dt.dayofweek
dataset['DAY'] = dataset['DATE'].dt.date
##################################
##################################

############################
# CONVERT COLUMNS TO FLOAT #
############################
dataset['PRICE'] = pd.to_numeric(dataset['PRICE'], downcast='float', errors='coerce')
#############################
#############################

############################################
# DIVIDING DATASETS INTO 24 DAILY DATASETS #
############################################
actualDates = pd.date_range(start=dataset['DATE'][0], end=dataset['DATE'][len(dataset['DATE'])-1], freq = 'D')
H = 3
dailyDataset = dataset.groupby('HOUR').get_group(H)
indexedDataset = dailyDataset.set_index(['DATE'])
index = pd.RangeIndex(0, len(indexedDataset))
dailyDataset = dailyDataset.set_index(index)
#############################################
#############################################

##########################
# PREPROCESSING OUTLIERS #
##########################
###################################
# COMPUTING 5 HOUR RUNNING MEDIAN #
###################################
medianPrices = dataset['PRICE'].rolling(window=5).mean()
demedianedPrices = dataset['PRICE'] - medianPrices
####################################
####################################

#############################
# CONSTRUCTING FILTER BANDS #
#############################
bandPlus = medianPrices + 3*demedianedPrices.std()
bandMinus = medianPrices - 3*demedianedPrices.std()
dataset['Bt +'] = bandPlus.fillna(dataset['PRICE'])
dataset['Bt -'] = bandMinus.fillna(dataset['PRICE'])
dataset['DEMED PRICES'] = demedianedPrices.fillna(dataset['PRICE'])
##############################
##############################

##############################################
# IDENTIFYING ALL OUTLIERS OUTSIDE THE BANDS #
##############################################
##############################################
# IDENTIFYING ALL OUTLIERS OUTSIDE THE BANDS #
##############################################
dataset['PRICE'][dataset['PRICE'] >= dataset['Bt +']] = dataset['Bt +']
dataset['PRICE'][dataset['PRICE'] <= dataset['Bt -']] = dataset['Bt -']
##############################################
##############################################

##############################################################
# DEALING WITH MISSING VALUES - SMOOTHED SIMILAR DAY METHOD  #
##############################################################
dataset['PRICE'] = dataset['PRICE'].fillna(0)
for i in range(0, len(dataset)):
    if dataset['PRICE'][i] == 0.0:
        doW = dataset['DOW'][i]
        if i <= 7 *24:
            for j in range(0, 7*24):
                dataset['PRICE'][i] = [dataset['PRICE'][i+7*24-j], dataset['PRICE'][i+7*24]][doW == 0 or doW == 5 or doW == 6]
        elif (i > 7*24)  and (i < len(dataset) - 7*24):
            for j in range(0, 7*24):
                doW = dataset['DOW'][i]
                dataset['PRICE'][i] = [(dataset['PRICE'][i-7*24-j] + dataset['PRICE'][i+7*24-j])/2, (dataset['PRICE'][i-7*24] + dataset['PRICE'][i+7*24])/2][doW == 0 or doW == 5 or doW == 6]
        else:
            for j in range(0, 7*24):
                dataset['PRICE'][i] = [dataset['PRICE'][i-7*24-j], dataset['PRICE'][i-7*24]][doW == 0 or doW == 5 or doW == 6]
###############################################################
###############################################################

#dataset.to_csv('C:\\Users\\Aymane\\Documents\\Power markets\\Books\\Weron - Modeling and Forecasting Electricity Loads and Prices' + '\\Chap4\\Implementation\\models\\C4-similarDay\\forecasts.csv')


######################################################
## TRANSFORMING PRICES : USING ASINH TRANSFORMATION ##
######################################################
initialCalibrationMedian = dataset['PRICE'][7*24:365*24+7*24].median()
initialCalibrationMAD = stats.median_absolute_deviation(dataset['PRICE'][7*24:365*24+7*24])*1.4826
dataset['MOD PRICE'] = (dataset['PRICE'] - initialCalibrationMedian)/initialCalibrationMAD
dataset['TRANS PRICE'] = np.log(np.array(dataset['MOD PRICE']) + np.sqrt(np.array(dataset['MOD PRICE']**2 + 1)))
####################################################
####################################################

#################################
## TWEAKING THE FINAL DATASETS ##
#################################
finalDataset = pd.DataFrame(dataset['PRICE'])
finalDataset.insert(1, "TRANS PRICE", dataset['TRANS PRICE'])
finalDataset.insert(2, "DATE", dataset['DATE'])
finalDataset.insert(3, "HOUR", dataset['HOUR'])
finalDataset.insert(4, "DOW", dataset['DOW'])
finalDataset.insert(5, "DAY", dataset['DAY'])
finalDataset = finalDataset.set_index(['DATE'])
finalDataset = finalDataset[0:1919*24]
#################################
#################################

################################################
## PLOTTING THE LOAD AND THE DIFFERENCED LOAD ##
################################################
rcParams['figure.figsize'] = 15, 7

plt.subplot(211)
plt.plot(finalDataset["PRICE"], label="Actual Price")
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(loc="best")
plt.title('Daily Price Profile for Hour ' + str(H))

plt.subplot(212)
plt.plot(finalDataset["TRANS PRICE"], label='Transformed Price', color = 'red')
plt.xlabel('Date')
plt.ylabel('¨Price')
plt.legend(loc="best")
###################################################
###################################################

##*******************************##
##-------------------------------##
##-------------------------------##
##        THE EN25 MODEL         ##
##-------------------------------##
##-------------------------------##
###########################################
## SPLITTING DATASET INTO TRAIN AND TEST ##
###########################################
Pd_j = [finalDataset[finalDataset['HOUR'] == j]['TRANS PRICE'].reset_index() for j in range(1, 25)]
hours = [timedelta(hours=j) for j in range(1, 25)]
PdMinus1_j = []
for j in range(0, 24):
    Pd_j[j]['DAY'] = (Pd_j[j]['DATE'] + hours[len(hours)-1-j]).dt.date
PdMinus1_j = [pd.merge(finalDataset['DAY'], Pd_j[j], on = 'DAY')['TRANS PRICE'] for j in range(0, 24)]
PdMinus1_j = [pd.concat([pd.Series([0]) for i in range(24)], ignore_index=True).append(PdMinus1_j[j], ignore_index = True) for j in range(0, 24)]
####################################
Pd_2j = [finalDataset[finalDataset['HOUR'] == j]['TRANS PRICE'].reset_index() for j in range(1, 25)]
hours2 = [timedelta(hours=j+24) for j in range(1, 25)]
PdMinus2_j = []
for j in range(0, 24):
    Pd_2j[j]['DAY'] = (Pd_2j[j]['DATE'] + hours2[len(hours2)-1-j]).dt.date
PdMinus2_j = [pd.merge(finalDataset['DAY'], Pd_2j[j], on = 'DAY')['TRANS PRICE'] for j in range(0, 24)]
PdMinus2_j = [pd.concat([pd.Series([0]) for i in range(48)], ignore_index=True).append(PdMinus2_j[j], ignore_index = True) for j in range(0, 24)]
####################################
####################################
Pd_3j = [finalDataset[finalDataset['HOUR'] == j]['TRANS PRICE'].reset_index() for j in range(1, 25)]
hours3 = [timedelta(hours=j+48) for j in range(1, 25)]
PdMinus3_j = []
for j in range(0, 24):
    Pd_3j[j]['DAY'] = (Pd_3j[j]['DATE'] + hours3[len(hours2)-1-j]).dt.date
PdMinus3_j = [pd.merge(finalDataset['DAY'], Pd_3j[j], on = 'DAY')['TRANS PRICE'] for j in range(0, 24)]
PdMinus3_j = [pd.concat([pd.Series([0]) for i in range(72)], ignore_index=True).append(PdMinus3_j[j], ignore_index = True) for j in range(0, 24)]
####################################
PdMinus1 = np.asarray(finalDataset['TRANS PRICE'].shift(24))
PdMinus7 = np.asarray(finalDataset['TRANS PRICE'].shift(7*24))
####################################
PdMin = [np.asarray(pd.merge(finalDataset['DAY'], finalDataset.groupby("DAY")["TRANS PRICE"].min().reset_index())['TRANS PRICE'].shift(j*24)) for j in range(1, 4)]
PdMax = [np.asarray(pd.merge(finalDataset['DAY'], finalDataset.groupby("DAY")["TRANS PRICE"].max().reset_index())['TRANS PRICE'].shift(j*24)) for j in range(1, 4)]
PdAvg = [np.asarray(pd.merge(finalDataset['DAY'], finalDataset.groupby("DAY")["TRANS PRICE"].mean().reset_index())['TRANS PRICE'].shift(j*24)) for j in range(1, 4)]
####################################
DoW = [np.asarray(finalDataset['DOW'] + 1 == j).astype(int) for j in range(1, 8)]
DoWPdMinus1 = [np.asarray(finalDataset['DOW'] + 1 == j).astype(int)*PdMinus1 for j in range(1, 8)]
####################################
PdMinus1_jStack = pd.DataFrame(np.column_stack(PdMinus1_j))
PdMinus2_jStack = pd.DataFrame(np.column_stack(PdMinus2_j))
PdMinus3_jStack = pd.DataFrame(np.column_stack(PdMinus3_j))
PdMinus7Stack = pd.DataFrame(PdMinus7)
PdMinStack = pd.DataFrame(np.column_stack(PdMin))
PdMaxStack = pd.DataFrame(np.column_stack(PdMax))
PdAvgStack = pd.DataFrame(np.column_stack(PdAvg))
DoWStack = pd.DataFrame(np.column_stack(DoW))
DoWPdMinus1Stack = pd.DataFrame(np.column_stack(DoWPdMinus1))
####################################
linearRegressionInputData = np.array([PdMinus1_jStack, PdMinus2_jStack, PdMinus3_jStack, PdMinus7Stack, PdMinStack, PdMaxStack, PdAvgStack, DoWStack, DoWPdMinus1Stack])
linearRegressionInputVariables = pd.DataFrame(np.column_stack(linearRegressionInputData))
####################################
####################################
trainLinearRegressionVariables = [np.asarray(linearRegressionInputVariables[7*24+91*24+j:91*24+365*24+7*24+j]) for j in range(0, 1456*24)]
cvLinearRegressionVariables = linearRegressionInputVariables[7*24:365*24+7*24]
testLinearRegressionVariables = np.asarray(linearRegressionInputVariables[91*24+365*24+7*24:len(finalDataset)])
cvTestLinearRegressionVariables = linearRegressionInputVariables[365*24+7*24:365*24+7*24 + 91*24]
####################################
trainTransPrice = [np.asarray(finalDataset[7*24+91*24+j:91*24+365*24+7*24+j]['TRANS PRICE']) for j in range(0, 1456*24)]
cvTransPrice = finalDataset[7*24:365*24+7*24]['TRANS PRICE']
cvPrice = finalDataset[365*24+7*24:365*24+7*24 + 91*24]['PRICE']
####################################
testDataset = finalDataset[365*24+91*24+7*24:len(finalDataset)]
forecastDataset = pd.Series(np.zeros(len(testDataset)))
####################################
forecastDataframe = pd.DataFrame(testDataset)
forecastDataframe.insert(1, "FCST PRICE", forecastDataset)
###########################################
###########################################

####################################
##      APPLYING THE EN25 MODEL   ##
####################################
l1_ratio = 0.25
alphas = np.logspace(0, -6, 25)
cvelasticNetModel = [linear_model.ElasticNet(alpha = alphas[i], l1_ratio = l1_ratio)  for i in range(0, len(alphas))]
cvelasticNetFit = [cvelasticNetModel[i].fit(cvLinearRegressionVariables, cvTransPrice) for i in range(0, len(alphas))]
cvForecastPrices = [cvelasticNetFit[i].predict(cvTestLinearRegressionVariables) for i in range(0, len(alphas))]
#####################################

#############################
## RETRIEVING ACTUAL PRICE ##
#############################
forecastData = [np.sinh(cvForecastPrices[i])*initialCalibrationMAD + initialCalibrationMedian for i in range(0, len(alphas))]
############################
############################

###############################
## AFTER SELECTING MIN ALPHA ##
###############################
alphaMin = alphas[10]
elasticNetModel = [linear_model.ElasticNet(alpha = alphaMin, l1_ratio = l1_ratio) for j in range(0, 1456*24)]
elasticNetFit = [elasticNetModel[j].fit(trainLinearRegressionVariables[j], trainTransPrice[j]) for j in range(0, 1456*24)]
forecastPrices = [elasticNetFit[j].predict(testLinearRegressionVariables[j:j+1]) for j in range(0, 1456*24)]
#####################################
#####################################
coeffOccurence = [[(elasticNetFit[k+j*24].coef_ != 0).astype(int) for j in range(0, 1456)] for k in range(0,24)]
coeffOccurences = np.column_stack([pd.DataFrame(coeffOccurence[k]).sum(axis = 0) for k in range(0, 24)])
plt.subplots(figsize=(20,80))
sns_plot = sns.heatmap(coeffOccurences, annot = True, fmt="", cmap='RdYlGn', linewidths=1.5)
sns_plot.figure.savefig('C:\\Users\\Aymane\\Documents\\Power markets\\Books\\Weron - Modeling and Forecasting Electricity Loads and Prices\\Chap4\\Implementation\\models\\C4-elasticNet\\c_occ25.png')
#####################################
#####################################

############################
## RETRIEVING ACTUAL LOAD ##
############################
forecastDataframe['FCST PRICE'] = np.sinh(forecastPrices)*initialCalibrationMAD + initialCalibrationMedian
############################
############################

####################################
## DEFINING FUNCTION COMPUTE MAPE ##
####################################
def computeMAPE(originalSignal, forecastedSignal):
    originalArray = np.array(originalSignal)
    forecastedArray = np.array(forecastedSignal)
    percentageErrorsArray = np.abs((originalArray - forecastedArray)/originalArray)

    return 1/len(percentageErrorsArray)*percentageErrorsArray.sum()
######################################
######################################

###################################
## DEFINING FUNCTION COMPUTE MAE ##
###################################
def computeMAE(originalSignal, forecastedSignal):
    originalArray = np.array(originalSignal)
    forecastedArray = np.array(forecastedSignal)
    errorsArray = np.abs(originalArray - forecastedArray)

    return 1/len(errorsArray)*errorsArray.sum()
######################################
######################################

####################################
## DEFINING FUNCTION COMPUTE RMSE ##
####################################
def computeRMSE(originalSignal, forecastedSignal):
    originalArray = np.array(originalSignal)
    forecastedArray = np.array(forecastedSignal)
    squareErrorsArray = (originalArray - forecastedArray)**2

    return mth.sqrt(1/len(squareErrorsArray)*squareErrorsArray.sum())
######################################
######################################

##---------------##
## FINAL RESULTS ##
##---------------##
#######################
## COMPUTING METRICS ##
#######################
MAPE = np.asarray([computeMAPE(cvPrice, forecastData[i]) for i in range(0, len(alphas))])
MAE = [computeMAE(cvPrice, forecastData[i]) for i in range(0, len(alphas))]
RMSE = [computeRMSE(cvPrice, forecastData[i]) for i in range(0, len(alphas))]
alphaIndexMin = np.argmin(MAPE)
#######################
#######################
forecastDataframe = forecastDataframe.dropna()
MAPE = computeMAPE(forecastDataframe['PRICE'],forecastDataframe['FCST PRICE'])
MAE = computeMAE(forecastDataframe['PRICE'],forecastDataframe['FCST PRICE'])
RMSE = computeRMSE(forecastDataframe['PRICE'],forecastDataframe['FCST PRICE'])
##-------------------------------##
##-------------------------------##
##-------------------------------##
##-------------------------------##
##*******************************##
