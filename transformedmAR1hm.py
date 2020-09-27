#######################
## IMPORTING MODULES ##
#######################
import numpy as np
import pandas as pd
import numpy
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
from scipy import stats
import warnings
import math as mth
import holidays
import statsmodels.api as sm
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
medianPrices = dataset['PRICE'].rolling(window=5*24).mean()
demedianedPrices = dataset['PRICE'] - medianPrices
####################################
####################################

#############################
# CONSTRUCTING FILTER BANDS #
#############################
bandPlus = pd.Series([medianPrices[i] + 3*demedianedPrices.std() for i in range(0, len(dataset))])
bandMinus = pd.Series([medianPrices[i] - 3*demedianedPrices.std() for i in range(0, len(dataset))])
dataset['Bt +'] = bandPlus
dataset['Bt -'] = bandMinus

##############################################
# IDENTIFYING ALL OUTLIERS OUTSIDE THE BANDS #
##############################################
##############################################
# IDENTIFYING ALL OUTLIERS OUTSIDE THE BANDS #
##############################################
for i in range(0, len(dataset)):
    if dataset['PRICE'][i] > bandPlus[i]:
        dataset['PRICE'][i] = bandPlus[i]
    if dataset['PRICE'][i] < bandMinus[i]:
        dataset['PRICE'][i] = bandMinus[i]
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

#######################################################
# DEALING WITH SEASONALITY : USING SEASONAL DECOMPOSE #
#######################################################
weeklySeaonalLoad = sm.tsa.seasonal_decompose(dataset['PRICE'], freq = 7*24)
annualySeasonalLoad= sm.tsa.seasonal_decompose(dataset['PRICE'], freq=365*24)
dataset['SEAS PRICE'] = weeklySeaonalLoad.seasonal + annualySeasonalLoad.seasonal
dataset['DESEAS PRICE'] = dataset['PRICE'] - dataset['SEAS PRICE']

######################################################
## TRANSFORMING PRICES : USING ASINH TRANSFORMATION ##
######################################################
initialCalibrationMedian = dataset['DESEAS PRICE'][7*24:365*24+7*24].median()
initialCalibrationMAD = stats.median_absolute_deviation(dataset['DESEAS PRICE'][7*24:365*24+7*24])*1.4826
dataset['MOD PRICE'] = (dataset['DESEAS PRICE'] - initialCalibrationMedian)/initialCalibrationMAD
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
## THE SIMILAR DAY (NAIVE MODEL) ##
##-------------------------------##
##-------------------------------##
###########################################
## SPLITTING DATASET INTO TRAIN AND TEST ##
###########################################
D0 = np.asarray(finalDataset['DOW'] == finalDataset['DOW']).astype(int)
DsatPdMinus1 = np.asarray(finalDataset['DOW'] + 1 == 6).astype(int)*np.asarray(finalDataset['TRANS PRICE'].shift(24))
DsunPdMinus1 = np.asarray(finalDataset['DOW'] + 1 == 7).astype(int)*np.asarray(finalDataset['TRANS PRICE'].shift(24))
DmonPdMinus1 = np.asarray(finalDataset['DOW'] + 1 == 1).astype(int)*np.asarray(finalDataset['TRANS PRICE'].shift(24))
####################################
PdMinus1 = np.asarray(finalDataset['TRANS PRICE'].shift(24))
PdMinus2 = np.asarray(finalDataset['TRANS PRICE'].shift(2*24))
PdMinus7 = np.asarray(finalDataset['TRANS PRICE'].shift(7*24))
PdMin = np.asarray(pd.merge(finalDataset['DAY'], finalDataset.groupby("DAY")["TRANS PRICE"].min().reset_index())['TRANS PRICE'].shift(24))
Dsat = np.asarray(finalDataset['DOW'] + 1 == 6).astype(int)
Dsun = np.asarray(finalDataset['DOW'] + 1 == 7).astype(int)
Dmon = np.asarray(finalDataset['DOW'] + 1 == 1).astype(int)
Dhol = np.asarray([finalDataset['DAY'][j] in holidays.DE() for j in range(0, len(finalDataset['DAY']))]).astype(int)
####################################
Pd_24 = finalDataset[finalDataset['HOUR'] == 24]['TRANS PRICE'].reset_index()
oneHour = timedelta(hours=1)
Pd_24['DAY'] = (Pd_24['DATE'] + oneHour).dt.date
PdMinus1_24 = pd.merge(finalDataset['DAY'], Pd_24, on = 'DAY')['TRANS PRICE']
PdMinus1_24 = pd.concat([pd.Series([0]) for i in range(24)], ignore_index=True).append(PdMinus1_24, ignore_index = True)
####################################
linearRegressionInputData = np.array([D0, DsatPdMinus1, DsunPdMinus1, DmonPdMinus1, PdMinus1, PdMinus2, PdMinus7, PdMin, Dsat, Dsun, Dmon, Dhol, PdMinus1_24])
linearRegressionInputVariables = pd.DataFrame(np.column_stack(linearRegressionInputData))
####################################
####################################
trainLinearRegressionVariables = [np.asarray(linearRegressionInputVariables[7*24+91*24+j:91*24+365*24+7*24+j]) for j in range(0, 1456*24)]
testLinearRegressionVariables = np.asarray(linearRegressionInputVariables[91*24+365*24+7*24:len(finalDataset)])
####################################
trainTransPrice = [np.asarray(finalDataset[7*24+91*24+j:91*24+365*24+7*24+j]['TRANS PRICE']) for j in range(0, 1456*24)]
####################################
testDataset = finalDataset[365*24+91*24+7*24:len(finalDataset)]
testSeasDataset = dataset['SEAS PRICE'][365*24+91*24+7*24:len(finalDataset)]
forecastDataset = pd.Series(np.zeros(len(testDataset)))
####################################
forecastDataframe = pd.DataFrame(testDataset)
forecastDataframe.insert(1, "FCST PRICE", forecastDataset)
###########################################
###########################################


####################################
## APPLYING THE SIMILAR DAY MODEL ##
####################################
####################################
mar1Model = [linear_model.LinearRegression() for j in range(0, 1456*24)]
mar1Fit = [mar1Model[j].fit(trainLinearRegressionVariables[j], trainTransPrice[j]) for j in range(0, 1456*24)]
forecastPrices = [mar1Fit[j].predict(testLinearRegressionVariables[j:j+1]) for j in range(0, 1456*24)]
#####################################
#####################################

############################
## RETRIEVING ACTUAL LOAD ##
############################
forecastDataframe['FCST PRICE'] = np.sinh(forecastPrices)*initialCalibrationMAD + initialCalibrationMedian
forecastDataframe = forecastDataframe.reset_index(drop=True)
forecastDataframe['FCST PRICE'] = forecastDataframe['FCST PRICE'] + testSeasDataset.reset_index(drop = True)
############################
############################

####################################
## DEFINING FUNCTION COMPUTE MAPE ##
####################################
def computeMAPE(forecastDataframe):
    percentageErrorsArray = np.array([])
    originalArray = np.array(forecastDataframe['PRICE'])
    forecastedArray = np.array(forecastDataframe['FCST PRICE'])
    percentageErrorsArray = np.abs((originalArray - forecastedArray)/originalArray)

    return 1/len(percentageErrorsArray)*percentageErrorsArray.sum()
######################################
######################################

###################################
## DEFINING FUNCTION COMPUTE MAE ##
###################################
def computeMAE(forecastDataframe):
    errorsArray = np.array([])
    originalArray = np.array(forecastDataframe['PRICE'])
    forecastedArray = np.array(forecastDataframe['FCST PRICE'])
    errorsArray = np.abs(originalArray - forecastedArray)

    return 1/len(errorsArray)*errorsArray.sum()
######################################
######################################

####################################
## DEFINING FUNCTION COMPUTE RMSE ##
####################################
def computeRMSE(forecastDataframe):
    squareErrorsArray = np.array([])
    originalArray = np.array(forecastDataframe['PRICE'])
    forecastedArray = np.array(forecastDataframe['FCST PRICE'])
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
forecastDataframe = forecastDataframe.dropna()
MAPE = computeMAPE(forecastDataframe)
MAE = computeMAE(forecastDataframe)
RMSE = computeRMSE(forecastDataframe)
##-------------------------------##
##-------------------------------##
##-------------------------------##
##-------------------------------##
##*******************************##
