import pandas as pd
import pickle
from KNN import *
from Random_forest import *
from Logistic_regression import *
from XGB import *
data = pd.read_pickle('content/HistoryPoolBorrowersStats.pkl')
df = pd.DataFrame.from_dict(data,orient='index')
chemin_fichier_csv = 'newStats.csv'
df.to_csv(chemin_fichier_csv, index=True)
df = pd.read_csv('newStats.csv')
columns_to_remove = ['datesSupplyCollateral', 'datesLoan','datesReimbursement','datesWithdrawCollateral','timeUTCFirstAnyTransactionAccount','borrowerAgeInYears','totalLoans','NumLoans','MeanLoans','totalTimeloans','MeanTimeLoans','totalSupplyCollateral','NumSupplyCollateral','MeanSupplyCollateral','totalReimbursements','NumReimbursements','MeanReimbursements','NumTransactions']  # Specify the names of the columns to remove
df = df.drop(columns=columns_to_remove)
df.to_csv('newStats.csv', index=False)


import pandas as pd
import json
from datetime import datetime
import time

df = pd.read_csv('newStats.csv')



def date_to_timestamp(date_str):
    dt_object = datetime.strptime(date_str, "%Y-%m-%d")
    timestamp = int(time.mktime(dt_object.timetuple()))
    return timestamp


for i, cell in enumerate(df['Calendar']):
    if isinstance(cell, str):
        cell = cell.replace("'", '"')
        calendar_dict = json.loads(cell)
        new_calendar_dict = {}
        for date_str, day_dict in calendar_dict.items():
            timestamp = date_to_timestamp(date_str)
            if day_dict['sizeDebtUSD'] != 0:
                ratio = day_dict['sizeCollateralUSD'] / day_dict['sizeDebtUSD']
            else:
                ratio = float('inf')
            day_dict['collateralDebtRatio'] = ratio

            new_calendar_dict[timestamp] = day_dict
        df.at[i, 'Calendar'] = json.dumps(new_calendar_dict)
    else:
        continue
df.to_csv('newStats.csv', index=False)

import pandas as pd
from datetime import datetime

def timestamp_to_category(timestamp):
    dt_object = datetime.fromtimestamp(int(float(timestamp)))
    hour = dt_object.hour
    if 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 18:
        return "Afternoon"
    elif 18 <= hour < 24:
        return "Evening"
    else:
        return "Night"

def most_frequent_category(lst):
    return max(set(lst), key=lst.count) if lst else None

df = pd.read_csv('newStats.csv')
timestamp_cols = ['timeStampsSupplyCollateral', 'timeStampsLoans', 'timeStampsReimbursement', 'timeStampsWithdrawCollateral', 'timeStampFirstAnyTransactionAccount']
df['MostFrequentTransactionTimeOfDay'] = ""
for i, row in df.iterrows():
    categories = []
    for col in timestamp_cols:
        if pd.isnull(row[col]):
            continue
        timestamps = str(row[col]).split()
        for timestamp in timestamps:
            categories.append(timestamp_to_category(timestamp))
    df.at[i, 'MostFrequentTransactionTimeOfDay'] = most_frequent_category(categories)
df.to_csv('newStats.csv', index=False)

import pandas as pd
from datetime import datetime

def timestamp_to_category(timestamp):
    dt_object = datetime.fromtimestamp(int(float(timestamp)))
    hour = dt_object.hour
    if 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 18:
        return "Afternoon"
    elif 18 <= hour < 24:
        return "Evening"
    else:
        return "Night"

def most_frequent_category(lst):
    return max(set(lst), key=lst.count) if lst else None

df = pd.read_csv('newStats.csv')
timestamp_cols = ['timeStampsSupplyCollateral', 'timeStampsLoans', 'timeStampsReimbursement', 'timeStampsWithdrawCollateral', 'timeStampFirstAnyTransactionAccount']
df['MostFrequentTransactionTimeOfDay'] = ""
for i, row in df.iterrows():
    categories = []
    for col in timestamp_cols:
        if pd.isnull(row[col]):
            continue
        timestamps = str(row[col]).split()
        for timestamp in timestamps:
            categories.append(timestamp_to_category(timestamp))
    df.at[i, 'MostFrequentTransactionTimeOfDay'] = most_frequent_category(categories)
df.to_csv('newStats.csv', index=False)

import json

threshold = 2.2

def count_days_with_high_ratio(calendar_str):
    if not isinstance(calendar_str, str):
        return None
    calendar = json.loads(calendar_str)

    count = 0
    for day, attributes in calendar.items():
        if 'collateralDebtRatio' in attributes and attributes['collateralDebtRatio'] > threshold:
            count += 1
    return count
df['DaysWithHighRatio'] = df['Calendar'].apply(count_days_with_high_ratio)
df.to_csv('newStats.csv', index=False)

import json
import time

x_percent=0.1

y_month=9
def loan_repaid_x_percent_in_last_y_months(calendar):
    six_months_ago = 1668297600 - y_month*30*24*60*60
    if isinstance(calendar, str):
        calendar_dict = json.loads(calendar)
    else:
        return False
    total_loan = 0
    total_repayment = 0
    for timestamp, day_dict in calendar_dict.items():
        timestamp = int(timestamp)
        if six_months_ago <= timestamp <= 1668297600:
            total_loan += day_dict['sizeDebtUSD']
            total_repayment += day_dict['amountPaymentsOnDay']
    if total_loan == 0:
        return False
    repayment_percentage = total_repayment / total_loan
    if repayment_percentage >= x_percent:
        return True
    else:
        return False
df['LoanRepaidxPercentInLastyMonths'] = df['Calendar'].apply(loan_repaid_x_percent_in_last_y_months)
df.to_csv('newStats.csv', index=False)

from datetime import datetime, timedelta

today = datetime.now()

six_months_ago = today - timedelta(days=15*30)
timestamp_today = int(today.timestamp())
timestamp_six_months_ago = int(six_months_ago.timestamp())


def evaluate_user_behavior(row):
    conditions_met = 0


    if row['DaysWithHighRatio'] > 60:
        conditions_met += 2
        if row['LoanRepaidxPercentInLastyMonths']:
          conditions_met += 4
    if row['timeStampFirstAnyTransactionAccount'] < timestamp_six_months_ago :
        conditions_met += 1

    if conditions_met==5 or conditions_met == 6 or conditions_met==7 or conditions_met == 3:
        return 'Good'
    elif conditions_met ==0 :
        return 'Bad'
    else:
        return 'Unknown'

df['user_class'] = df.apply(evaluate_user_behavior, axis=1)

import pandas as pd

df = df.drop(columns='Calendar')
df.to_csv('updated_data.csv', index=False)

## maybe add user class distribution function


 ## PREPARING THE DATA
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

df = df.drop(columns=['Unnamed: 0'])
#df = df.drop(columns=['Calendar'])
for col in ['timeStampsSupplyCollateral', 'timeStampsLoans', 'timeStampsReimbursement', 'timeStampsWithdrawCollateral','maximumDebt', 'atTimeMaximumDebtCollateralProvided']:
    df= df.drop(columns=col)

rows_with_nan = df[df.isna().any(axis=1)]

df.dropna()
df['LoanRepaidxPercentInLastyMonths'] = df['LoanRepaidxPercentInLastyMonths'].astype(int)
le = preprocessing.LabelEncoder()
df['MostFrequentTransactionTimeOfDay'] = le.fit_transform(df['MostFrequentTransactionTimeOfDay'])
df['user_class'] = le.fit_transform(df['user_class'])
#print(df.describe())
scaler = StandardScaler()
num_cols = ['sizeLoansUSD', 'sizeCollateralUSD', 'sizeReimbursementsUSD', 'numLoansUser', 'ratioCollateralToLoans', 'averageOfDailyCollateralToDebt', 'numTransactionsUser', 'DaysWithHighRatio']
df[num_cols] = scaler.fit_transform(df[num_cols])
for cols in ['DaysWithHighRatio','timeStampFirstAnyTransactionAccount','LoanRepaidxPercentInLastyMonths']:
  df = df.drop(columns=cols)
X = df.drop('user_class', axis=1)
y = df['user_class']

#print(X.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train = X_train.dropna()
y_train = y_train[X_train.index]
Knn_algorithm(X_train, y_train, X_test, pd, y_test)
print("second")
Random_forest(X_train,y_train,X_test,pd,y_test)
Logistic_regression(X_train,y_train,X_test,pd,y_test)
XGB_algorithm(X_train,y_train,X_test,pd,y_test)