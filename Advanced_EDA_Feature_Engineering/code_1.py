# My first internship project
#importing required libraries...
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
#defining data frame and reading the fata from excel file...
df = pd.read_excel("data/Dataset for Data Analytics.xlsx")
#displaying the first five rows of the data frame and other infoemation about the data frame...
print(df.head())
print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

#filling the missing values in the coupon code with "Unknown" to handle the missing data in the coupon code column.
df["CouponCode"] = df["CouponCode"].fillna("Unknown")
#checking the coupon code column fter filling the missing values 
print("\nMissing Values After Treatment:")
print(df.isnull().sum())


# IQR Outlier Detection for TotalPrice

Q1 = df["TotalPrice"].quantile(0.25)
Q3 = df["TotalPrice"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("\nIQR Analysis for TotalPrice:")
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)

#Outliers identifying
outliers = df[
    (df["TotalPrice"] < lower_bound) |
    (df["TotalPrice"] > upper_bound)
]
print("Above Upper Bound:")
print((df["TotalPrice"] > upper_bound).sum())

print("Below Lower Bound:")
print((df["TotalPrice"] < lower_bound).sum())
print("\nNumber of TotalPrice Outliers:")
print(len(outliers))

# Statistical Imputation Code(filling missing values in numeric columns with median)
numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

#neutralizing the outliers in the TotalPrice column by replacing them with the median value of the column.
df["TotalPrice"] = df["TotalPrice"].clip(
    lower=lower_bound,
    upper=upper_bound
)

#Feature Enginnering: creating new features based on existing data to enhance the dataset for model training and analysis.
#creating a new feature called "CustomerPurchaseFrequency"
customer_order_frequency = df["CustomerID"].value_counts()

df["CustomerPurchaseFrequency"] = df["CustomerID"].map(customer_order_frequency)

print(df[["CustomerID", "CustomerPurchaseFrequency"]].head(10))
#new feature called "OrderValuePerCartItem" which represents the average value of each item in cart for a given order

df["OrderValuePerCartItem"] = df["TotalPrice"] / df["ItemsInCart"]

print(df["OrderValuePerCartItem"].describe())
# new feature OrderDayofWeek 
df["OrderDayOfWeek"] = df["Date"].dt.dayofweek
# a new feature IsWeekend
df["IsWeekend"] = df["OrderDayOfWeek"].isin([5, 6]).astype(int)




df.to_csv("cleaned_data.csv", index=False)