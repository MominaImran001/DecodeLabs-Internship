# Advanced EDA & Feature Engineering

## About the Project

This project is focused on exploring and cleaning a real-world order dataset and preparing it for Machine Learning.

The dataset initially contains raw customer and order information. The main purpose of this project is to understand the data, deal with missing values, identify unusual values, and create new features that can make the data more useful for further analysis and Machine Learning.

## What I Worked On

During this project, I worked on the following areas:

* Explored the dataset and its basic structure
* Checked for missing values
* Handled missing values appropriately
* Identified outliers using the IQR method
* Reduced the effect of extreme values through clipping
* Created new features from the existing data
* Prepared a cleaner dataset for future Machine Learning tasks

## Dataset

The dataset contains **1200 records and 14 columns** related to customer orders.

Some of the main columns are:

* `Quantity`
* `UnitPrice`
* `ItemsInCart`
* `TotalPrice`
* `PaymentMethod`
* `OrderStatus`
* `CouponCode`
* `ReferralSource`

## Missing Values

During the initial analysis, the `CouponCode` column was found to have missing values.

Since `CouponCode` is a categorical column, the missing values were replaced with **"Unknown"** instead of using a numerical method such as mean or median.

## Outlier Detection

I used the **Interquartile Range (IQR)** method to detect outliers in the `TotalPrice` column.

The IQR is calculated as:


IQR = Q3 - Q1

The lower and upper limits are:

Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR

For `TotalPrice`, the analysis identified **8 values above the upper limit**.

Instead of removing these records, I used **IQR-based clipping** to limit the extreme values while keeping the original records in the dataset.

## Feature Engineering

I created new features using the existing columns to capture additional information from the order data.

### PricePerItem

```text
PricePerItem = TotalPrice / Quantity
```

This represents the average price associated with each item.

### AverageCartPrice

```text
AverageCartPrice = TotalPrice / ItemsInCart
```

This gives an idea of the average value per item in the customer's cart.

### QuantityPerCartItem

```text
QuantityPerCartItem = Quantity / ItemsInCart
```

This helps describe the relationship between the ordered quantity and the number of items in the cart.

## Tools & Libraries

* Python
* Pandas
* NumPy
* Scikit-learn
* OpenPyXL

## Project Structure

```text
Project_1/
│
├── data/
│   ├── raw_data.csv
│   └── cleaned_data.csv
│
├── code_1.py
├── README.md
└── requirements.txt
```

## How to Run

First, install the required libraries:

pip install -r requirements.txt

Then run the Python script:

python code_1.py


## Outcome

After the cleaning and feature engineering steps, the dataset is more structured and suitable for use in further Machine Learning experiments.

This project helped me practice practical data science skills such as **EDA, data cleaning, statistical analysis, outlier handling, and feature engineering**.








Numerical missing values were handled using median imputation because the median is less affected by extreme values.
Used capping to handle outliers.(Filled outliers with median)
