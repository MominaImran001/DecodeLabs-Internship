import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Loading Dataset 
df = pd.read_csv(r"C:\Users\mic\OneDrive\Desktop\Project_2\Customer_Segmentation\data\marketing_campaign.csv", sep="\t")

print("Dataset loaded successfully!")

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

#Handling Missing Values 

median_income = df["Income"].median()

df["Income"] = df["Income"].fillna(median_income)

print("\nMissing Income values after imputation:")
print(df["Income"].isnull().sum())

# Converting Date Column
df["Dt_Customer"] = pd.to_datetime(
    df["Dt_Customer"],
    dayfirst=True,
    errors="coerce"
)

print("\nDt_Customer data type:")
print(df["Dt_Customer"].dtype)

# Handling Incone Outlier

Q1 = df["Income"].quantile(0.25)
Q3 = df["Income"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

income_outliers = df[
    (df["Income"] < lower_bound) |
    (df["Income"] > upper_bound)
]

print("\nIncome Outlier Analysis:")

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)

print("\nNumber of Income Outliers:")
print(len(income_outliers))


# Cliping income outliers

df["Income"] = df["Income"].clip(
    lower=lower_bound,
    upper=upper_bound
)


# Feature Engineering

# 1. Total Spending

df["TotalSpending"] = (
    df["MntWines"]
    + df["MntFruits"]
    + df["MntMeatProducts"]
    + df["MntFishProducts"]
    + df["MntSweetProducts"]
    + df["MntGoldProds"]
)


# 2. Total Purchases

df["TotalPurchases"] = (
    df["NumWebPurchases"]
    + df["NumCatalogPurchases"]
    + df["NumStorePurchases"]
)


# 3. Total Children

df["TotalChildren"] = (
    df["Kidhome"]
    + df["Teenhome"]
)


# 4. Total Campaign Accepted

df["TotalCampaignAccepted"] = (
    df["AcceptedCmp1"]
    + df["AcceptedCmp2"]
    + df["AcceptedCmp3"]
    + df["AcceptedCmp4"]
    + df["AcceptedCmp5"]
)


# 5. Total Products Purchased
# Creating for analysis but not using it in final clustering
# because it duplicates TotalSpending.

df["TotalProductsPurchased"] = (
    df["MntWines"]
    + df["MntFruits"]
    + df["MntMeatProducts"]
    + df["MntFishProducts"]
    + df["MntSweetProducts"]
    + df["MntGoldProds"]
)


# 6. Average Purchase Value

df["AveragePurchaseValue"] = (
    df["TotalSpending"]
    / df["TotalPurchases"].replace(0, np.nan)
)

df["AveragePurchaseValue"] = (
    df["AveragePurchaseValue"].fillna(0)
)


# 7. Web Purchase Ratio

df["WebPurchaseRatio"] = (
    df["NumWebPurchases"]
    / df["TotalPurchases"].replace(0, np.nan)
)

df["WebPurchaseRatio"] = (
    df["WebPurchaseRatio"].fillna(0)
)


# 8. Store Purchase Ratio

df["StorePurchaseRatio"] = (
    df["NumStorePurchases"]
    / df["TotalPurchases"].replace(0, np.nan)
)

df["StorePurchaseRatio"] = (
    df["StorePurchaseRatio"].fillna(0)
)


# 9. Customer Age

df["Age"] = 2014 - df["Year_Birth"]


# 10. Customer Enrollment Year
# Creating it for analysis, but NOT gonna use it in final clustering
# because removing it improved the Silhouette Score.

df["EnrollmentYear"] = df["Dt_Customer"].dt.year

# Handling Urealistic Ages

print("\nAge Outliers:")

print(
    df[df["Age"] > 100][
        ["ID", "Year_Birth", "Age"]
    ]
)

median_age = df.loc[
    df["Age"] <= 100,
    "Age"
].median()

df.loc[
    df["Age"] > 100,
    "Age"
] = median_age

print("\nAge Statistics After Cleaning:")
print(df["Age"].describe())


# Final Feature Selection

# EnrollmentYear removed because:
# Silhouette with EnrollmentYear    = 0.2612
# Silhouette without EnrollmentYear = 0.2944

final_features = [

    "TotalSpending",

    "TotalPurchases",

    "TotalChildren",

    "TotalCampaignAccepted",

    "AveragePurchaseValue",

    "WebPurchaseRatio",

    "StorePurchaseRatio",

    "Age"
]

print("\nFinal Features:")
print(final_features)

print("\nNumber of Final Features:")
print(len(final_features))

# Data Quality Check

print("\nMissing Values:")

print(
    df[final_features].isnull().sum()
)

print("\nInfinite Values:")

print(
    np.isinf(df[final_features]).sum()
)

# Preparing Features

X = df[final_features].copy()

print("\nFeature Data Shape:")
print(X.shape)

# Standard Scaling 
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nScaling completed!")

print("Scaled Data Shape:")
print(X_scaled.shape)

#  PCA

# PCA with all components for explained variance

pca_full = PCA()

X_pca_full = pca_full.fit_transform(X_scaled)

explained_variance = (
    pca_full.explained_variance_ratio_
)

cumulative_variance = np.cumsum(
    explained_variance
)

print("\nExplained Variance Ratio:")
print(explained_variance)

print("\nCumulative Explained Variance:")
print(cumulative_variance)


#PCA 2D VISUALIZATION

pca_2d = PCA(
    n_components=2
)

X_pca_2d = pca_2d.fit_transform(
    X_scaled
)

print("\nPCA 2D Shape:")
print(X_pca_2d.shape)

print("\nPCA 2D Explained Variance:")
print(
    pca_2d.explained_variance_ratio_
)

print(
    "\nTotal PCA 2D Explained Variance:",
    pca_2d.explained_variance_ratio_.sum()
)


# ELBOW METHOD

k_values = range(2, 11)

inertia_values = []

print("\nElbow Method Inertia Values:")

for k in k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertia_values.append(
        model.inertia_
    )

    print(
        f"K = {k} --> "
        f"Inertia = {model.inertia_:.2f}"
    )


plt.figure(figsize=(9, 6))

plt.plot(
    list(k_values),
    inertia_values,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.xticks(list(k_values))

plt.grid(True)

plt.show()


# SILHOUETTE SCORE

silhouette_values = []

print("\nSilhouette Scores:")

for k in k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(
        X_scaled
    )

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_values.append(score)

    print(
        f"K = {k} --> "
        f"Silhouette Score = {score:.4f}"
    )


plt.figure(figsize=(9, 6))

plt.plot(
    list(k_values),
    silhouette_values,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score")

plt.xticks(list(k_values))

plt.grid(True)

plt.show()

# K COMPARISON

comparison_results = []

for k in [2, 3, 4, 5]:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(
        X_scaled
    )

    score = silhouette_score(
        X_scaled,
        labels
    )

    comparison_results.append({

        "K": k,

        "Inertia": model.inertia_,

        "Silhouette Score": score

    })


comparison_df = pd.DataFrame(
    comparison_results
)

print("\n" + "=" * 60)

print("K-MEANS MODEL COMPARISON")

print("=" * 60)

print(
    comparison_df.round(4)
)


# Final K

# K = 2 has the highest Silhouette Score.

final_k = 2

print("\nFinal K selected:", final_k)


# Final K-Means Model

final_model = KMeans(

    n_clusters=final_k,

    random_state=42,

    n_init=10

)

df["Cluster"] = final_model.fit_predict(
    X_scaled
)

print("\nK-Means clustering completed!")

# Final Silhouette Score

final_silhouette = silhouette_score(

    X_scaled,

    df["Cluster"]

)

print("\nFinal Silhouette Score:")

print(
    round(final_silhouette, 4)
)


# Cluster Size

cluster_size = (

    df["Cluster"]

    .value_counts()

    .sort_index()

)

cluster_percentage = (

    df["Cluster"]

    .value_counts(normalize=True)

    .sort_index()

    * 100

)


print("\n" + "=" * 60)

print("FINAL CLUSTER SIZE")

print("=" * 60)

print("\nNumber of Customers:")

print(cluster_size)

print("\nPercentage of Customers:")

print(
    cluster_percentage.round(2)
)


# Cluster Profile

profile_features = [

    "TotalSpending",

    "TotalPurchases",

    "TotalChildren",

    "TotalCampaignAccepted",

    "AveragePurchaseValue",

    "WebPurchaseRatio",

    "StorePurchaseRatio",

    "Age"

]


cluster_profile = (

    df.groupby("Cluster")[profile_features]

    .mean()

    .round(2)

)


print("\n" + "=" * 60)

print("FINAL CLUSTER PROFILE")

print("=" * 60)

print(cluster_profile)


# Detailed Customer Personas

print("\n" + "=" * 60)

print("CUSTOMER PERSONAS")

print("=" * 60)


# Cluster 0

cluster_0 = df[
    df["Cluster"] == 0
]


print("\nCLUSTER 0")

print(
    "Customers:",
    len(cluster_0)
)

print(
    "Percentage:",
    round(
        len(cluster_0) / len(df) * 100,
        2
    ),
    "%"
)

print(
    "Average Spending:",
    round(
        cluster_0["TotalSpending"].mean(),
        2
    )
)

print(
    "Average Purchases:",
    round(
        cluster_0["TotalPurchases"].mean(),
        2
    )
)

print(
    "Average Purchase Value:",
    round(
        cluster_0["AveragePurchaseValue"].mean(),
        2
    )
)

print(
    "Average Campaign Acceptance:",
    round(
        cluster_0["TotalCampaignAccepted"].mean(),
        2
    )
)

print(
    "Average Web Purchase Ratio:",
    round(
        cluster_0["WebPurchaseRatio"].mean(),
        2
    )
)

print(
    "Average Store Purchase Ratio:",
    round(
        cluster_0["StorePurchaseRatio"].mean(),
        2
    )
)

print(
    "Average Age:",
    round(
        cluster_0["Age"].mean(),
        2
    )
)

print(
    "Average Children:",
    round(
        cluster_0["TotalChildren"].mean(),
        2
    )
)


# Cluster 1

cluster_1 = df[
    df["Cluster"] == 1
]


print("\nCLUSTER 1")

print(
    "Customers:",
    len(cluster_1)
)

print(
    "Percentage:",
    round(
        len(cluster_1) / len(df) * 100,
        2
    ),
    "%"
)

print(
    "Average Spending:",
    round(
        cluster_1["TotalSpending"].mean(),
        2
    )
)

print(
    "Average Purchases:",
    round(
        cluster_1["TotalPurchases"].mean(),
        2
    )
)

print(
    "Average Purchase Value:",
    round(
        cluster_1["AveragePurchaseValue"].mean(),
        2
    )
)

print(
    "Average Campaign Acceptance:",
    round(
        cluster_1["TotalCampaignAccepted"].mean(),
        2
    )
)

print(
    "Average Web Purchase Ratio:",
    round(
        cluster_1["WebPurchaseRatio"].mean(),
        2
    )
)

print(
    "Average Store Purchase Ratio:",
    round(
        cluster_1["StorePurchaseRatio"].mean(),
        2
    )
)

print(
    "Average Age:",
    round(
        cluster_1["Age"].mean(),
        2
    )
)

print(
    "Average Children:",
    round(
        cluster_1["TotalChildren"].mean(),
        2
    )
)



# PCA CLUSTER VISUALIZATION


plt.figure(figsize=(11, 7))

plt.scatter(

    X_pca_2d[:, 0],

    X_pca_2d[:, 1],

    c=df["Cluster"],

    alpha=0.6

)

plt.xlabel("PCA Component 1")

plt.ylabel("PCA Component 2")

plt.title(
    "Customer Segmentation using K-Means"
)

plt.grid(True)

plt.show()



# CLUSTER SIZE BAR CHART


plt.figure(figsize=(8, 5))

plt.bar(

    cluster_size.index.astype(str),

    cluster_size.values

)

plt.xlabel("Cluster")

plt.ylabel("Number of Customers")

plt.title(
    "Number of Customers in Each Cluster"
)

plt.grid(
    axis="y"
)

plt.show()



#  SPENDING COMPARISON


average_spending = (

    df.groupby("Cluster")["TotalSpending"]

    .mean()

)


plt.figure(figsize=(8, 5))

plt.bar(

    average_spending.index.astype(str),

    average_spending.values

)

plt.xlabel("Cluster")

plt.ylabel("Average Total Spending")

plt.title(
    "Average Spending by Cluster"
)

plt.grid(
    axis="y"
)

plt.show()



#  PURCHASE COMPARISON


average_purchases = (

    df.groupby("Cluster")["TotalPurchases"]

    .mean()

)


plt.figure(figsize=(8, 5))

plt.bar(

    average_purchases.index.astype(str),

    average_purchases.values

)

plt.xlabel("Cluster")

plt.ylabel("Average Total Purchases")

plt.title(
    "Average Purchases by Cluster"
)

plt.grid(
    axis="y"
)

plt.show()



# CAMPAIGN RESPONSE COMPARISON


campaign_response = (

    df.groupby("Cluster")["TotalCampaignAccepted"]

    .mean()

)


plt.figure(figsize=(8, 5))

plt.bar(

    campaign_response.index.astype(str),

    campaign_response.values

)

plt.xlabel("Cluster")

plt.ylabel("Average Campaigns Accepted")

plt.title(
    "Campaign Acceptance by Cluster"
)

plt.grid(
    axis="y"
)

plt.show()


# CREATING CUSTOMER SEGMENT LABELS


cluster_names = {

    0: "Low-Spending / Low-Activity Customers",

    1: "High-Spending / High-Activity Customers"

}


df["CustomerSegment"] = (

    df["Cluster"]

    .map(cluster_names)

)



#  FINAL SEGMENT SUMMARY


segment_summary = (

    df.groupby(

        ["Cluster", "CustomerSegment"]

    )[profile_features]

    .mean()

    .round(2)

)


print("\n" + "=" * 60)

print("FINAL CUSTOMER SEGMENT SUMMARY")

print("=" * 60)

print(segment_summary)



#  SAVING FINAL DATASET


df.to_csv(

    "customer_segmentation_results.csv",

    index=False

)

print(

    "\nFinal dataset saved as "

    "'customer_segmentation_results.csv'"

)


#  FINAL PROJECT SUMMARY


print("\n" + "=" * 60)

print("PROJECT SUMMARY")

print("=" * 60)

print("\nDataset Size:", df.shape)

print(
    "Number of Features Used:",
    len(final_features)
)

print(
    "Final Number of Clusters:",
    final_k
)

print(
    "Final Silhouette Score:",
    round(final_silhouette, 4)
)


print("\nCustomer Segments:")

for cluster, name in cluster_names.items():

    count = (
        df["Cluster"] == cluster
    ).sum()

    percentage = (
        count / len(df) * 100
    )

    print(

        f"Cluster {cluster}: "

        f"{name} | "

        f"{count} customers | "

        f"{percentage:.2f}%"

    )

