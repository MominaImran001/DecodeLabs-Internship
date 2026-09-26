# Customer Segmentation Using K-Means Clustering

##  Project Overview

This project performs **customer segmentation** using unsupervised machine learning techniques.

The main objective is to group customers into meaningful segments based on their **spending behavior, purchasing activity, campaign acceptance, purchasing channels, family characteristics, and age**.

The project uses **K-Means Clustering** as the final clustering algorithm and applies **Standard Scaling** and **Principal Component Analysis (PCA)** as part of the preprocessing and analysis pipeline.

---

##  Project Objectives

* Clean and preprocess the customer dataset
* Handle missing values and outliers
* Convert date information into a usable format
* Create meaningful customer-level features
* Identify effective features for clustering
* Standardize numerical features
* Apply PCA for dimensionality analysis and visualization
* Determine an appropriate number of clusters
* Evaluate clusters using:

  * Elbow Method
  * Silhouette Score
* Build customer segments using K-Means
* Analyze and profile the resulting customer groups
* Assign meaningful customer segment labels
* Save the final segmented dataset

---

##  Dataset

The project uses a **marketing campaign customer dataset** containing customer demographic information, purchasing behavior, website activity, store activity, and campaign responses.

### Dataset Size

* **Rows:** 2,240
* **Original Columns:** 29

The dataset includes information such as:

* Customer ID
* Year of Birth
* Education
* Marital Status
* Income
* Number of children and teenagers
* Customer enrollment date
* Recency
* Product spending
* Web purchases
* Catalog purchases
* Store purchases
* Campaign acceptance
* Complaint information
* Campaign response

---

##  Data Preprocessing

### Missing Values

The `Income` column contained missing values.

Missing income values were handled using **median imputation**.

```python
median_income = df["Income"].median()
df["Income"] = df["Income"].fillna(median_income)
```

### Date Conversion

The `Dt_Customer` column was converted into a proper datetime format.

```python
df["Dt_Customer"] = pd.to_datetime(
    df["Dt_Customer"],
    dayfirst=True,
    errors="coerce"
)
```

### Income Outlier Handling

Income outliers were detected using the **IQR (Interquartile Range) method**.

The lower and upper bounds were calculated as:

```text
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

Instead of removing the outlier records, income values outside the valid range were **clipped** to the calculated bounds.

---

##  Feature Engineering

Several new features were created to represent customer behavior more effectively.

### 1. Total Spending

Total amount spent across all product categories.

```text
TotalSpending =
Wines + Fruits + Meat + Fish + Sweets + Gold
```

### 2. Total Purchases

Total number of purchases made through web, catalog, and store channels.

```text
TotalPurchases =
Web Purchases + Catalog Purchases + Store Purchases
```

### 3. Total Children

Combined number of children and teenagers in the household.

```text
TotalChildren = Kidhome + Teenhome
```

### 4. Total Campaign Accepted

Total number of marketing campaigns accepted by the customer.

### 5. Total Products Purchased

A product-purchase feature was created for analysis but was not included in the final clustering features because it duplicates the information represented by `TotalSpending`.

### 6. Average Purchase Value

Average spending per purchase.

```text
AveragePurchaseValue =
TotalSpending / TotalPurchases
```

### 7. Web Purchase Ratio

Proportion of purchases made through the web channel.

```text
WebPurchaseRatio =
WebPurchases / TotalPurchases
```

### 8. Store Purchase Ratio

Proportion of purchases made through stores.

```text
StorePurchaseRatio =
StorePurchases / TotalPurchases
```

### 9. Customer Age

Customer age was calculated from the year of birth.

```text
Age = 2014 - Year_Birth
```

Unrealistic ages above 100 were handled using the median age of customers with realistic ages.

### 10. Customer Enrollment Year

The year in which the customer enrolled was extracted from `Dt_Customer`.

This feature was evaluated but ultimately removed from the final clustering model because removing it improved the Silhouette Score.

---

## 🔎 Final Features

After feature engineering and feature evaluation, the following **8 features** were used for final clustering:

```text
TotalSpending
TotalPurchases
TotalChildren
TotalCampaignAccepted
AveragePurchaseValue
WebPurchaseRatio
StorePurchaseRatio
Age
```

`EnrollmentYear` was excluded because its inclusion resulted in a lower Silhouette Score.

| Feature Set            | Silhouette Score |
| ---------------------- | ---------------: |
| With EnrollmentYear    |           0.2612 |
| Without EnrollmentYear |           0.2944 |

This indicates that removing `EnrollmentYear` produced better cluster separation according to the Silhouette Score.

---

##  Feature Scaling

Since K-Means is a distance-based algorithm, the numerical features were standardized using `StandardScaler`.

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

This ensures that features with larger numerical ranges do not dominate the clustering process.

---

##  Principal Component Analysis (PCA)

PCA was applied to analyze the variance captured by the features and to create a two-dimensional representation for visualization.

### PCA Analysis

First, PCA was performed with all components to calculate:

* Explained variance ratio
* Cumulative explained variance

A separate 2-component PCA was then used for visualization.

```python
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)
```

The resulting two-dimensional representation was used to visualize the customer clusters.

---

##  Choosing the Number of Clusters

Two evaluation techniques were used to determine the clustering structure.

### Elbow Method

K-Means was tested for values of **K from 2 to 10**.

Inertia was recorded for each value of K and visualized using the Elbow Method.

```text
K = 2 → 10
```

The Elbow Method helps identify a point where increasing the number of clusters provides diminishing improvement in within-cluster variation.

### Silhouette Score

Silhouette Score was also calculated for K values from 2 to 10.

The score measures how well-separated and internally cohesive the clusters are.

Higher values indicate stronger separation between clusters.

---

##  Final K-Means Model

The final model uses:

```text
Algorithm: K-Means Clustering
Number of Clusters: 2
Random State: 42
n_init: 10
```

The final K-Means model was trained using the standardized final features.

```python
final_model = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)
```

The final clustering produced **2 customer segments**.

---

##  Customer Segments

The two clusters were given descriptive labels based on their observed purchasing behavior.

### Cluster 0

**Low-Spending / Low-Activity Customers**

This segment is represented by customers with comparatively lower spending and purchasing activity.

### Cluster 1

**High-Spending / High-Activity Customers**

This segment is represented by customers with comparatively higher spending and purchasing activity.

The labels are descriptive interpretations of the cluster profiles rather than predefined classes.

---

##  Cluster Analysis

For each cluster, the project calculates:

* Number of customers
* Percentage of customers
* Average total spending
* Average total purchases
* Average purchase value
* Average campaign acceptance
* Average web purchase ratio
* Average store purchase ratio
* Average customer age
* Average number of children

This provides a behavioral profile of each customer segment.

---

##  Visualizations

The project generates several visualizations:

### 1. Elbow Method

Shows K-Means inertia for different numbers of clusters.

### 2. Silhouette Score

Shows the Silhouette Score for different values of K.

### 3. PCA Cluster Visualization

Displays customers in two-dimensional PCA space and colors them according to their assigned cluster.

### 4. Cluster Size

Shows the number of customers belonging to each cluster.

### 5. Average Spending

Compares average total spending between the customer segments.

### 6. Average Purchases

Compares average purchasing activity between the segments.

### 7. Campaign Response

Compares average campaign acceptance between the segments.

---

##  Output

After clustering, the final dataset is saved as:

```text
customer_segmentation_results.csv
```

The output contains the original customer information together with the generated features, cluster assignments, and customer segment labels.

Two additional columns are created:

```text
Cluster
CustomerSegment
```

---

##  Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

### Machine Learning Techniques

* Median Imputation
* IQR Outlier Handling
* Feature Engineering
* Standard Scaling
* Principal Component Analysis (PCA)
* K-Means Clustering
* Elbow Method
* Silhouette Score

---

##  Project Structure

```text
Project_2/
│
├── code_1.py
├── data/
│   └── marketing_campaign.csv
│
├── customer_segmentation_results.csv
│
└── README.md
```

---

## ▶ How to Run

### 1. Clone the Repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the Project

```bash
cd Project_2
```

### 3. Install Required Libraries

```bash
pip install numpy pandas matplotlib scikit-learn
```

### 4. Run the Project

```bash
python code_1.py
```

The program will perform preprocessing, feature engineering, clustering, evaluation, visualization, and save the final segmented dataset.

---

##  Project Workflow

```text
Raw Dataset
     ↓
Data Loading
     ↓
Missing Value Handling
     ↓
Date Conversion
     ↓
Outlier Detection & Handling
     ↓
Feature Engineering
     ↓
Feature Selection
     ↓
Data Scaling
     ↓
PCA Analysis
     ↓
Elbow Method
     ↓
Silhouette Score
     ↓
K-Means Clustering
     ↓
Cluster Profiling
     ↓
Customer Personas
     ↓
Final Segmented Dataset
```

---

##  Key Takeaways

* Customer behavior can be analyzed without predefined target labels using unsupervised learning.
* Feature engineering helps represent customer purchasing behavior more effectively.
* Standardization is important for distance-based clustering algorithms such as K-Means.
* PCA provides a useful way to analyze dimensionality and visualize clusters.
* Both the Elbow Method and Silhouette Score were used to evaluate the clustering structure.
* The final model divides customers into two behavioral segments.
* Cluster profiling converts the mathematical clusters into interpretable customer personas.

---

##  Future Improvements

Possible future improvements include:

* Testing additional clustering algorithms such as DBSCAN and Agglomerative Clustering
* Performing deeper cluster validation
* Creating an interactive customer segmentation dashboard
* Developing more detailed marketing personas
* Comparing customer segments across individual product categories
* Analyzing campaign effectiveness for each segment
* Building a recommendation or targeted marketing system based on customer segments

---

##  Project Purpose

This project demonstrates an end-to-end **unsupervised machine learning workflow for customer segmentation**, from raw data preprocessing and feature engineering to clustering, evaluation, visualization, and business-oriented customer profiling.
