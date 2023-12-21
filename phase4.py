import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import LabelEncoder
import preprocessing
import pandas as pd
from sklearn.utils import resample
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor

import seaborn as sns

from sklearn.utils import resample
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
# Load data
import warnings
warnings.filterwarnings("ignore")



df= pd.read_csv('/home/grads/mshutonu/ML_Project/Phase 1/Dataset_final.csv')
print(df.head())


print("=============Step1: Handling missing values=============")
print(df.isna().sum())
#print(df.describe().to_csv('describe.csv'))


print("=============Step2:Checking and handling duplicate instances==============")
duplicated_rows = df[df.duplicated()]
print(duplicated_rows.count())
df=df.drop_duplicates()
df.reset_index(drop=True, inplace=True)

#Encoding string type data
rem = {"Male": 0, "Female": 1}

df['sex']= df['sex'].replace(rem)
rem2 = {"N": 0, "Y": 1}

df['DRK_YN']= df['DRK_YN'].replace(rem2)




print("=============Step3: Anomaly or Outlier Detection===============")


df_before_outlier= df.shape[0]

columns_to_check = [
    "age",
    "height",
    "weight",
    "waistline",
    "sight_left",
    "sight_right",
    "SBP",
    "DBP",
    "BLDS",
    "tot_chole",
    "HDL_chole",
    "LDL_chole",
    "triglyceride",
    "hemoglobin",
    "serum_creatinine",
    "SGOT_AST",
    "SGOT_ALT",
    "gamma_GTP",
]
#
z_scores = (
    df[columns_to_check] - df[columns_to_check].mean()
) / df[columns_to_check].std()

# 3σ Standart deviation
threshold = 3
outliers = np.abs(z_scores) > threshold
outlier_columns = outliers.columns[outliers.any()]
df_cleaned = df[~outliers.any(axis=1)]
df_after_outlier = df_cleaned.shape[0]

print(
    f"Due to the removal of outliers the amout of entries reduced from {df_before_outlier} to {df_after_outlier} by {df_before_outlier-df_after_outlier} entries."
)


print("=============Step3: Downsampling===============")





minority_class =df['SMK_stat_type_cd'].value_counts().idxmin()

# Separate data by class
majority_classes = df[df['SMK_stat_type_cd'] != minority_class]
minority_class = df[df['SMK_stat_type_cd'] == minority_class]

# Calculate the target count in majority classes
majority_class_count = len(majority_classes)

# Downsample each majority class to match the minority class
downsampled_majority = pd.DataFrame()
for label in majority_classes['SMK_stat_type_cd'].unique():
    majority_class_subset = majority_classes[majority_classes['SMK_stat_type_cd'] == label]
    downsampled_class = resample(majority_class_subset, replace=False, n_samples=len(minority_class), random_state=42)
    downsampled_majority = pd.concat([downsampled_majority, downsampled_class])

# Combine classes
balanced_data = pd.concat([downsampled_majority, minority_class])

# Shuffle the data
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)






print("=============Step4:Discretization & Binarization:one hot encoding===================")
balanced_data=pd.get_dummies(balanced_data,columns=['sex', 'hear_left','hear_right', 'urine_protein'],drop_first=True ).astype(int)
print("Data after OHE:", balanced_data.head().to_string())
print("=============Step6:Variable Transformation: Normalization, standardization===================")
num_cols=[ "age",
    "height",
    "weight",
    "waistline",
    "sight_left",
    "sight_right",
    "SBP",
    "DBP",
    "BLDS",
    "tot_chole",
    "HDL_chole",
    "LDL_chole",
    "triglyceride",
    "hemoglobin",
    "serum_creatinine",
    "SGOT_AST",
    "SGOT_ALT",
    "gamma_GTP",]
balanced_data_standardized=balanced_data
for col in num_cols:
    balanced_data_standardized[col] = (balanced_data_standardized[col]-balanced_data_standardized[col].mean())/balanced_data_standardized[col].std()
print("Data after Standardization:", balanced_data_standardized.head().to_string())


print("=============Step5:Dimensionality Reduction/Feature Selection===================")

X = balanced_data_standardized.drop(['DRK_YN','SMK_stat_type_cd' ], axis=1, inplace=False)
y=balanced_data_standardized["SMK_stat_type_cd"]


#random forest feature importance





print("------Method 2:Random Forest Feature Importance (threshold 0.01)-------")

rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)


Feature_importances = rf.feature_importances_
features = X.columns
sorted_indices = np.argsort(Feature_importances)


threshold = 0.01

selected_features_rf=[]
eliminated_features_rf=[]

for feature, importance in zip(features,Feature_importances):
    if importance>threshold:
        selected_features_rf.append(feature)
    else:
        eliminated_features_rf.append(feature)

print("Eliminated Features:")
print(eliminated_features_rf)
print("\nFinal Selected Features:")
print(selected_features_rf)

X=X[selected_features_rf]



pca = PCA(n_components=2)
pca.fit(X)
X_transform = pca.transform(X)
# Label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)



k_values = []
accuracy_scores = []

# Function to calculate WCSS for a range of k values
def calculate_wcss(X, k_range):
    wcss_values = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X,y)
        wcss_values.append(kmeans.inertia_)
    return wcss_values

# Range of k values to try
k_values = range(1, 50)  # You can adjust the range as needed

# Calculate WCSS values
wcss = calculate_wcss(X_transform,k_values)

# Plotting WCSS values for different k values
plt.plot(k_values, wcss, marker='o', linestyle='-', color='b')
plt.title('Elbow Method For Optimal k (KMeans)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.savefig("Elbow_kmeans.jpg", dpi=300)
plt.tight_layout()
plt.show()

# K-means clustering with Silhouette analysis
silhouette_scores = []
k_values = range(2, 10)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_transform)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(X_transform, labels)
    silhouette_scores.append(silhouette_avg)

# Plot Silhouette scores for different k values
plt.plot(k_values, silhouette_scores, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis for K-means Clustering')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("silhouette.jpg", dpi=300)
plt.show()

# K-means clustering with optimal k
optimal_k = k_values[np.argmax(silhouette_scores)]
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_labels = kmeans.fit_predict(X_transform)

# Plotting the clusters with distinct colors
for cluster in range(optimal_k):
    cluster_points = X_transform[kmeans_labels == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster + 1}', edgecolor='k')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X', label='Centroids')
plt.title(f'K-means Clustering (k={optimal_k})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.savefig("Kmeans.jpg", dpi=300)
plt.show()

# dbscan = DBSCAN(eps=1.5, min_samples=15)
# #dbscan = DBSCAN(eps=.5, min_samples=20)
# dbscan_labels = dbscan.fit_predict(X_transform)
#
# # Plotting the clusters with distinct colors
# unique_labels = np.unique(dbscan_labels)
#
# for label in unique_labels:
#     if label == -1:  # Noise points in DBSCAN
#         noise_points = X_transform[dbscan_labels == label]
#         plt.scatter(noise_points[:, 0], noise_points[:, 1], label='Noise', color='gray', edgecolor='k', alpha=0.3)
#     else:
#         cluster_points = X_transform[dbscan_labels == label]
#         plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label + 1}',edgecolor='k')
#
# plt.title('DBSCAN Clustering')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.legend()
# plt.show()

# Apriori algorithm (Association Rule Mining)
# Create a binary dataset for association rule mining
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

df= pd.read_csv('/home/grads/mshutonu/ML_Project/Phase 1/smoking_driking_dataset_Ver01.csv')
print(df.head())
data=df.sample(n=60000, random_state=42)

#data = data[['SMK_stat_type_cd', 'DRK_YN', 'sex_1', 'hear_left_2.0', 'hear_right_2.0', 'urine_protein_2.0', 'urine_protein_3.0', 'urine_protein_4.0', 'urine_protein_5.0', 'urine_protein_6.0']]
data = data[['SMK_stat_type_cd', 'DRK_YN', 'sex', 'hear_left','hear_right', 'urine_protein']]

data=pd.get_dummies(data )
# Convert boolean values to 0 and 1
change = {False: 0, True: 1}
data = data.replace(change)

# Applying Apriori
a = TransactionEncoder()
a_data = a.fit(data).transform(data)
#a_data=data
df = pd.DataFrame(a_data, columns=a.columns_)

print("Processed Data:")
print(data.head())

print("\nDataFrame after Apriori:")
print(df.head())

# ===============================
# Applying Apriori and Resulting
# ==============================
df_frequent = apriori(df, min_support=0.0001, use_colnames=True, verbose=1)
print("\nDataFrame with Frequent Itemsets:")
print(df_frequent.head().to_string())

# Check if df_frequent is not empty before generating association rules
if not df_frequent.empty:
    df_ar = association_rules(df_frequent, metric='confidence', min_threshold=0.6)
    df_ar = df_ar.sort_values(['confidence', 'lift'], ascending=[False, False])
    print("\nAssociation Rules:")
    print(df_ar.to_string())
else:
    print("No frequent itemsets found. Check your data and support threshold.")

# te = TransactionEncoder()
# te_ary = te.fit(dataset).transform(dataset)
# df = pd.DataFrame(te_ary, columns=te.columns_)
#
# # Apply Apriori algorithm
# frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)
#
# # Generate association rules
# rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)
#
# # Display the results
# print("Frequent Itemsets:")
# print(frequent_itemsets)
#
# print("\nAssociation Rules:")
# print(rules)


# # Selecting specific columns
# df = df[['DRK_YN', 'SMK_stat_type_cd']]
# df['DRK_YN'] = label_encoder.fit_transform(df['DRK_YN'])
# df['SMK_stat_type_cd'] = label_encoder.fit_transform(df['SMK_stat_type_cd'])
# df.replace(2, 1, inplace=True)
#
# # Apply Apriori algorithm
# frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
# rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.7)
#
# # Display the association rules
# print('Association Rules:')
# print(rules.to_string())
