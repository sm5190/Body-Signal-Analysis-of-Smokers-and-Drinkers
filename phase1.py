import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from scipy.linalg import svd
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import seaborn as sns

from numpy import mean
from numpy import std

from sklearn.utils import resample

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
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


label_encoder = LabelEncoder()
df['SMK_stat_type_cd'] = label_encoder.fit_transform(df[['SMK_stat_type_cd']])

print("=============Step3: Anomaly or Outlier Detection===============")
columns_to_check = df.columns[0:]

num_rows = 8
num_cols = 3

num_plots = min(len(columns_to_check), num_rows * num_cols)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 50))

axes = axes.flatten()

sns.set(style="whitegrid")

for i in range(num_plots):
    column = columns_to_check[i]
    sns.histplot(data=df, x=column, ax=axes[i])
    axes[i].set_xlabel(column, fontsize=24)  # Adjust the fontsize as needed
    axes[i].set_ylabel("Count", fontsize=24)  # Adjust the fontsize as needed

for i in range(num_plots, num_rows * num_cols):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig("outlier_detection.jpg", dpi=300)
plt.show()


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
    f"Due to the removal of outliers the amount of entries reduced from {df_before_outlier} to {df_after_outlier} by {df_before_outlier-df_after_outlier} entries."
)

print("=============Step3: Check if Balanced===============")


sns.countplot(data=df, x='SMK_stat_type_cd')
plt.xlabel("SMK_stat_type_cd")
plt.ylabel("Number of Samples")
plt.title("Count plot of target class")
plt.savefig("countplot_before.jpg", dpi=300)
plt.show()




print("=============Step4: Downsampling===============")

shape_before_downsampling= df_cleaned.shape[0]



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

sns.countplot(data=balanced_data, x='SMK_stat_type_cd')
plt.xlabel("SMK_stat_type_cd")
plt.ylabel("Number of Samples")
plt.title("Count plot of target class")
plt.savefig("Countplt_after.jpg", dpi=300)
plt.show()
shape_after_downsampling=balanced_data.shape[0]
print(
    f"Due to the downsampling the amount of entries reduced from {shape_before_downsampling} to {shape_after_downsampling} by {shape_before_downsampling-shape_after_downsampling} entries."
)


print("=============Step5:Discretization & Binarization:one hot encoding===================")
balanced_data=pd.get_dummies(balanced_data,columns=['sex', 'hear_left','hear_right', 'urine_protein'],drop_first=True ).astype(int)
print("Data after OHE:", balanced_data.head().to_string())
print("=============Step6:Variable Transformation: Normalization, standardization, differencing===================")
num_cols=[    "age",
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
    "gamma_GTP"]

balanced_data_normalized= balanced_data
for col in num_cols:
    balanced_data_normalized[col] = (balanced_data_normalized[col]-balanced_data_normalized[col].min())/(balanced_data_normalized[col].max()-balanced_data_normalized[col].min())
print("Data after Normalization:\n", balanced_data_normalized.head().to_string())
balanced_data_standardized=balanced_data
for col in num_cols:
    balanced_data_standardized[col] = (balanced_data_standardized[col]-balanced_data_standardized[col].mean())/balanced_data_standardized[col].std()
print("Data after Standardization:\n", balanced_data_standardized.head().to_string())
balanced_data_differenced=balanced_data
balanced_data_differenced= balanced_data_differenced.diff().fillna(0)
print("Data after Differenciation:\n", balanced_data_differenced.head().to_string())

print("=============Covariance and Correlation===================")

X = balanced_data_standardized.drop(['DRK_YN','SMK_stat_type_cd'], axis=1, inplace=False)
y=balanced_data_standardized["SMK_stat_type_cd"]

# Sample Covariance Matrix display through a heatmap graph
X_num=X[num_cols]
covariance_matrix = X_num.cov()
#print(covariance_matrix.to_string())

# Adjust figure size based on the number of features
plt.figure(figsize=(26, 24))

# Adjust font size for annot values, xticks, and yticks
sns.heatmap(covariance_matrix, annot=True, cmap='coolwarm', fmt='.2f', xticklabels=X_num.columns, yticklabels=X_num.columns, annot_kws={"size": 20})
plt.title('Sample Covariance Matrix Heatmap', fontsize=24)
plt.xticks(rotation=45, ha='right', fontsize=20)  # Adjust rotation and font size for better visibility
plt.yticks(rotation=0, fontsize=20)
plt.tight_layout()
plt.savefig("COV.jpg", dpi=300)  # Ensures the plot layout is tight
plt.show()


# Sample Pearson Correlation Coefficients Matrix display through a heatmap graph
correlation_matrix = X.corr()
#print(correlation_matrix.to_string())
# Adjust figure size based on the number of features
plt.figure(figsize=(26, 24))

# Adjust font size for annot values, xticks, and yticks
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', xticklabels=X.columns, yticklabels=X.columns, annot_kws={"size": 14})
plt.title('Sample Correlation Matrix Heatmap', fontsize=18)
plt.xticks(rotation=45, ha='right', fontsize=14)  # Adjust rotation and font size for better visibility
plt.yticks(rotation=0, fontsize=14)
plt.tight_layout()
plt.savefig("CORR.jpg", dpi=300)  # Ensures the plot layout is tight
plt.show()

print("=============Step5:Dimensionality Reduction/Feature Selection===================")
print("------Method 1:PCA --------------")
pca = PCA()
pca.fit(X)


explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)


n_features_for_90_percent_variance = np.argmax(cumulative_explained_variance >= 0.90) + 1
print(f'Number of features are needed that explain more than 90% of the dependent variance: {n_features_for_90_percent_variance}')



plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_explained_variance, marker='o', linestyle='--')
plt.title('Cumulative Explained Variance vs. Number of Features')
plt.xlabel('Number of Features')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)

plt.axhline(y=0.90, color='r', linestyle='-')
plt.axvline(x=n_features_for_90_percent_variance, color='r', linestyle='-')

plt.tight_layout()
plt.savefig("pca_01.jpg",dpi=300)
plt.show()


#random forest feature importance





print("------Method 2:Random Forest Feature Importance (threshold 0.01)-------")

rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)


Feature_importances = rf.feature_importances_
features = X.columns
sorted_indices = np.argsort(Feature_importances)


threshold = 0.01
plt.figure(figsize=(10, 6))

plt.barh(range(X.shape[1]), Feature_importances[sorted_indices], align='center')
plt.axvline(x=threshold, color='r', linestyle='-')
plt.yticks(range(X.shape[1]), [features[i] for i in sorted_indices])
plt.xlabel('Feature Importance')
plt.title('Feature Importances')
plt.tight_layout()
plt.savefig('Feature_importances.jpg',dpi=300)
plt.show()



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




print("------Method 3: SVD; Finding best n_components with Logistic Regression model-------")

def get_models():
    models = dict()
    for i in range(1, 25):
        steps = [('svd', TruncatedSVD(n_components=i)), ('m', LogisticRegression(max_iter=20000))]
        models[str(i)] = Pipeline(steps=steps)
    return models


# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores



# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model, X, y)
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
plt.figure(figsize=(25, 10))
boxplot = plt.boxplot(results, labels=names, showmeans=True)

# Increase font size for tick labels, title, and axis labels
plt.xticks(rotation=45, ha='right', rotation_mode='anchor', fontsize=14)
plt.yticks(fontsize=14)
plt.title("Box Plot of SVD Number of Components vs. Classification Accuracy", fontsize=18)
plt.xlabel("Number of components", fontsize=16)
plt.ylabel("Classification Accuracy", fontsize=16)

# Adjust mean marker size and color
for element in ['means']:
    plt.setp(boxplot[element], markersize=10, markerfacecolor='blue', markeredgecolor='blue')

plt.tight_layout()  # Ensures the plot layout is tight
plt.savefig("SVD_components_vs_accuracy.jpg", dpi=300)
plt.show()


print("------Method 3:Singular Value Decomposition with n_components=20-------")

#X = X.iloc[0:1000, :]
#y=y.iloc[0: 1000]
svd_T = TruncatedSVD(n_components=18)

explained_variance = svd_T.fit(X).explained_variance_
X_reduced = svd_T.fit_transform(X)
S = svd_T.singular_values_
print(S)

# Plot the explained variance
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.title('Explained Variance by Singular Value')
#plt.axhline(y=0.95, color='r', linestyle='-', label='Explained Variance Threshold: 0.95')
plt.xlabel('Singular Value')
plt.ylabel('Explained Variance')
plt.grid(True)
plt.savefig("SVD_explained_variance.jpg", dpi=300)
plt.show()

print("------Method 4:Variance Inflation Factor (threshold=5.00)-------")
print("Initial VIF")
vif_data1 = pd.DataFrame()
vif_data1["Features"] = X.columns
vif_data1["VIF"] = [variance_inflation_factor(X.values.astype(float), i) for i in range(X.shape[1])]
print(vif_data1)
print("------Dropped tot_chole-------")
X.drop(['tot_chole'], axis=1, inplace=True)
vif_data2 = pd.DataFrame()

vif_data2["Features"] = X.columns
vif_data2["VIF"] = [variance_inflation_factor(X.values.astype(float), i) for i in range(X.shape[1])]
print(vif_data2)
# print("------Dropped height-------")
# X.drop(['height'], axis=1, inplace=True)
# vif_data3 = pd.DataFrame()
#
# vif_data3["Features"] = X.columns
# vif_data3["VIF"] = [variance_inflation_factor(X.values.astype(float), i) for i in range(X.shape[1])]
# print(vif_data3)
# print("------Dropped SBP-------")
# X.drop(['SBP'], axis=1, inplace=True)
# vif_data4= pd.DataFrame()
#
# vif_data4["Features"] = X.columns
# vif_data4["VIF"] = [variance_inflation_factor(X.values.astype(float), i) for i in range(X.shape[1])]
# print(vif_data4)
#
# print("------Dropped waistline-------")
# X.drop(['waistline'], axis=1, inplace=True)
# vif_data5= pd.DataFrame()
#
# vif_data5["Features"] = X.columns
# vif_data5["VIF"] = [variance_inflation_factor(X.values.astype(float), i) for i in range(X.shape[1])]
# print(vif_data5)
#
# print("------Dropped hemoglobin-------")
# X.drop(['hemoglobin'], axis=1, inplace=True)
# vif_data6= pd.DataFrame()
#
# vif_data6["Features"] = X.columns
# vif_data6["VIF"] = [variance_inflation_factor(X.values.astype(float), i) for i in range(X.shape[1])]
# print(vif_data6)
#
# print("------Dropped DBP-------")
# X.drop(['DBP'], axis=1, inplace=True)
# vif_data7= pd.DataFrame()
#
# vif_data7["Features"] = X.columns
# vif_data7["VIF"] = [variance_inflation_factor(X.values.astype(float), i) for i in range(X.shape[1])]
# print(vif_data7)
#
# print("------Dropped weight-------")
# X.drop(['weight'], axis=1, inplace=True)
# vif_data8= pd.DataFrame()
#
# vif_data8["Features"] = X.columns
# vif_data8["VIF"] = [variance_inflation_factor(X.values.astype(float), i) for i in range(X.shape[1])]
# print(vif_data8)
# print("------Dropped BLDS-------")
# X.drop(['BLDS'], axis=1, inplace=True)
# vif_data9= pd.DataFrame()
#
# vif_data9["Features"] = X.columns
# vif_data9["VIF"] = [variance_inflation_factor(X.values.astype(float), i) for i in range(X.shape[1])]
# print(vif_data9)
#
#
# print("------Dropped HDL_chole-------")
# X.drop(['HDL_chole'], axis=1, inplace=True)
# vif_data10= pd.DataFrame()
#
# vif_data10["Features"] = X.columns
# vif_data10["VIF"] = [variance_inflation_factor(X.values.astype(float), i) for i in range(X.shape[1])]
# print(vif_data10)
# Plot VIF values for each feature
# Sort the DataFrame by VIF values
vif_data_sorted = vif_data2.sort_values(by="VIF", ascending=True)

# Create a horizontal bar plot
plt.figure(figsize=(10, 6))
plt.barh(vif_data_sorted["Features"], vif_data_sorted["VIF"])
plt.title("VIF vs Final Remaining Features")
plt.xlabel("VIF")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("VIF_vs_features.jpg", dpi=300)
plt.show()

print("------Method 4: Removing highly correlated features ( threshold 0.5)-------")
correlated_features = set()
correlation_matrix = X.corr()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)

if not correlated_features:
    print("No correlated features found.")
else:
    print("Correlated features:")
    print(correlated_features)









