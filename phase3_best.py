
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, make_scorer, log_loss
#import preprocessing
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.model_selection import train_test_split
from sklearn import metrics, tree
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, auc, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

from sklearn.utils import resample
import warnings
warnings.filterwarnings("ignore")

#Preprocessing
df= pd.read_csv('/home/grads/mshutonu/ML_Project/Phase 1/Dataset_final.csv')
print(df.head())
df=df.sample(n=1000, random_state=42)

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

print("Data shape before downsapling:",df.shape)



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
num_cols=["age",
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


print("------Step 6:Method 2:Random Forest Feature Importance (threshold 0.01)-------")

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


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=5805)
def specificity_score(y_true, y_pred, class_label):
    """
    Calculate specificity for a particular class in a multiclass setting.

    Parameters:
    - y_true: True labels.
    - y_pred: Predicted labels.
    - class_label: The class for which specificity is calculated.

    Returns:
    - specificity: Specificity for the specified class.
    """
    cm = confusion_matrix(y_true, y_pred)
    true_negative = np.sum(np.delete(np.delete(cm, class_label, axis=0), class_label, axis=1))
    false_positive = np.sum(cm[:, class_label]) - cm[class_label, class_label]
    specificity = true_negative / (true_negative + false_positive)
    return specificity


def evaluate(Model,model, X, y, num_classes=2, k_folds=5):
    """
    Evaluate a classification model.

    Parameters:
    - model: The classification model to evaluate.
    - X: Feature matrix.
    - y: Labels.
    - num_classes: Number of classes (default is binary classification).
    - k_folds: Number of folds for stratified k-fold cross-validation.

    Returns:
    - None (displays various evaluation metrics).
    """
    # Confusion Matrix
    y_pred = model.predict(X)



    # Stratified K-fold Cross-validation
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    y_proba = cross_val_predict(model, X, y, cv=skf, method='predict_proba')
    y_pred_cv = np.argmax(y_proba, axis=1)

    # Plot Confusion Matrix for Cross-validation
    cm_cv = confusion_matrix(y, y_pred_cv)
    plt.figure(figsize=(8, 6))

    cf_matrix_normalized = cm_cv / cm_cv.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cf_matrix_normalized, annot=True, fmt='0.2%')

    plt.title("Cross-Validation Confusion Matrix")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig('KCM' + Model + '.jpg', dpi=300)
    plt.show()

    precision = precision_score(y, y_pred, average='weighted')

    # Recall (Sensitivity)
    recall = recall_score(y, y_pred, average='weighted')

    # Specificity (for multiclass classification)
    specificity = {}
    for class_label in range(num_classes):
        specificity[class_label] = specificity_score(y, y_pred, class_label)

    # F-score
    f_score = f1_score(y, y_pred, average='weighted')

    # ROC-AUC score
    y_bin = label_binarize(y, classes=range(num_classes))
    y_pred_proba = model.predict_proba(X)

    roc_auc = {}
    plt.figure(figsize=(8, 6))
    for class_label in range(num_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, class_label], y_pred_proba[:, class_label])
        roc_auc[class_label] = roc_auc_score(y_bin[:, class_label], y_pred_proba[:, class_label])

        plt.plot(fpr, tpr, label=f'Class {class_label} (AUC = {roc_auc[class_label]:.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig('KRoc' + Model + '.jpg', dpi=300)
    plt.show()

    return precision, recall, specificity, f_score, roc_auc


results_table = pd.DataFrame(columns=['Model', 'Precision', 'Recall', 'Specificity', 'F-score', 'Roc-Auc'])

print("============Decision Trees==================")
DT= DecisionTreeClassifier(random_state=5805)
DT.fit(X_train,y_train)
y_train_predicted = DT.predict(X_train)
y_test_predicted = DT.predict(X_test)
y_proba_DT=DT.predict_proba(X_test)
print(f'Train accuracy :{accuracy_score(y_train, y_train_predicted):.2f}')
print(f'Test accuracy : {accuracy_score(y_test, y_test_predicted): .2f}')

# Get feature importances
feature_importances = DT.feature_importances_
features = list(X.columns)

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print("Feature Importances:")
print(feature_importance_df)


# Specify a threshold for feature importance (e.g., 0.05)
threshold = 0.05

# Drop features with importance below the threshold
selected_features=[]
eliminated_features=[]

for feature, importance in zip(features,feature_importances):
    if importance>threshold:
        selected_features.append(feature)
    else:
        eliminated_features.append(feature)

print("Eliminated Features:")
print(eliminated_features)
print("\nFinal Selected Features:")
print(selected_features)

X_DT=X[selected_features]
X_train, X_test, y_train, y_test = train_test_split(X_DT, y , test_size=0.2, random_state=5805)

#grid search for best pre pruning
best_model= DecisionTreeClassifier(random_state=5805, criterion='gini', max_depth=7, max_features=None, min_samples_leaf=4, min_samples_split=2, splitter='best')

best_model.fit(X_train,y_train)
feature_names_list = X.columns.tolist()

# Now, use the list in the plot_tree function
tree.plot_tree(best_model, feature_names=feature_names_list, filled=True, class_names=['never', 'used to smoke but quit', 'still smoke'])
plt.title('Pre-pruned Tree')
plt.savefig('Pre-pruned Tree.jpg', dpi=300)
plt.show()

#Post pruning

optimal_alpha =0.00017839864883602806
print(f'Optimal Alpha: {optimal_alpha}')
pruned_model = DecisionTreeClassifier(random_state=5805, ccp_alpha=optimal_alpha)
pruned_model.fit(X_train, y_train)
y_pred_pruned_train = pruned_model.predict(X_train)
y_pred_pruned_test = pruned_model.predict(X_test)

print(f'Train accuracy on pruned model :{accuracy_score(y_train, y_pred_pruned_train):.2f}')
print(f'Test accuracy on pruned model: {accuracy_score(y_test, y_pred_pruned_test): .2f}')

feature_names_list = X.columns.tolist()

# Now, use the list in the plot_tree function
tree.plot_tree(pruned_model, feature_names=feature_names_list, filled=True, class_names=['never', 'used to smoke but quit', 'still smoke'])
plt.title('Post-pruned Tree')
plt.savefig('Post-pruned Tree.jpg', dpi=300)
plt.show()

#Evaluating DT
print('K-fold cross evaluation of Decision Tree Pre pruned model:')

precision, recall, specificity, f_score, roc_auc=evaluate(Model="DT Pre Pruned",model=best_model, X=X_test, y=y_test, num_classes=3)
results_table = results_table._append({
        'Model': 'DT Pre Pruned',
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F-score': f_score,
        'Roc-Auc': roc_auc
    }, ignore_index=True)

print(results_table.to_string())

print('K-fold cross evaluation of Decision Tree Post pruned model:')

precision, recall, specificity, f_score, roc_auc=evaluate(Model="DT Post Pruned",model=pruned_model, X=X_test, y=y_test, num_classes=3)


results_table = results_table._append({
        'Model': 'DT Post Pruned',
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F-score': f_score,
        'Roc-Auc': roc_auc
    }, ignore_index=True)

print(results_table.to_string())

# Split the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=5805)

print("=================Logistic Regression=========================")
# Grid search


best_lr_classifier = LogisticRegression(random_state=42,C=0.01, penalty='l2', solver='liblinear')

best_lr_classifier.fit(X_train,y_train)

#Evaluating LR

print('K-fold cross evaluation of Logistic Regression model:')

precision, recall, specificity, f_score, roc_auc=evaluate(Model="LR",model=best_lr_classifier, X=X_test, y=y_test, num_classes=3)

results_table = results_table._append({
        'Model': 'Logistic Regression',
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F-score': f_score,
        'Roc-Auc': roc_auc
    }, ignore_index=True)

print(results_table.to_string())

print("=================K Nearest Neighbours=========================")

# Find the k value with the highest accuracy
best_k = 24



knn_with_best_k=KNeighborsClassifier(n_neighbors=best_k).fit(X_train,y_train)

print('K-fold cross evaluation of K-Nearest Neighbours model:')
precision, recall, specificity, f_score, roc_auc=evaluate(Model="KNN",model=knn_with_best_k, X=X_test, y=y_test, num_classes=3)

results_table = results_table._append({
        'Model': 'K-Nearest Neighbours',
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F-score': f_score,
        'Roc-Auc': roc_auc
    }, ignore_index=True)

print(results_table.to_string())

print("=================Support Vector Machine=========================")

#SVM
best_svm_model = SVC(probability=True, C=1, kernel='linear').fit(X_train,y_train)

#Evaluating SVM

print('K-fold cross evaluation of Support Vector Classifier:')

precision, recall, specificity, f_score, roc_auc=evaluate(Model="SVC",model=best_svm_model, X=X_test, y=y_test, num_classes=3)
# Example Usage:
# Assuming you have a trained model 'clf' and feature matrix 'X' and labels 'y'

#results_table = pd.DataFrame(columns=['Model', 'Precision', 'Recall','Specificity', 'F-score', 'Roc-Auc'])

results_table = results_table._append({
        'Model': 'SVC',
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F-score': f_score,
        'Roc-Auc': roc_auc
    }, ignore_index=True)

print(results_table.to_string())

print("=================Naive Bayes=========================")
# Naïve Bayes

best_nb_classifier = GaussianNB(var_smoothing=1e-09).fit(X_train,y_train)

#Evaluating Naive Bayes
print('K-fold cross evaluation of Naive Bayes model:')

precision, recall, specificity, f_score, roc_auc=evaluate(Model="NB",model=best_nb_classifier, X=X_test, y=y_test, num_classes=3)


results_table = results_table._append({
        'Model': 'Naive Bayes',
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F-score': f_score,
        'Roc-Auc': roc_auc
    }, ignore_index=True)

print(results_table.to_string())

print("===========================Random Forest=================================")


best_rf = RandomForestClassifier(random_state=42,max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=200).fit(X_train,y_train)

print('K-fold cross evaluation of Random Forest model:')

precision, recall, specificity, f_score, roc_auc=evaluate(Model="RF",model=best_rf, X=X_test, y=y_test, num_classes=3)


results_table = results_table._append({
        'Model': 'Random Forest',
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F-score': f_score,
        'Roc-Auc': roc_auc
    }, ignore_index=True)

print(results_table.to_string())
#
# # Bagging with Random Forest
#
rf_classifier = RandomForestClassifier(max_depth=10, min_samples_leaf=1, min_samples_split=10, n_estimators=200)
bagging_classifier = BaggingClassifier(base_estimator=rf_classifier, random_state=42)
bagging_classifier.fit(X_train, y_train)
bagging_classifier.fit(X_train, y_train)
#
#
# # Stacking with Random Forest
#
base_classifiers = [('XGB', xgb.XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=100, random_state=42)),
                    ('svm_model', SVC(C=1, kernel='linear', probability=True)),
                    ('MLP',MLPClassifier(activation='logistic', alpha=0.001, hidden_layer_sizes=(50, 50), max_iter=300))
                    ]
stacking_classifier = StackingClassifier(estimators=base_classifiers, final_estimator=RandomForestClassifier(random_state=42))
stacking_classifier.fit(X_train, y_train)

# Boosting with XGBoost


best_xgb = xgb.XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=100, random_state=42).fit(X_train,y_train)


print('K-fold cross evaluation of Bagging Classifier:')

precision, recall, specificity, f_score, roc_auc=evaluate(Model="Bagging",model=bagging_classifier, X=X_test, y=y_test, num_classes=3)
# Example Usage:
# Assuming you have a trained model 'clf' and feature matrix 'X' and labels 'y'

#results_table = pd.DataFrame(columns=['Model', 'Precision', 'Recall','Specificity', 'F-score', 'Roc-Auc'])

results_table = results_table._append({
        'Model': 'Bagging',
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F-score': f_score,
        'Roc-Auc': roc_auc
    }, ignore_index=True)

print(results_table.to_string())

print('K-fold cross evaluation of Stacking Classifier:')

precision, recall, specificity, f_score, roc_auc=evaluate(Model="Stacking",model=stacking_classifier, X=X_test, y=y_test, num_classes=3)
# Example Usage:
# Assuming you have a trained model 'clf' and feature matrix 'X' and labels 'y'

#results_table = pd.DataFrame(columns=['Model', 'Precision', 'Recall','Specificity', 'F-score', 'Roc-Auc'])

results_table = results_table._append({
        'Model': 'Stacking',
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F-score': f_score,
        'Roc-Auc': roc_auc
    }, ignore_index=True)

print(results_table.to_string())

print('K-fold cross evaluation of Boosting:')

precision, recall, specificity, f_score, roc_auc=evaluate(Model="Boosting",model=best_xgb, X=X_test, y=y_test, num_classes=3)
# Example Usage:
# Assuming you have a trained model 'clf' and feature matrix 'X' and labels 'y'

#results_table = pd.DataFrame(columns=['Model', 'Precision', 'Recall','Specificity', 'F-score', 'Roc-Auc'])

results_table = results_table._append({
        'Model': 'Boosting',
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F-score': f_score,
        'Roc-Auc': roc_auc
    }, ignore_index=True)

print(results_table.to_string())

print("===========MLPClassifier=========================")
# Multilayer perception network



best_nn_classifier = MLPClassifier(random_state=42, activation='logistic',alpha=0.001, hidden_layer_sizes=(50, 50), max_iter=300)
#log_loss_scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
best_nn_classifier.fit(X_train, y_train)

print('K-fold cross evaluation of MLP Classifier:')

precision, recall, specificity, f_score, roc_auc=evaluate(Model="MLPC",model=best_nn_classifier, X=X_test, y=y_test, num_classes=3)

results_table = results_table._append({
        'Model': 'MLPC',
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F-score': f_score,
        'Roc-Auc': roc_auc
    }, ignore_index=True)

print(results_table.to_string())