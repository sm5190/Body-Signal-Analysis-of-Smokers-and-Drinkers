import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, make_scorer
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
df= pd.read_csv('/home/grads/mshutonu/ML_Project/Phase 1/Dataset_final.csv')

print(df.head().to_string())
print("=============Step1: Handling missing values=============")

print(df.isna().sum())
print("=============Step2: Dropping Duplicated Instances=============")
duplicated_rows = df[df.duplicated()]
print("Duplicated Rows: \n",duplicated_rows.count())
df=df.drop_duplicates()
df.reset_index(drop=True, inplace=True)


rem = {"Male": 0, "Female": 1}

df['sex']= df['sex'].replace(rem)
rem2 = {"N": 0, "Y": 1}

df['DRK_YN']= df['DRK_YN'].replace(rem2)
print(df.head())

print("=============Step3: Standardization=============")

num_cols=[  "age",
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

original_tot_chole_mean=df['tot_chole'].mean()
original_tot_chole_std=df['tot_chole'].std()
for col in num_cols:
    df[col] = (df[col]-df[col].mean())/df[col].std()


print("=============Step4: Checking and removing outliers=============")
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

df_cleaned=pd.get_dummies(df_cleaned,columns=['sex', 'hear_left','hear_right', 'urine_protein', 'DRK_YN', 'SMK_stat_type_cd'],drop_first=True ).astype(int)



X = df_cleaned.drop(['tot_chole' ], axis=1, inplace=False)
y=df_cleaned['tot_chole']

print("------Method 4:Variance Inflation Factor (threshold=5.00)-------")
print("Initial VIF")
vif_data1 = pd.DataFrame()
vif_data1["Features"] = X.columns
vif_data1["VIF"] = [variance_inflation_factor(X.values.astype(float), i) for i in range(X.shape[1])]
print(vif_data1)


# print("------Dropped waistline-------")
# X.drop(['waistline'], axis=1, inplace=True)
# vif_data5= pd.DataFrame()
#
# vif_data5["Features"] = X.columns
# vif_data5["VIF"] = [variance_inflation_factor(X.values.astype(float), i) for i in range(X.shape[1])]
# print(vif_data5)
#
# print("------Dropped HDL_chole-------")
# X.drop(['HDL_chole'], axis=1, inplace=True)
# vif_data3 = pd.DataFrame()
#
# vif_data3["Features"] = X.columns
# vif_data3["VIF"] = [variance_inflation_factor(X.values.astype(float), i) for i in range(X.shape[1])]
# print(vif_data3)



#
# # Plot VIF values for each feature
# plt.figure(figsize=(10, 6))
# plt.barh(vif_data3["Features"], vif_data3["VIF"])
# plt.title("VIF vs  Final Remaining Features")
# plt.xlabel("VIF")
# plt.ylabel("Features")
# plt.xticks(rotation=90)
# plt.savefig("P2_VIF_vs_features.jpg", dpi=300)
# plt.show()



print("------Checking for multicolinearity-------")
correlated_features = []
correlation_matrix = X.corr()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > .5:
            colname = correlation_matrix.columns[i]
            correlated_features.append(colname)


if not correlated_features:
    print("No correlated features found.")
else:
    print("Correlated features:")
    print(correlated_features)

X.drop(correlated_features, axis=1,inplace=True)

X = sm.add_constant(X)  # Add a constant term for the intercept

#print(X.dtypes)
#print(y.dtypes)


print("=============Step5: Step wise regression =============")
X_reg=X


model = sm.OLS(y, X_reg).fit()
print(model.summary())

selected_features = list(X_reg.columns)
eliminated_features = []

results_table = pd.DataFrame(columns=['Removed Feature', 'AIC', 'BIC', 'Adjusted R-squared', 'P-value'])
feature='urine_protein_3'
print(f'=============================Dropped {feature}=============================')

results_table = results_table._append({
        'Removed Feature': feature,
        'AIC': model.aic,
        'BIC': model.bic,
        'Adjusted R-squared': model.rsquared_adj,
        'P-value': model.pvalues[feature]
    }, ignore_index=True)

eliminated_features.append(feature)
selected_features.remove(feature)
X_reg.drop([feature],axis=1,inplace=True)


model = sm.OLS(y, X_reg).fit()
print(model.summary())


feature='urine_protein_4'
print(f'=============================Dropped {feature}=============================')

results_table = results_table._append({
        'Removed Feature': feature,
        'AIC': model.aic,
        'BIC': model.bic,
        'Adjusted R-squared': model.rsquared_adj,
        'P-value': model.pvalues[feature]
    }, ignore_index=True)

eliminated_features.append(feature)
selected_features.remove(feature)
X_reg.drop([feature],axis=1,inplace=True)


model = sm.OLS(y, X_reg).fit()
print(model.summary())




feature='sight_right'
print(f'=============================Dropped {feature}=============================')

results_table = results_table._append({
        'Removed Feature': feature,
        'AIC': model.aic,
        'BIC': model.bic,
        'Adjusted R-squared': model.rsquared_adj,
        'P-value': model.pvalues[feature]
    }, ignore_index=True)

eliminated_features.append(feature)
selected_features.remove(feature)
X_reg.drop([feature],axis=1,inplace=True)


model = sm.OLS(y, X_reg).fit()
print(model.summary())


feature='SMK_stat_type_cd_3'
print(f'=============================Dropped {feature}=============================')

results_table = results_table._append({
        'Removed Feature': feature,
        'AIC': model.aic,
        'BIC': model.bic,
        'Adjusted R-squared': model.rsquared_adj,
        'P-value': model.pvalues[feature]
    }, ignore_index=True)

eliminated_features.append(feature)
selected_features.remove(feature)
X_reg.drop([feature],axis=1,inplace=True)


model = sm.OLS(y, X_reg).fit()
print(model.summary())

feature='urine_protein_6'
print(f'=============================Dropped {feature}=============================')

results_table = results_table._append({
        'Removed Feature': feature,
        'AIC': model.aic,
        'BIC': model.bic,
        'Adjusted R-squared': model.rsquared_adj,
        'P-value': model.pvalues[feature]
    }, ignore_index=True)

eliminated_features.append(feature)
selected_features.remove(feature)
X_reg.drop([feature],axis=1,inplace=True)


model = sm.OLS(y, X_reg).fit()
print(model.summary())





feature='urine_protein_2'
print(f'=============================Dropped {feature}=============================')

results_table = results_table._append({
        'Removed Feature': feature,
        'AIC': model.aic,
        'BIC': model.bic,
        'Adjusted R-squared': model.rsquared_adj,
        'P-value': model.pvalues[feature]
    }, ignore_index=True)

eliminated_features.append(feature)
selected_features.remove(feature)
X_reg.drop([feature],axis=1,inplace=True)



model = sm.OLS(y, X_reg).fit()
print(model.summary())

feature='SGOT_ALT'
print(f'=============================Dropped {feature}=============================')

results_table = results_table._append({
        'Removed Feature': feature,
        'AIC': model.aic,
        'BIC': model.bic,
        'Adjusted R-squared': model.rsquared_adj,
        'P-value': model.pvalues[feature]
    }, ignore_index=True)

eliminated_features.append(feature)
selected_features.remove(feature)
X_reg.drop([feature],axis=1,inplace=True)



model = sm.OLS(y, X_reg).fit()
print(model.summary())

feature='SGOT_AST'
print(f'=============================Dropped {feature}=============================')

results_table = results_table._append({
        'Removed Feature': feature,
        'AIC': model.aic,
        'BIC': model.bic,
        'Adjusted R-squared': model.rsquared_adj,
        'P-value': model.pvalues[feature]
    }, ignore_index=True)

eliminated_features.append(feature)
selected_features.remove(feature)
X_reg.drop([feature],axis=1,inplace=True)



model = sm.OLS(y, X_reg).fit()
print(model.summary())


feature='sight_left'
print(f'=============================Dropped {feature}=============================')

results_table = results_table._append({
        'Removed Feature': feature,
        'AIC': model.aic,
        'BIC': model.bic,
        'Adjusted R-squared': model.rsquared_adj,
        'P-value': model.pvalues[feature]
    }, ignore_index=True)

eliminated_features.append(feature)
selected_features.remove(feature)
X_reg.drop([feature],axis=1,inplace=True)



model = sm.OLS(y, X_reg).fit()
print(model.summary())

feature='urine_protein_5'
print(f'=============================Dropped {feature}=============================')

results_table = results_table._append({
        'Removed Feature': feature,
        'AIC': model.aic,
        'BIC': model.bic,
        'Adjusted R-squared': model.rsquared_adj,
        'P-value': model.pvalues[feature]
    }, ignore_index=True)

eliminated_features.append(feature)
selected_features.remove(feature)
X_reg.drop([feature],axis=1,inplace=True)



model = sm.OLS(y, X_reg).fit()
print(model.summary())



feature='hear_left_2'
print(f'=============================Dropped {feature}=============================')

results_table = results_table._append({
        'Removed Feature': feature,
        'AIC': model.aic,
        'BIC': model.bic,
        'Adjusted R-squared': model.rsquared_adj,
        'P-value': model.pvalues[feature]
    }, ignore_index=True)

eliminated_features.append(feature)
selected_features.remove(feature)
X_reg.drop([feature],axis=1,inplace=True)



model = sm.OLS(y, X_reg).fit()
print(model.summary())


feature='BLDS'
print(f'=============================Dropped {feature}=============================')

results_table = results_table._append({
        'Removed Feature': feature,
        'AIC': model.aic,
        'BIC': model.bic,
        'Adjusted R-squared': model.rsquared_adj,
        'P-value': model.pvalues[feature]
    }, ignore_index=True)

eliminated_features.append(feature)
selected_features.remove(feature)
X_reg.drop([feature],axis=1,inplace=True)






model = sm.OLS(y, X_reg).fit()
print(model.summary())


# Display the final selected features
print("Final Selected Features:")
print(selected_features)

# Display the eliminated features
print("Eliminated Features:")
print(eliminated_features)

# Display the results table
print("\nResults Table:")
print(results_table.to_string())



print("=============Step6: Final Regression Model =============")



X=X[selected_features]

X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, shuffle=True, random_state=5805)

print(X_train.shape)
print(X_test.shape)

model=sm.OLS(y_train,X_train).fit()
y_pred = model.predict(X_test)




y_original = (y_test * original_tot_chole_std) + original_tot_chole_mean
y_pred_original = (y_pred * original_tot_chole_std) + original_tot_chole_mean
y_train_original = (y_train * original_tot_chole_std) + original_tot_chole_mean



plt.figure(figsize=(10, 6))
data_points1 = np.arange(len(y_train_original))
data_points= np.arange(len(y_original))


plt.plot(data_points1, y_train_original,label="Train Values" )
plt.plot(data_points, y_original, label='Test Values',  linestyle='-')
plt.plot(data_points, y_pred_original, label='Predicted Values', linestyle='-')
plt.title('Train, Test Set vs. Predicted Values')
plt.xlabel('# of samples')
plt.ylabel('Predicted Values  vs True values vs Train values')
plt.grid(True)
plt.legend()
plt.savefig('predvstrueplot_rf.jpg',dpi=300)
plt.show()


mse = ((y_original - y_pred_original) ** 2).mean()
results_table2 = pd.DataFrame(columns=['AIC', 'BIC', 'R-squared','Adjusted R-squared', 'MSE'])

results_table2 = results_table2._append({

        'AIC': model.aic,
        'BIC': model.bic,
        'R-squared': model.rsquared,
        'Adjusted R-squared': model.rsquared_adj,
        'MSE': mse
    }, ignore_index=True)

print("Final results:\n", results_table2)
# T-test analysis on coefficients
#Step1: T - Test Analysis(Individual Coefficients)
t_test_results = pd.DataFrame({
    'Coefficient': model.params,
    't-value': model.tvalues,
    'p-value': model.pvalues
})


print("T-Test Analysis:")
print(t_test_results)

print (X.columns)
# F-test analysis
f_test_results = model.f_test('age=height=weight=SBP=HDL_chole=LDL_chole=triglyceride=hemoglobin=serum_creatinine=gamma_GTP=DRK_YN_1=SMK_stat_type_cd_2')
print("\nF-Test Analysis:")
print(f_test_results)
#f_test_result = model.f_test()
#print("F-Test Result:")
#print(f_test_result)

# Confidence interval analysis
conf_int = model.conf_int()
# print("Confidence Intervals:")
# print(conf_int)
pred = model.get_prediction(X_test)
pred_summary = pred.summary_frame(alpha=0.05)
y_pred_original = pred_summary['mean']
lower_interval = pred_summary['obs_ci_lower']
upper_interval = pred_summary['obs_ci_upper']
y_pred_original = (y_pred_original * original_tot_chole_std) + original_tot_chole_mean
lower_interval = (lower_interval * original_tot_chole_std) + original_tot_chole_mean
upper_interval = (upper_interval * original_tot_chole_std) + original_tot_chole_mean
y_test_original = (y_test * original_tot_chole_std) + original_tot_chole_mean

# Randomly sample a subset of data points for clarity
subset_size = 100  # Adjust this value based on your preference
random_indices = np.random.choice(len(y_test_original), size=subset_size, replace=False)

data_points = np.arange(len(random_indices))
plt.figure(figsize=(10, 6))
plt.plot(data_points, y_pred_original.iloc[random_indices], label='Predicted values', marker='.')
plt.fill_between(data_points, lower_interval.iloc[random_indices], upper_interval.iloc[random_indices], alpha=0.5, label='CI')
plt.title('Predicted Total Cholesterol with Confidence Intervals (95%)')
plt.xlabel('Randomly Sampled Data Points')
plt.ylabel('Total_Cholesterol')
plt.legend()
plt.grid(True)
plt.savefig("Confidendence_Interval.jpg", dpi=300)
plt.show()




