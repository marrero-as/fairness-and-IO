"""
Replication Code for: "Public AI and Educational Targeting: Bridging Algorithmic Fairness
and Inequality of Opportunity"

Developed by:
- Marrero, A.S. (Ángel S. Marrero)
- Giovanelli, J. (Joseph Giovanelli)

Description:
This script performs the core analysis of the paper: the estimation of inequality of
opportunity (IO) in 3rd grade and its decomposition by circumstance (Sections 4.1 and 5.1),
the training of the three predictive models of 6th-grade performance with OLS and Random
Forest (Section 4.2) and the calculation of accuracy and AI fairness metrics (TPR and TPR
ratios, Sections 4.3 and 5.2).

Inputs:
- ULL_panel_data.csv: Preprocessed longitudinal student data.

USAGE NOTE:
The estimation method is set with 'use_random_forest' (PREDICTIVE MODELS section):
True = Random Forest (RF), False = OLS. Run the script once with each value to obtain all
the results of the paper.

Outputs:
- Results_IO.xlsx: IO regression (Table 2), inequality and IO measures (Gini, MLD, variance) and
  relative IO weights of each circumstance (Table 3).
- Results_ols.xlsx / Results_rf.xlsx: Accuracy by performance segment (Table 4) and TPR and
  TPR ratios by circumstance and segment (Tables A1 and A2 of the Appendix).
- TPR_Ratios_ols.png / TPR_Ratios_rf.png: EO metric across distribution segments
  (Figures 1 and 2).
- graph_01.png ... graph_20.png and legend.png (RF only): TPR by model, group and circumstance
  for the lower (01-10) and upper (11-20) tails (Pareto gains, Figure 3).

License: MIT License
"""

import sys
import copy
import pandas as pd
from sklearn.model_selection import KFold
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import matplotlib as mpl

# Print UTF-8 to the console (the Windows cp1252 console cannot print 'ν' in the model summaries)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Load data
data = pd.read_csv('ULL_panel_data.csv', sep=',')

# Select only the variables we want to work for
data = data[['id_student_16_19', 'score_MAT', 'score_MAT3', 'a1',
             'mother_education', 'father_education', 'mother_occupation', 'father_occupation', 
             'inmigrant_second_gen', 'start_schooling_age', 'books',
             'capital_island', 'd14']]

# Drop observations with missing data in any of the variables that we will use in the models
    # Here, synthetic data methods can be used instead to fill in missing values
missing_columns = ['score_MAT3', 'a1', 'mother_education', 'father_education',
    'mother_occupation', 'father_occupation', 'inmigrant_second_gen',
    'start_schooling_age', 'books',
    'capital_island', 'd14']

data = data.dropna(subset=missing_columns)

# Generate quartiles of scores in sixth grade
data['scores_MATq'] = pd.qcut(data['score_MAT'], 4, labels=["1", "2", "3","4"])
data['scores_MATq'] = data['scores_MATq'].astype(int)

# ==============================================
# INEQUALITY OF OPPORTUNITY (IO) IN 3RD GRADE
# ==============================================
# Sections 4.1 (methodology) and 5.1 (results) of the paper, Equations 1-3 and Tables 2 and 3.
# All circumstances enter with all their categories
# (reference category = lowest code), so this must be run before d14 is binarized below

# Circumstances in the IO regression (raw name -> name in the paper)
io_names = {
    'a1': 'gender',
    'mother_education': 'mother_education',
    'father_education': 'father_education',
    'mother_occupation': 'mother_occupation',
    'father_occupation': 'father_occupation',
    'inmigrant_second_gen': 'immigrant_status',
    'start_schooling_age': 'age_at_enrollment',
    'books': 'books_in_home',
    'capital_island': 'geographic_area',
    'd14': 'teacher_transfer_rate',
}

# Equation 1: OLS of 3rd-grade scores on circumstances
io_formula = 'score_MAT3 ~ ' + ' + '.join(f'C({c})' for c in io_names)
io_model = smf.ols(io_formula, data=data).fit()
print(io_model.summary())

def gini(y):
    y = np.sort(np.asarray(y, dtype=float))
    n = len(y)
    i = np.arange(1, n + 1)
    return np.sum((2 * i - n - 1) * y) / (n * n * y.mean())

def mld(y):
    y = np.asarray(y, dtype=float)
    return np.mean(np.log(y.mean() / y))

def inequality_measures(y):
    y = pd.Series(y)
    return {'GINI': 100 * gini(y), 'MLD': 100 * mld(y), 'Variance': y.var(),
            'Std. dev.': y.std(), 'Coef. var.': 100 * y.std() / y.mean()}

# Total inequality, IO (inequality of the fitted values) and IO ratios (Equation 2)
total_ineq = inequality_measures(data['score_MAT3'])
io_ineq = inequality_measures(io_model.fittedvalues)
io_measures = pd.DataFrame({'Total inequality': total_ineq, 'IO': io_ineq})
io_measures['IO ratio (%)'] = 100 * io_measures['IO'] / io_measures['Total inequality']
io_measures.loc[['Std. dev.', 'Coef. var.'], 'IO ratio (%)'] = np.nan

# Equation 3: regression-based decomposition (Shorrocks), normalized to sum 100
X_io = io_model.model.exog
y_hat_io = io_model.fittedvalues.values
io_decomposition = []
for raw_name, paper_name in io_names.items():
    cols = [i for i, name in enumerate(io_model.model.exog_names) if name.startswith(f'C({raw_name})')]
    contribution = X_io[:, cols] @ io_model.params.values[cols]
    io_decomposition.append({
        'Circumstance': paper_name,
        'Relative IO weight': 100 * np.cov(contribution, y_hat_io)[0, 1] / np.var(y_hat_io, ddof=1)
    })
io_weights = pd.DataFrame(io_decomposition).sort_values('Relative IO weight', ascending=False)

# Regression table (Table 2)
io_table = pd.DataFrame({
    'Coefficient': io_model.params,
    't-stat': io_model.tvalues,
    'p-value': io_model.pvalues,
})
io_table.loc['R2', 'Coefficient'] = io_model.rsquared
io_table.loc['N', 'Coefficient'] = int(io_model.nobs)

with pd.ExcelWriter("Results_IO.xlsx", engine='openpyxl') as writer:
    io_table.to_excel(writer, sheet_name='IO_Regression')
    io_measures.to_excel(writer, sheet_name='IO_Measures')
    io_weights.to_excel(writer, sheet_name='IO_Weights', index=False)

# Order of the circumstances in all figures: by relative IO weight (Table 3)
order_custom = io_weights['Circumstance'].tolist()

# ==============================================
# PREDICTIVE MODELS OF 6TH-GRADE PERFORMANCE
# ==============================================
# Section 4.2 of the paper (Models 1, 2 and 3, Equations 4-8), evaluated with the accuracy and
# AI fairness metrics of Section 4.3 (Equations 9-10). Results in Section 5.2 (Table 4,
# Figures 1-3) and in the Appendix (Tables A1-A2)

# Some data corrections
data['d14'] = data['d14'].apply(lambda x: 1 if x == 1 else 0) # Variable d14: category 1 (almost no teacher change) is the unprotected group; categories 2-4 (any teacher turnover) are the protected group

# Rename the variables according to the names in the paper
data = data.rename(columns={
    'id_student_16_19': 'student_id',
    'a1': 'gender',
    'inmigrant_second_gen': 'immigrant_status',
    'start_schooling_age': 'age_at_enrollment',
    'books': 'books_in_home',
    'capital_island': 'geographic_area',
    'd14': 'teacher_transfer_rate',
})

# Now we define protected (prot) and unprotected (unprot) groups for each circumstance
def define_protected_groups(data):

    protected_conditions = {
        # Variables with 4 levels (top_level=4 is unprotected)
        'mother_education': data['mother_education'] != 4,
        'father_education': data['father_education'] != 4,
        'mother_occupation': data['mother_occupation'] != 4,
        'father_occupation': data['father_occupation'] != 4,
        'books_in_home': data['books_in_home'] != 4,
        
        # Variables with 3 levels (top_level=1 is unprotected - special case for age_at_enrollment)
        'age_at_enrollment': data['age_at_enrollment'] != 1,
        
        # Binary variables (top_level=1 is unprotected)
        'immigrant_status': data['immigrant_status'] != 1,
        'geographic_area': data['geographic_area'] != 1,
        'gender': data['gender'] != 1,
        'teacher_transfer_rate': data['teacher_transfer_rate'] != 1
    }
    
    return protected_conditions

# Variables for the models
Y_t_1 = "score_MAT3"
C = data[["gender", "immigrant_status", "age_at_enrollment", "mother_education", "father_education", "mother_occupation", "father_occupation", "books_in_home", "teacher_transfer_rate", "geographic_area"]]
Circumstances = ["gender", "immigrant_status", "age_at_enrollment", "mother_education", "father_education", "mother_occupation", "father_occupation", "books_in_home", "teacher_transfer_rate", "geographic_area"]

# Dummy variables (all variables C are categorical variables)
dummy_variables = pd.get_dummies(C ,columns = Circumstances ,drop_first = True)

# Five-fold cross-validation (Section 4.3): training-test splits of the complete-case sample
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# From now on, complete_data refers to the whole dataset
complete_data = copy.deepcopy(data)

# Estimation method: True = Random Forest (RF), False = OLS
use_random_forest = False  # Change to True for Random Forest
file_suffix = "rf" if use_random_forest else "ols"

# Loop over the 5 cross-validation folds
with pd.ExcelWriter(f"Results_{file_suffix}.xlsx", engine='openpyxl') as writer:
    all_results = pd.DataFrame()
    all_recall_results = []

    for fold_index, (train_indeces, test_indeces) in enumerate(kf.split(complete_data)):
        # Training and test splits of the data and of the circumstance dummies
        train_data, test_data = complete_data.iloc[train_indeces, :].copy(), complete_data.iloc[test_indeces, :].copy()
        train_dummies, test_dummies = dummy_variables.iloc[train_indeces], dummy_variables.iloc[test_indeces]

        # Predictors of Model 1: past performance (Y_t_1) + circumstance dummies (C)
        train_combined = pd.concat([train_data[Y_t_1], train_dummies], axis=1)
        test_combined = pd.concat([test_data[Y_t_1], test_dummies], axis=1)

        # "data" stores the test-fold predictions used to compute the metrics
        data = copy.deepcopy(test_data)

        if use_random_forest:

            # Model 1 (Equation 4): past performance and circumstances
            model1 = RandomForestRegressor(n_estimators=100, random_state=42, bootstrap= True, oob_score=True, min_samples_split=0.02 , min_samples_leaf=0.01, max_depth=10  )
            model1.fit(train_combined, train_data["score_MAT"])
            data['model1_pred'] = model1.predict(test_combined)

            # Model 2 (Equation 5): past performance only (Fairness Through Unawareness)
            model2 = RandomForestRegressor(n_estimators=100, random_state=42, bootstrap= True, oob_score=True, min_samples_split=0.02 , min_samples_leaf=0.01, max_depth=10  )
            model2.fit(train_data[[Y_t_1]], train_data["score_MAT"])
            data['model2_pred'] = model2.predict(test_data[[Y_t_1]])

            # Model 3 (Equations 6-8): past performance purged of circumstances
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42, bootstrap= True, oob_score=True, min_samples_split=0.02 , min_samples_leaf=0.01, max_depth=10  )
            rf_model.fit(train_data[Circumstances], train_data[Y_t_1])

            # First step (Equation 6): fitted values (Y_t_1_hat) and residuals (ν_hat = R_hat)
            train_data["Y_t_1_hat"] = rf_model.predict(train_data[Circumstances])
            train_data["ν_hat"] = train_data[Y_t_1] - train_data["Y_t_1_hat"]

            test_data["Y_t_1_hat"] = rf_model.predict(test_data[Circumstances])
            test_data["ν_hat"] = test_data[Y_t_1] - test_data["Y_t_1_hat"]

            data["Y_t_1_hat"] = test_data["Y_t_1_hat"]
            data["ν_hat"] = test_data["ν_hat"]

            # Second step (Equation 7)
            rf_model2 = RandomForestRegressor(n_estimators=100, random_state=42, bootstrap= True, oob_score=True, min_samples_split=0.02 , min_samples_leaf=0.01, max_depth=10  )
            rf_model2.fit(train_data[["Y_t_1_hat", "ν_hat"]], train_data["score_MAT"])
            data["model3_pred"] = rf_model2.predict(test_data[["Y_t_1_hat", "ν_hat"]])

            # Prediction exclusively of circumstances (not reported in the paper)
            test_data_circum = test_data[["Y_t_1_hat"]].copy()
            test_data_circum["ν_hat"] = 0
            data['model3_pred_circum'] = rf_model2.predict(test_data_circum)

            # Fairness-adjusted prediction (Equation 8): circumstance component set at its mean
            mean_circu = data['Y_t_1_hat'].mean()
            data['model3_pred_X'] = rf_model2.predict(test_data.assign(Y_t_1_hat=mean_circu)[["Y_t_1_hat", "ν_hat"]])
        else:
            # Model 1 (Equation 4): past performance and circumstances
            train_combined = train_combined.apply(pd.to_numeric, errors='coerce').astype("float64")
            X = np.asarray(train_combined, dtype=np.float64)
            y = np.asarray(train_data["score_MAT"], dtype=np.float64)
            model1 = sm.OLS(y, sm.add_constant(X)).fit()
            print(model1.summary())
            data["model1_pred"] = model1.predict(sm.add_constant(test_combined))

            # Model 2 (Equation 5): past performance only (Fairness Through Unawareness)
            model2 = sm.OLS(train_data["score_MAT"], sm.add_constant(train_data[Y_t_1])).fit()
            print(model2.summary())
            data["model2_pred"] =  model2.predict(sm.add_constant(test_data[Y_t_1]))

            # Model 3 (Equations 6-8): past performance purged of circumstances
            train_dummies = train_dummies.apply(pd.to_numeric, errors='coerce').astype("float64")
            X = np.asarray(train_dummies, dtype=np.float64)
            y = np.asarray(train_data["score_MAT3"], dtype=np.float64)
            model3 = sm.OLS(y, sm.add_constant(X)).fit()
            print(model3.summary())

            # First step (Equation 6): fitted values (Y_t_1_hat) and residuals (ν_hat = R_hat)
            train_data['Y_t_1_hat'] = model3.fittedvalues
            train_data['ν_hat'] = model3.resid
            test_data["Y_t_1_hat"] = model3.predict(sm.add_constant(test_dummies))
            test_data["ν_hat"] = test_data[Y_t_1] - test_data["Y_t_1_hat"]
            data["Y_t_1_hat"] = test_data["Y_t_1_hat"]
            data["ν_hat"] = test_data["ν_hat"]

            # Second step (Equation 7)
            model4 = sm.OLS(train_data["score_MAT"], sm.add_constant(train_data[["Y_t_1_hat", "ν_hat"]])).fit()
            print(model4.summary())
            data["model3_pred"] = model4.predict(sm.add_constant(test_data[["Y_t_1_hat", "ν_hat"]]))

            # Prediction exclusively of circumstances (not reported in the paper)
            data['model3_pred_circum'] = model4.params['const'] + model4.params['Y_t_1_hat'] * data['Y_t_1_hat']
            # Fairness-adjusted prediction (Equation 8): circumstance component set at its mean
            mean_circu = data['Y_t_1_hat'].mean()
            data['mean_circu'] = mean_circu
            data['model3_pred_X'] = (model4.params['const'] + 
                                    model4.params['ν_hat'] * data['ν_hat'] + 
                                    model4.params['Y_t_1_hat'] * mean_circu)

        # Transform predictions(continuous) to quartiles(categorical)
        data['scores_MAT_pred1'] = pd.qcut(data['model1_pred'], 4, labels=["1", "2", "3","4"])
        data['scores_MAT_pred1'] = data['scores_MAT_pred1'].astype(int)
        data['scores_MAT_pred2'] = pd.qcut(data['model2_pred'], 4, labels=["1", "2", "3","4"])
        data['scores_MAT_pred2'] = data['scores_MAT_pred2'].astype(int)
        data['scores_MAT_pred3'] = pd.qcut(data['model3_pred'], 4, labels=["1", "2", "3","4"])
        data['scores_MAT_pred3'] = data['scores_MAT_pred3'].astype(int)
        data['scores_MAT_pred_C'] = pd.qcut(data['model3_pred_circum'], 4, labels=["1", "2", "3","4"])
        data['scores_MAT_pred_C'] = data['scores_MAT_pred_C'].astype(int)
        data['scores_MAT_pred_X'] = pd.qcut(data['model3_pred_X'], 4, labels=["1", "2", "3","4"])
        data['scores_MAT_pred_X'] = data['scores_MAT_pred_X'].astype(int)

        # Transform predictions(continuous) to percentiles but percentiles 2 and 3 equal (between 25 and 75 percentil) (lower and upper tail and middle range)
        data['scores_MAT_pred1_t'] = data['scores_MAT_pred1'].apply(lambda x: 1 if x == 1 else (2 if x == 2 or x == 3 else 3))
        data['scores_MAT_pred2_t'] = data['scores_MAT_pred2'].apply(lambda x: 1 if x == 1 else (2 if x == 2 or x == 3 else 3))
        data['scores_MAT_pred_X_t'] = data['scores_MAT_pred_X'].apply(lambda x: 1 if x == 1 else (2 if x == 2 or x == 3 else 3))

        # Discretize scores MAT in lower and upper tail and middle range 
        data['scores_MATq_3'] = data['scores_MATq'].apply(lambda x: 1 if x == 1 else (2 if x in [2, 3] else 3))

        # Create a DataFrame 
        predictions_df = data[['scores_MATq', 'scores_MATq_3', 
                             'scores_MAT_pred1', 'scores_MAT_pred2', 'scores_MAT_pred_X',
                             'scores_MAT_pred1_t', 'scores_MAT_pred2_t', 'scores_MAT_pred_X_t']].copy()
        
        # Save predictions
        #predictions_df.to_excel(writer, sheet_name=f'Fold_{fold_index+1}_Pred', index=False)   #if you want to save the predictions of each fold
        
        # --- Calculate accuracy (Table 4) ---
        models = {
            "Model1_t": "scores_MAT_pred1_t",
            "Model2_t": "scores_MAT_pred2_t",    
            "Model_X_t": "scores_MAT_pred_X_t"
        }

        fold_results = {"Fold": fold_index + 1}

        # Mask for each segment
        mask_p25 = (data['scores_MATq_3'] == 1)  # Lower tail (percentil 25)
        mask_p75 = (data['scores_MATq_3'] == 3)  # Upper tail (percentil 75)
        mask_middle = (data['scores_MATq_3'] == 2)  # Middle range

        for model_name, pred_col in models.items():
            # Accuracy full range
            fold_results[f"{model_name}_global"] = accuracy_score(data['scores_MATq_3'], data[pred_col])
            
            # Accuracy for each segment
            fold_results[f"{model_name}_p25"] = accuracy_score(
                data[mask_p25]['scores_MATq_3'], 
                data[mask_p25][pred_col]
            )
            fold_results[f"{model_name}_p75"] = accuracy_score(
                data[mask_p75]['scores_MATq_3'], 
                data[mask_p75][pred_col]
            )
            fold_results[f"{model_name}_middle"] = accuracy_score(
                data[mask_middle]['scores_MATq_3'], 
                data[mask_middle][pred_col]
            )

        # Save results
        if fold_index == 0:
            all_results = pd.DataFrame([fold_results])
        else:
            all_results = pd.concat([all_results, pd.DataFrame([fold_results])], ignore_index=True)

        # Recall (TPR, Equation 9) for the different segments of the distribution and for protected and unprotected groups of the circumstances
        protected_conditions = define_protected_groups(data)
        recall_results = []

        for var_name, prot_cond in protected_conditions.items():
            for model_name, pred_col in models.items():
                for seg_name, seg_val in [('Lower_tail', 1), 
                                        ('Middle_range', 2), 
                                        ('Upper_tail', 3)]:

                    # PROTECTED GROUP
                    # Recall
                    recall_mask_prot = (data['scores_MATq_3'] == seg_val) & prot_cond
                    if recall_mask_prot.sum() > 0:
                        y_true_recall = data[recall_mask_prot]['scores_MATq_3']
                        y_pred_recall = data[recall_mask_prot][pred_col]

                        tp = sum((y_true_recall == seg_val) & (y_pred_recall == seg_val))
                        fn = sum((y_true_recall == seg_val) & (y_pred_recall != seg_val))
                        prot_recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
                    else:
                        prot_recall = np.nan

                    # UNPROTECTED GROUP
                    # Recall
                    recall_mask_unprot = (data['scores_MATq_3'] == seg_val) & ~prot_cond
                    if recall_mask_unprot.sum() > 0:
                        y_true_recall = data[recall_mask_unprot]['scores_MATq_3']
                        y_pred_recall = data[recall_mask_unprot][pred_col]

                        tp = sum((y_true_recall == seg_val) & (y_pred_recall == seg_val))
                        fn = sum((y_true_recall == seg_val) & (y_pred_recall != seg_val))
                        unprot_recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
                    else:
                        unprot_recall = np.nan

                    # Save results
                    recall_results.append({
                        'Fold': fold_index + 1,
                        'Variable': var_name,
                        'Segment': seg_name,
                        'Model': model_name,
                        'Protected_Recall': prot_recall,
                        'Unprotected_Recall': unprot_recall
                    })
        #pd.DataFrame(recall_results).to_excel(writer, sheet_name=f'Fold_{fold_index+1}_Recall', index=False)  #if you want to save the recalls of each fold
        all_recall_results.extend(recall_results)
    # Save accuracy results by fold (Table 4 reports the average across the five folds)
    all_results.to_excel(writer, sheet_name='Accuracy_Summary', index=False)
    recall_df = pd.DataFrame(all_recall_results)

    # Group to get average recall values
    recall_summary = recall_df.groupby(['Variable', 'Segment', 'Model']).agg({
        'Unprotected_Recall': 'mean',
        'Protected_Recall': 'mean'
    }).reset_index()

    # Pivot: columns will be (Segment, Model), values will be the recalls
    recall_pivot = recall_summary.pivot_table(
        index='Variable',
        columns=['Segment', 'Model'],
        values=['Unprotected_Recall', 'Protected_Recall']
    )

    # Rearrange levels: we want columns as (Segment, Model), with Recall as row index
    # Step 1: Stack to create 'Group' index
    recall_pivot = recall_pivot.stack(level=0)

    # Step 2: Rename indexes
    recall_pivot.index.set_names(['Variable', 'Group'], inplace=True)
    recall_pivot.columns.set_names(['Segment', 'Model'], inplace=True)

    # Step 3: Reorder rows so Unprotected_Recall comes before Protected_Recall for each Variable
    ordered_index = (
        recall_pivot.index.get_level_values('Variable').unique()
        .to_series()
        .repeat(2)
        .reset_index(drop=True)
        .to_frame(name='Variable')
    )
    ordered_index['Group'] = ['Unprotected_Recall', 'Protected_Recall'] * (len(ordered_index) // 2)
    ordered_index = pd.MultiIndex.from_frame(ordered_index)

    # Reindex using the new order
    recall_pivot = recall_pivot.reindex(ordered_index)

    # Step 4: Sort columns by Segment and Model
    recall_pivot = recall_pivot.sort_index(axis=1, level=['Segment', 'Model'])

    recall_with_ratios = recall_pivot.copy()

    segments = recall_pivot.columns.levels[0]
    models = recall_pivot.columns.levels[1]

    new_columns = []

    for segment in segments:
        for model in models:
            col = (segment, model)
            if col not in recall_pivot.columns:
                continue

            # Add original column
            new_columns.append(col)

            # Get protected and unprotected values
            protected = recall_pivot.xs('Protected_Recall', level='Group').loc[:, col]
            unprotected = recall_pivot.xs('Unprotected_Recall', level='Group').loc[:, col]

            # Compute TPR ratio (Equation 10)
            ratio = protected / unprotected

            # Create full column with NaNs
            ratio_series = pd.Series(np.nan, index=recall_pivot.index)

            # Assign 1.0 to Unprotected_Recall rows
            ratio_series.loc[recall_pivot.index.get_level_values('Group') == 'Unprotected_Recall'] = 1.0

            # Assign ratio only to Protected_Recall rows
            for idx in ratio.index:
                ratio_series.loc[(idx, 'Protected_Recall')] = ratio.loc[idx]

            # Add new column to DataFrame
            ratio_col = (segment, f'{model}_ratio')
            recall_with_ratios[ratio_col] = ratio_series

            # Add new column to order list
            new_columns.append(ratio_col)

    # Sort columns
    recall_with_ratios = recall_with_ratios.loc[:, new_columns]

    # Custom variable order: order_custom (circumstances sorted by relative IO weight, see IO section)
    new_index = pd.MultiIndex.from_tuples(
    [(var, group)
     for var in order_custom
     for group in ['Unprotected_Recall', 'Protected_Recall']],
    names=['Variable', 'Group']
        )
    
    new_index = new_index.intersection(recall_with_ratios.index)

    # Reindex
    recall_with_ratios = recall_with_ratios.reindex(new_index)

    # Export (Table A1 for OLS, Table A2 for RF)
    recall_with_ratios.to_excel(writer, sheet_name='TPR_Summary')

# ==============================================
# PLOTS
# ==============================================
# TPR ratios across distribution segments: Figure 1 (OLS) and Figure 2 (RF)
variable_rename = {
    "father_education": "Father educ.",
    "books_in_home": "Nº books",
    "mother_education": "Mother educ.",
    "age_at_enrollment": "Age at enroll.",
    "teacher_transfer_rate": "Teach.trans.rate",
    "father_occupation": "Father occup.",
    "gender": "Gender",
    "mother_occupation": "Mother occup.",  
    "geographic_area": "Geog. area",
    "immigrant_status": "Immigr. stat."
}

model_rename = {
    "Model1_t": "Model 1",
    "Model2_t": "Model 2",
    "Model_X_t": "Model 3"
}

# Colors for screen; markers and line styles keep the models distinguishable in black and white
model_styles = {
    "Model1_t": {
        "linestyle": "-",
        "marker": "o",
        "color": "#2a78d6"
    },
    "Model2_t": {
        "linestyle": "--",
        "marker": "s",
        "color": "#eb6834"
    },
    "Model_X_t": {
        "linestyle": "-",
        "marker": "^",
        "color": "#1baf7a"
    }
}

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    
    "font.size": 22,
    "axes.titlesize": 26,
    "axes.labelsize": 26,
    "xtick.labelsize": 22,
    "ytick.labelsize": 24,
    "legend.fontsize": 24,

    "lines.linewidth": 3.5,
    "lines.markersize": 9,

    "axes.linewidth": 1.2,
    "axes.edgecolor": "black",

    "grid.linestyle": "--",
    "grid.alpha": 0.3,

    "figure.facecolor": "white",
    "axes.facecolor": "white",

    "axes.spines.top": False,
    "axes.spines.right": False,
})


filename = f"Results_{file_suffix}.xlsx"
recall_df = pd.read_excel(
    filename,
    sheet_name='TPR_Summary',
    header=[0, 1],
    index_col=[0, 1]
)

segments = ['Lower_tail', 'Middle_range', 'Upper_tail']

fig, axes = plt.subplots(1, 3, figsize=(16, 8), sharey=True)

for ax, segment in zip(axes, segments):

    df_prot = recall_df.loc[
        (slice(None), 'Protected_Recall'),
        recall_df.columns.get_level_values(0) == segment
    ]

    ratio_cols = [col for col in df_prot.columns if 'ratio' in col[1]]

    variables = df_prot.index.get_level_values(0)
    renamed_variables = [variable_rename.get(var, var) for var in variables]

    for col in ratio_cols:
        model_name = col[1].replace('_ratio', '')
        legend_label = model_rename.get(model_name, model_name)

        style = model_styles.get(model_name, {})

        ax.plot(
            renamed_variables,
            df_prot[col],
            linestyle=style.get("linestyle", "-"),
            marker=style.get("marker", "o"),
            linewidth=3,
            markersize=10,
            label=legend_label,
            color=style.get("color", "black")
        )

    ax.set_title(segment.replace('_', ' '), fontsize=26)
    ax.axhline(
    1.0,
    color='grey',
    linestyle='--',
    linewidth=2,
    zorder=0
)
    ax.set_xticks(range(len(renamed_variables)))
    ax.set_xticklabels(
        renamed_variables,
        rotation=45,
        ha='right',
        fontsize=20
    )
    ax.grid(axis='x')

# Shared Y-axis
axes[0].set_ylabel('TPR Ratio', fontsize=26)
axes[0].set_ylim(0.4, 2.45)
axes[0].set_yticks(np.arange(0.4, 2.45, 0.2))
axes[0].tick_params(axis='y', labelsize=22)

# Common legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc='lower right',
    ncol=len(handles),
    fontsize=22,
    frameon=False
)

plt.tight_layout(rect=[0, 0.08, 1, 1])

# Save panel figure (Figure 1 or 2)
panel_filename = f"TPR_Ratios_{file_suffix}.png"
plt.savefig(panel_filename, dpi=300)
plt.close()

# ==============================================
# PARETO GAINS PLOTS (RF only)
# ==============================================
# Figure 3: one plot per circumstance and tail with the TPR of each group across models
# graph_01-10: lower tail, graph_11-20: upper tail (circumstances sorted by IO weight)
if use_random_forest:
    model_cols = ["Model1_t", "Model2_t", "Model_X_t"]
    model_labels = [model_rename[m] for m in model_cols]
    variables = recall_df.index.get_level_values('Variable').unique()

    # Group colors (different from the model colors of Figures 1-2); dark vs light so that,
    # together with the markers, the groups remain distinguishable in black and white
    unprot_color = "#6250d6"
    prot_color = "#eda100"

    graph_index = 1
    for segment in ['Lower_tail', 'Upper_tail']:
        for var in variables:
            unprot = recall_df.loc[(var, 'Unprotected_Recall'), [(segment, m) for m in model_cols]].values
            prot = recall_df.loc[(var, 'Protected_Recall'), [(segment, m) for m in model_cols]].values

            fig, ax = plt.subplots(figsize=(10, 3))

            ax.plot(model_labels, unprot, marker='o', color=unprot_color, linestyle='-',
                    linewidth=4, markersize=18)
            ax.plot(model_labels, prot, marker='^', color=prot_color, linestyle='-',
                    linewidth=4, markersize=18)

            ax.set_ylabel('TPR', fontsize=28)
            ax.set_ylim(0.3, 0.65)
            yticks = np.arange(0.30, 0.651, 0.10)
            ax.set_yticks(yticks)
            ax.set_yticklabels([f"{tick:.2f}" for tick in yticks], fontsize=25)
            ax.set_xticks(range(len(model_labels)))
            ax.set_xticklabels(model_labels, fontsize=28)

            for spine in ax.spines.values():
                spine.set_linewidth(2.5)

            ax.tick_params(axis='x', pad=20)

            plt.tight_layout()
            plt.savefig(f'graph_{graph_index:02d}.png', dpi=300)
            plt.close(fig)
            graph_index += 1

    # Common legend saved as a separate image
    fig, ax = plt.subplots(figsize=(6, 2))
    line_unprot, = ax.plot([], [], marker='o', linestyle='-', color=unprot_color, label='Unprotected')
    line_prot, = ax.plot([], [], marker='^', linestyle='-', color=prot_color, label='Protected')
    ax.axis('off')
    legend = ax.legend(handles=[line_unprot, line_prot], loc='center', ncol=2, frameon=False, fontsize=20)
    # Crop the image to the legend itself so that it takes up as little vertical space as possible
    fig.canvas.draw()
    legend_bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig('legend.png', dpi=300, bbox_inches=legend_bbox, pad_inches=0.02)
    plt.close(fig)

