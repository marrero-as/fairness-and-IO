import copy
import pandas as pd
from sklearn.model_selection import KFold
import statsmodels.api as sm
import numpy as np
import os
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
import matplotlib as mpl

# Load data
data = pd.read_csv('ULL_panel_data.csv', sep=',')

# Select only the variables we want to work for
data = data[['id_student_16_19', 'score_MAT', 'score_MAT3', 'a1',
             'mother_education', 'father_education', 'mother_occupation', 'father_occupation', 
             'inmigrant_second_gen', 'start_schooling_age', 'books', 'f12a', 'public_private', 
             'capital_island', 'd14']]

# Drop observations with missing data in any of the variables that we will use in the models
    # Here, synthetic data methods can be used instead to fill in missing values
missing_columns = ['score_MAT3', 'a1', 'mother_education', 'father_education',
    'mother_occupation', 'father_occupation', 'inmigrant_second_gen',
    'start_schooling_age', 'books', 'f12a', 'public_private',
    'capital_island', 'd14']

data = data.dropna(subset=missing_columns)

# Generate quartiles of scores in sixth grade
data['scores_MATq'] = pd.qcut(data['score_MAT'], 4, labels=["1", "2", "3","4"])
data['scores_MATq'] = data['scores_MATq'].astype(int)


# Some data corrections 
data['d14'] = data['d14'].apply(lambda x: 1 if x == 1 else 0) # Variable d14 top category(4) is the protected group (more than 50% of teachers change school), so the results must be inverted

# Rename the variables according to the names in the paper
data = data.rename(columns={
    'id_student_16_19': 'student_id',
    'a1': 'gender',
    'inmigrant_second_gen': 'immigrant_status',
    'start_schooling_age': 'age_at_enrollment',
    'books': 'books_in_home',
    'f12a': 'adults_reading_books',
    'public_private': 'school_ownership',
    'capital_island': 'geographic_area',
    'd14': 'teacher_transfer_rate',
})

data.to_excel("data.xlsx", index=False)

# Now we define protected (prot) and unprotected (unprot) groups for each circumstance
def define_protected_groups(data):

    protected_conditions = {
        # Variables with 5 levels (top_level=5 is unprotected)
        'adults_reading_books': data['adults_reading_books'] != 5,
        
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
        'school_ownership': data['school_ownership'] != 2, #In this case 1 is the protected group. Public school
        'geographic_area': data['geographic_area'] != 1,
        'gender': data['gender'] != 1,
        'teacher_transfer_rate': data['teacher_transfer_rate'] != 1
    }
    
    return protected_conditions

# Variables for the models
Y_t_1 = "score_MAT3"
C = data[["gender", "immigrant_status", "age_at_enrollment", "mother_education", "father_education", "mother_occupation", "father_occupation", "books_in_home", "adults_reading_books", "school_ownership", "teacher_transfer_rate", "geographic_area"]]
Circumstances = ["gender", "immigrant_status", "age_at_enrollment", "mother_education", "father_education", "mother_occupation", "father_occupation", "books_in_home", "adults_reading_books", "school_ownership", "teacher_transfer_rate", "geographic_area"]

# Dummy variables (all variables C are categorical variables)
dummy_variables = pd.get_dummies(C ,columns = Circumstances ,drop_first = True)

# Get the KFold object from the package scikit-learn, which will let me to do splits
kf = KFold(n_splits=5)
# I do a copy and renaming of the data
# From now on, I will always refer to complete_data for the whole dataset
complete_data = copy.deepcopy(data)

use_random_forest = True
file_suffix = "rf" if use_random_forest else "ols"

# This is the loop that iterates 5 times (one for each cross-validation fold)
with pd.ExcelWriter(f"Results_{file_suffix}.xlsx", engine='openpyxl') as writer:
    all_results = pd.DataFrame()
    all_recall_results = []
 
    for fold_index, (train_indeces, test_indeces) in enumerate(kf.split(complete_data)):
        # I just get different splits of the same data structures you were using
        train_data, test_data = complete_data.iloc[train_indeces, :], complete_data.iloc[test_indeces, :]
        train_dummies, test_dummies = dummy_variables.iloc[train_indeces], dummy_variables.iloc[test_indeces]

        # I do the same (getting the different splits) but with the data_combined structure
        # Join Y_t_1 + C
        train_combined = pd.concat([train_data[Y_t_1], train_dummies], axis=1)
        test_combined = pd.concat([test_data[Y_t_1], test_dummies], axis=1)

        # I want to use "data" for the results,
        data = copy.deepcopy(test_data)

        if use_random_forest:

            # Model 1
            model1 = RandomForestRegressor(n_estimators=100, random_state=42, bootstrap= True, oob_score=True, min_samples_split=0.02 , min_samples_leaf=0.01, max_depth=10  )
            model1.fit(train_combined, train_data["score_MAT"])
            data['model1_pred'] = model1.predict(test_combined)

            # Model 2
            model2 = RandomForestRegressor(n_estimators=100, random_state=42, bootstrap= True, oob_score=True, min_samples_split=0.02 , min_samples_leaf=0.01, max_depth=10  )
            model2.fit(train_data[[Y_t_1]], train_data["score_MAT"])
            data['model2_pred'] = model2.predict(test_data[[Y_t_1]])

            # Model 3
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42, bootstrap= True, oob_score=True, min_samples_split=0.02 , min_samples_leaf=0.01, max_depth=10  )
            rf_model.fit(train_data[Circumstances], train_data[Y_t_1])

            # First step
            train_data["Y_t_1_hat"] = rf_model.predict(train_data[Circumstances])
            train_data["ν_hat"] = train_data[Y_t_1] - train_data["Y_t_1_hat"]

            test_data["Y_t_1_hat"] = rf_model.predict(test_data[Circumstances])
            test_data["ν_hat"] = test_data[Y_t_1] - test_data["Y_t_1_hat"]

            data["Y_t_1_hat"] = test_data["Y_t_1_hat"]
            data["ν_hat"] = test_data["ν_hat"]

            # Second step
            rf_model2 = RandomForestRegressor(n_estimators=100, random_state=42, bootstrap= True, oob_score=True, min_samples_split=0.02 , min_samples_leaf=0.01, max_depth=10  )
            rf_model2.fit(train_data[["Y_t_1_hat", "ν_hat"]], train_data["score_MAT"])
            data["model3_pred"] = rf_model2.predict(test_data[["Y_t_1_hat", "ν_hat"]])

            # Prediction exclusively of circumstances
            test_data_circum = test_data[["Y_t_1_hat"]].copy()
            test_data_circum["ν_hat"] = 0
            data['model3_pred_circum'] = rf_model2.predict(test_data_circum)

            # Prediction exclusively of residual
            mean_circu = data['Y_t_1_hat'].mean()
            data['model3_pred_X'] = rf_model2.predict(test_data.assign(Y_t_1_hat=mean_circu)[["Y_t_1_hat", "ν_hat"]])
        else:
            # Model 1
            train_combined = train_combined.apply(pd.to_numeric, errors='coerce').astype("float64")
            X = np.asarray(train_combined, dtype=np.float64)
            y = np.asarray(train_data["score_MAT"], dtype=np.float64)
            model1 = sm.OLS(y, sm.add_constant(X)).fit()
            print(model1.summary())
            data["model1_pred"] = model1.predict(sm.add_constant(test_combined))

            # Model 2
            model2 = sm.OLS(train_data["score_MAT"], sm.add_constant(train_data[Y_t_1])).fit()
            print(model2.summary())
            data["model2_pred"] =  model2.predict(sm.add_constant(test_data[Y_t_1]))

            # Model 3
            train_dummies = train_dummies.apply(pd.to_numeric, errors='coerce').astype("float64")
            X = np.asarray(train_dummies, dtype=np.float64)
            y = np.asarray(train_data["score_MAT3"], dtype=np.float64)
            model3 = sm.OLS(y, sm.add_constant(X)).fit()
            print(model3.summary())

            # First step
            train_data['Y_t_1_hat'] = model3.fittedvalues
            train_data['ν_hat'] = model3.resid
            test_data["Y_t_1_hat"] = model3.predict(sm.add_constant(test_dummies))
            test_data["ν_hat"] = test_data[Y_t_1] - test_data["Y_t_1_hat"]
            data["Y_t_1_hat"] = test_data["Y_t_1_hat"]
            data["ν_hat"] = test_data["ν_hat"]

            # Second step
            model4 = sm.OLS(train_data["score_MAT"], sm.add_constant(train_data[["Y_t_1_hat", "ν_hat"]])).fit()
            print(model4.summary())
            data["model3_pred"] = model4.predict(sm.add_constant(test_data[["Y_t_1_hat", "ν_hat"]]))

            # Prediction exclusively of circumstances
            data['model3_pred_circum'] = model4.params['const'] + model4.params['Y_t_1_hat'] * data['Y_t_1_hat']
            # Prediction exclusively of residual
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
        
        # --- Calculate accuracy ---
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

        # Recall for the different segments of the distribution and for protected and unprotected groups of the circumstances
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
    # Save results
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

            # Compute ratio
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

    # Custom variable order
    order_custom = [
        "father_education",
        "books_in_home",
        "mother_education",
        "age_at_enrollment",
        "teacher_transfer_rate",
        "father_occupation",
        "gender",
        "mother_occupation",
        "geographic_area",
        "immigrant_status",
        "adults_reading_books",
        "school_ownership",     
        ]
    
    new_index = pd.MultiIndex.from_tuples(
    [(var, group)
     for var in order_custom
     for group in ['Unprotected_Recall', 'Protected_Recall']],
    names=['Variable', 'Group']
        )
    
    new_index = new_index.intersection(recall_with_ratios.index)

    # Reindex
    recall_with_ratios = recall_with_ratios.reindex(new_index)

    # Export
    recall_with_ratios.to_excel(writer, sheet_name='TPR_Summary')

# ==============================================
# PLOTS
# ==============================================
filename = f"Results_{file_suffix}.xlsx"
recall_df = pd.read_excel(filename, sheet_name='TPR_Summary', header=[0, 1], index_col=[0, 1])

variable_rename = {
    "father_education": "Father educ.",
    "books_in_home": "Nº books",
    "mother_education": "Mother educ.",
    "age_at_enrollment": "Age at enroll.",
    "teacher_transfer_rate": "Teach.trans.rate",
    "school_ownership": "School own.",
    "father_occupation": "Father occup.",
    "adults_reading_books": "Ad.read.books",
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

model_colors = {
    "Model1_t": "#FF0000",  
    "Model2_t": "#0000FF",  
    "Model_X_t": "#00AA00"  
}

model_styles = {
    "Model1_t": {
        "linestyle": "-",
        "marker": "o"    
    },
    "Model2_t": {
        "linestyle": "--",
        "marker": "s"    
    },
    "Model_X_t": {
        "linestyle": "-",
        "marker": "^"    
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

fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)

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
            linewidth=2.5,
            markersize=7,
            label=legend_label,
            color="black"   
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

# Save panel figure
panel_filename = f"TRP_Ratios_Panels_{file_suffix}.png"
plt.savefig(panel_filename, dpi=300)
plt.close()
