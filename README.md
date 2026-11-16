# Replication Code: Public AI and Educational Targeting: Bridging Algorithmic Fairness and Inequality of Opportunity

### Developers
This code has been developed by **Marrero, A.S. (Ángel S. Marrero)** and **Giovanelli, J. (Joseph Giovanelli)**.

### This repository contains the source code to replicate the results of the following paper:
Marrero, A.S., Marrero, G., Bethencourt, C., Giovanelli, J., Calegari, R. "Public AI and Educational Targeting: Bridging Algorithmic Fairness and Inequality of Opportunity".

### Project structure

+ `Data/`: code that builds the input data of the paper (`ULL_panel_data.csv`) from the original dataset.
+ `Replication_results/`: code that generates all the tables and figures of the paper (`code.py`).

### Requirements
All code is written in Python 3. You will need the following packages: pandas, numpy, statsmodels, scikit-learn, matplotlib and openpyxl.

```
pip install pandas numpy statsmodels scikit-learn matplotlib openpyxl
```

### How to reproduce the results in the paper
The files need to be run in order:

1. Download the `original.csv` file from __[Zenodo](https://zenodo.org/records/11171863)__.
2. Run `data_code.py` in the `Data` folder. This code generates `ULL_panel_data.csv`.
3. Copy `ULL_panel_data.csv` to the `Replication_results` folder and run `code.py` there.
4. `code.py` estimates the models either with OLS or with Random Forest, depending on the `use_random_forest` flag (in the *PREDICTIVE MODELS* section). Run it once with `use_random_forest = False` (OLS) and once with `use_random_forest = True` (Random Forest) to obtain all the results of the paper.

### Outputs of `code.py`

| File | Content | Paper |
|---|---|---|
| `Results_IO.xlsx` | Inequality of opportunity (IO) regression on 3rd-grade scores; total inequality, IO and IO ratios (Gini, MLD, variance); relative IO weight of each circumstance | Tables 2 and 3 |
| `Results_ols.xlsx` / `Results_rf.xlsx` | Sheet `Accuracy_Summary`: accuracy of the three models by fold and performance segment. Sheet `TPR_Summary`: True Positive Rates (TPR) and TPR ratios by circumstance, group (protected/unprotected) and segment | Table 4; Tables A1 (OLS) and A2 (RF) |
| `TPR_Ratios_ols.png` / `TPR_Ratios_rf.png` | Equal Opportunity metric (TPR ratio) across the segments of the distribution | Figures 1 (OLS) and 2 (RF) |
| `graph_01.png` ... `graph_20.png`, `legend.png` (RF only) | TPR of protected and unprotected groups across the three models, for the lower (01-10) and upper (11-20) tails | Figure 3 |

Circumstances appear in all tables and figures sorted by their relative IO weight (Table 3).

### Explanation
+ `original.csv` contains a dataset on academic performance of primary and secondary education students from the Canary Islands.
Each row refers to a single student at a given grade (3rd and 6th grades of primary education and 4th grade of secondary education) and academic year (2015-16, 2016-17, 2017-18, 2018-19). Longitudinal data (panel data) is also included: students in 3rd grade of primary education in 2015-16 are sampled again in 6th grade of primary education in 2018-19.
+ The columns of the dataset represent relevant features collected for each student: identifiers, academic performance in various subjects, and responses to questionnaires addressed to the students, their families, their teachers, and the school principals. In the `Data` folder, there is an Excel file (`data_dictionary.xlsx`) where you can find the definitions of each variable and the possible values they can take.
+ `data_code.py` in the `Data` folder takes the complete database (`original.csv`) and retains only the longitudinal data. Since our goal is to predict students' future academic performance using current information, the code merges the students' academic performance in 6th grade with the information from 3rd grade, generating `ULL_panel_data.csv`. Each row in `ULL_panel_data.csv` represents a student who was surveyed in both 3rd grade and 6th grade. The columns include their academic performance in mathematics in 6th grade and information about a set of variables (called circumstances) in 3rd grade.
+ `code.py` in the `Replication_results` folder takes `ULL_panel_data.csv` and:
  1. Estimates inequality of opportunity in 3rd grade and decomposes it by circumstance (Sections 4.1 and 5.1).
  2. Trains the three predictive models of 6th-grade performance (Model 1: past performance and circumstances; Model 2: past performance only; Model 3: past performance purged of circumstances) with OLS and Random Forest, using five-fold cross-validation (Section 4.2).
  3. Computes accuracy and AI fairness metrics (TPR and TPR ratios) for protected and unprotected groups in each segment of the performance distribution (Sections 4.3 and 5.2), and draws the figures of the paper.

### Citation
When using this archive, please cite this paper:
Marrero, A.S., Marrero, G., Bethencourt, C., Giovanelli, J., Calegari, R. "Public AI and Educational Targeting: Bridging Algorithmic Fairness and Inequality of Opportunity".

If you use the original dataset, please cite this paper:
Giovanelli, J., Magnini, M., James, L., Ciatto, G., Marrero, A. S., Borghesi, A., Marrero, G. A., & Calegari, R. (2024). Unfair Inequality in Education: A Benchmark for AI-Fairness Research (1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.11171863

### License
+ Software is distributed under the terms of the __[MIT License](https://opensource.org/licenses/MIT)__.
