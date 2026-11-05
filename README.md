# Explainable AI and Model Risk Governance: Logistic Regression vs XGBoost for Credit Risk

**Author:** Tashvir Singh (ST10491030)
**Qualification:** Postgraduate Diploma in Data Analytics, The Independent Institute of Education (The IIE)
**Module:** RPDA8412 Research Project
**Supervisor:** Dr Deborah Oluwadele

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tash108vc/TashvirResearchProject/blob/main/RPDA8412_Annexure_B_Full_Analysis_Code_Tashvir_Singh.ipynb)

This repository contains the complete analysis code for my research project (Annexure B of the report). The study compares an interpretable benchmark (**Logistic Regression**) with a machine learning model (**XGBoost**) for credit risk classification. The comparison covers predictive performance, cost-sensitive decisions and the **stability of SHAP explanations**, all viewed through the lens of model risk management (MRM) and BCBS 239.

---

## Research questions

| # | Question |
|---|---|
| RQ1 | Does XGBoost deliver a material, consistent improvement in predictive performance over Logistic Regression? |
| RQ2 | Which features drive XGBoost's credit risk predictions according to SHAP, and are they consistent with credit intuition? |
| RQ3 | Are SHAP feature-importance rankings stable when the model is retrained on different data splits? |
| RQ4 | How do the two models compare on explainability and governance suitability? |

## Dataset

**Statlog (German Credit Data)** (Hofmann, 1994) from the UCI Machine Learning Repository: [doi.org/10.24432/C5NC77](https://doi.org/10.24432/C5NC77)

- 1,000 loan applicants with 20 attributes (7 numerical, 13 categorical)
- 700 good and 300 bad, a 30% default rate
- No missing values
- The original 5:1 cost matrix: approving a bad applicant costs five times as much as rejecting a good one

The notebook downloads the data automatically with `ucimlrepo` (dataset id 144), so you don't need to download anything. If the UCI service is unavailable, it falls back to an identical copy of `german.data` on GitHub.

## How to run

1. Click **Open in Colab** above, or upload the `.ipynb` file to [Google Colab](https://colab.research.google.com).
2. Choose **Runtime → Run all**.
3. Wait about 2–3 minutes. The required packages (`shap`, `xgboost`, `ucimlrepo`) are installed in the first cell.

All figures are saved as PNG files in the working folder.

To run it locally instead:

```bash
pip install numpy pandas scikit-learn xgboost shap scipy matplotlib seaborn ucimlrepo jupyter
jupyter notebook
```

## Notebook structure

The notebook follows the nine-stage workflow in Table 2 of the report:

| Section | Stage | Output in the report |
|---|---|---|
| 0 | Setup and library versions | Section 3 |
| 1 | Data review and exploratory analysis | Section 4.1, Figure 2 |
| 2 | Target definition (1 = bad / default) | Section 3.5.2 |
| 3–4 | Preprocessing (scaling and one-hot encoding) and a stratified 70/30 split | Section 3.5.3, Section 4.1 |
| 5 | Model training and 5-fold stratified cross-validation | Annexure B, Table B1 |
| 6 | Test-set evaluation (ROC-AUC, Gini, KS, recall, precision, F1) | Tables 5–6, Figures 3 and 5 |
| 7 | Cost-sensitive threshold analysis (5·FN + FP) | Figure 6 |
| 8 | Logistic Regression coefficients and odds ratios | Section 4.6 |
| 9 | SHAP explanations and SHAP stability across 10 resampled runs | Tables 7–8, Figures 4, 7 and 8 |
| 10 | Summary of results by research objective | Table 9 |

## Key results

| Measure | Logistic Regression | XGBoost |
|---|---|---|
| ROC-AUC, primary split (seed 42) | **0.804** | 0.801 |
| Recall (bad class), primary split | **0.767** | 0.644 |
| Expected cost (5·FN + FP), primary split | **165** | 202 |
| Mean ROC-AUC over 10 resampled splits | 0.779 | 0.785 |
| Spearman ρ of importance rankings over 45 run pairs | 0.929 (coefficients) | 0.913 (SHAP) |

- **Performance:** neither model showed a consistent or material advantage. XGBoost was ahead in 6 of the 10 resampled splits, and the mean difference (0.006) was smaller than the run-to-run variation (SD 0.021).
- **Cost:** Logistic Regression's cost-minimising threshold was about 0.40, which cut its expected cost from 165 to 146.
- **Explanations:** SHAP rankings were stable overall (ρ = 0.913 ≥ 0.90). The top two drivers, checking account status and loan purpose, were fixed in every run, and the instability was confined to mid-ranked features.
- **Agreement:** the two models agreed only moderately on which features mattered (ρ = 0.574).

## Reproducibility

- **Random seeds:** seed 42 for the primary split and models, and seeds 0–9 for the ten resampled runs.
- **No data leakage:** preprocessing is fitted on the training data only, and refitted inside every cross-validation fold.
- **Reported results** were produced in Google Colab with these versions:

| Package | Version |
|---|---|
| Python | 3.13.18 |
| NumPy | 2.4.6 |
| pandas | 2.2.3 |
| scikit-learn | 1.6.1 |
| XGBoost | 3.4.1 |
| SHAP | 0.54.1 |
| SciPy | 1.16.3 |

Other library versions, particularly of XGBoost, may change the results slightly in the third decimal place.

## Ethics

This is a secondary-data study using a public, fully anonymised dataset with no human participants. Ethical clearance was granted on 20 August 2026 (reference **CSREC/ST10491030/047**).

## References

- Chen, T. and Guestrin, C., 2016. XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp.785–794.
- Hofmann, H., 1994. *Statlog (German Credit Data)* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77
- Lundberg, S.M. and Lee, S.-I., 2017. A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.
- Lundberg, S.M. et al., 2020. From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2(1), pp.56–67.
- Pedregosa, F. et al., 2011. Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, pp.2825–2830.

---

*Submitted in partial fulfilment of the Postgraduate Diploma in Data Analytics, The IIE, 2026.*
