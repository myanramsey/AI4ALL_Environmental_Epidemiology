# [AI4ALL] Environmental Epidemiology
## Linking Agricultural Pesticide Use to Respiratory Health Outcomes in California (2000–2023)

This project investigates whether long-term pesticide exposure contributes to increased COPD (Chronic Obstructive Pulmonary Disease) rates across California counties. We built a full data-engineering and machine-learning pipeline to process pesticide use, demographic confounders, and respiratory hospitalization data. This work was completed as part of the AI4ALL Ignite program, applying epidemiology, machine learning, environmental justice analysis, and responsible AI principles to understand how agricultural practices affect community health.

[<kbd>TEST OUT OUR MODEL ON STREAMLIT</kbd>](https://ai4allenvironmentalepidemiology.streamlit.app/)
---
---<img width="648" height="864" alt="AI4ALL Poster" src="https://github.com/user-attachments/assets/3301ee8f-c8b3-40c6-a060-479a4b92104f" />


# Problem Statement
California is the agricultural engine of the United States, producing most of the nation’s fruits, nuts, and vegetables. However, this productivity relies heavily on pesticide use, which threatens environmental and public health—especially in rural and farmworking communities.

Respiratory illnesses such as asthma and COPD are disproportionately high in regions like the San Joaquin Valley, where air pollution, pesticides, and agricultural emissions co-occur.  

Our project evaluates whether pesticide application intensity, lagged exposure, and cumulative exposure are associated with COPD hospitalization rates from 2000–2023. This analysis aims to support public-health insights, environmental justice efforts, and data-driven policymaking.

---

# Key Results
- Constructed a **clean county-level dataset (2000–2023)** for all CA counties with:
  - Annual COPD hospitalization rates  
  - Pesticide application totals  
  - Demographic confounders (income, education, race, age, etc.)
- Implemented a full pesticide-processing pipeline:
  - Condensed raw pesticide data to county-year applications  
  - Created **lag features** (1-year, 3-year, 5-year)  
  - Created **cumulative exposure features** (multi-year rolling sums/averages)
- Removed counties with heavy missing data: **Sierra, Alpine, Lassen, Modoc**
- Prepared ML-ready datasets and **implemented three key modeling approaches**:
  - **Random Forest Regression**
  - **XGBoost Regression**
  - **Linear Regression**
- Built a framework to compare whether **pesticide exposures** or **demographic confounders** better explain COPD hospitalization rates  
- Ensured every team member can independently follow the analysis workflow in Google Colab

---

# Methodologies

### Data Processing Workflow
- Mounted Google Drive in Google Colab  
- Imported COPD and pesticide datasets  
- Dropped irrelevant pesticide columns:  
  - `SOURCE_FILE`, `APPLICATION_METHOD_DESC`, `APPLICATION_METHOD`,  
    `TYPE_CODE`, `USE_CODE`, `FORM_CODE`, `RECORD_ID`, `COUNTY_CD`, `SITE_CODE`
- Removed counties with too much missing data: **Sierra, Alpine, Lassen, Modoc**
- Aggregated pesticide use to **county-year totals**
- Created:
  - **Lag features** (shifted pesticide use for prior years)
  - **Cumulative features** (rolling sums/means)
- Merged pesticide features with COPD + demographic confounders  
- Validated data for missing values, inconsistencies, and outliers  

### Machine Learning Methods
- Constructed combined feature matrices for ML  
- Implemented three modeling approaches:
  - **Random Forest Regression** (tree-based, nonlinear exposure modeling)  
  - **XGBoost Regression** (gradient-boosted decision trees for high predictive power)  
  - **Linear Regression** (baseline statistical modeling & interpretability)
- Evaluated model performance using R² and MAE  
- Assessed feature importance & explored whether confounders outperform pesticides in explaining COPD rates  
- Conducted environmental justice analysis:
  - Identifying regions with elevated combined environmental and sociodemographic risks  

---

# Data Sources
- **COPD Rate + Confounding Factors (2000–2023)**  
  https://drive.google.com/file/d/1tlRSbGSMwh6MY_-fpaP1Fff1HxoZSsGk/view?usp=drive_link

- **California Pesticide Use Dataset (1989–2016)**  
  https://drive.google.com/file/d/1EBQEWkHmT6WmfZ3YNGcXOgCqLz55WpD5/view?usp=drive_link

- **Project Slide Deck (Contextual Overview)**  
  Included: Proposal Slides.pdf

- Additional supporting datasets (optional when modeling):  
  - ACS Demographics  
  - USDA RUCC for urban/rural classification

---

# Technologies Used
- Python  
- Google Colab  
- pandas  
- NumPy  
- scikit-learn  
- XGBoost  
- Matplotlib / Seaborn  
- Git & GitHub  
- AI4ALL Ignite Program Tools & Support  

---

# Authors
This project was completed in collaboration with:

- **Mya Ramsey**  
  - Email: myaramsey@ufl.edu
  - GitHub: https://github.com/myanramsey
    
- **Akil Creswell**  
  - Email: akilbcwork2@gmail.com
  - GitHub: https://github.com/Acreswe
  
- **Armando Tamayo**  
  - Email: mandoschool1@ucsc.edu  
  - GitHub: https://github.com/MandoBug  

- **Ameya Patkar**  
  - Email: ameyaspatkar@gmail.com
  - GitHub: https://github.com/Ameya-P

- **Heidy Naranjo**  
  - Email: heidynaranjo@brandeis.edu
  - GitHub: https://github.com/nayena/nayena

---
