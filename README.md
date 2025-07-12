# 🚢 Titanic Survival Prediction using Machine Learning

This project aims to predict the survival of passengers aboard the Titanic using classic machine learning models. It uses the publicly available Titanic dataset from Kaggle, which includes features such as passenger class, sex, age, fare, and more.

🔗 **Live Notebook**: [View on Kaggle](https://www.kaggle.com/code/mrmelvin/titanic-survival-prediction-using-machine-learning)

---

## 📂 Dataset

- **Source**: [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic/)
- Files used:
  - `train.csv` — with survival labels
  - `test.csv` — for prediction
  - `gender_submission.csv` — sample format for submission

---

##  Models Used

- Logistic Regression  
- Decision Tree  
- Random Forest  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)

---

##  Workflow

1. **Data Exploration**
   - Checked for null values
   - Analyzed distributions of survival by gender, class, and fare

2. **Data Cleaning**
   - Filled missing values in `Age`, `Embarked`
   - Dropped `Cabin`, `Ticket`, `Name` for simplicity

3. **Feature Engineering**
   - Created new features like `FamilySize`, `IsAlone`
   - Converted `Sex` and `Embarked` using label encoding

4. **Model Training**
   - Trained multiple ML models
   - Compared accuracy scores

5. **Prediction & Submission**
   - Final predictions made using Random Forest (best performing model)
   - Prepared `submission.csv` file for Kaggle

---

## 📈 Results

- Achieved a Kaggle test score of **0.76**
- Best features influencing survival:
  - `Sex`
  - `Pclass`
  - `Fare`
  - `Age`
  - `FamilySize`

---

## 🛠️ Tech Stack

- Python 3.x
- Jupyter Notebook
- Libraries:
  - `pandas`, `numpy`
  - `matplotlib`, `seaborn`
  - `scikit-learn`

---

## 📁 Project Structure

```
├── titanic_survival_prediction.ipynb  # Main notebook
├── train.csv                          # Training dataset
├── test.csv                           # Test dataset
├── submission.csv                     # Output predictions
├── README.md                          # Project summary
```

---

## 🚀 How to Run Locally

1. Clone the repository or download the notebook.
2. Install required libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
3. Open the Jupyter Notebook and run the cells step-by-step.

---

## 💡 Future Enhancements

- Use ensemble methods like XGBoost or LightGBM  
- Apply hyperparameter tuning using GridSearchCV  
- Use cross-validation for better generalization  
- Add advanced visualization and feature importance plots

---

## 🙋‍♂️ Author

**Vivek Nani**  
📧 Reach out for collaboration or feedback!  
🔗 [Kaggle Notebook](https://www.kaggle.com/code/mrmelvin/titanic-survival-prediction-using-machine-learning)  
🔗 [Kaggle Profile](https://www.kaggle.com/mrmelvin)
