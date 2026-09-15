
# 🌸 Data Classification Using AI - Project 2

**Internship Project @ Decodelabs**

## 📋 Overview

This project demonstrates the fundamentals of supervised machine learning by building a basic classification model using the famous Iris dataset. The goal is to predict iris flower species based on petal and sepal measurements, showcasing essential data science and machine learning skills.

##  Project Goals

- Build a basic classification model using a small dataset
- Understand the complete machine learning pipeline
- Apply supervised learning techniques for data classification
- Evaluate model performance using appropriate metrics

## ✨ Key Features

- **Data Loading & Exploration**: Comprehensive understanding of dataset structure and characteristics
- **Data Preprocessing**: Handling missing values, feature scaling, and data transformation
- **Train-Test Split**: Proper data partitioning (80/20) for model validation
- **Multiple Classification Algorithms**: 
  - Logistic Regression
  - Decision Tree
  - K-Nearest Neighbors (KNN)
- **Model Evaluation**: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix
- **Hyperparameter Tuning**: Optimization using Grid Search
- **Cross-Validation**: 5-fold cross-validation for robust performance assessment
- **Visualizations**: Interactive plots for data exploration and results interpretation

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical visualizations
- **Scikit-learn** - Machine learning library

## 📦 Installation

1. **Clone or download this project**
   ```bash
   git clone <repository-url>
   cd <project-folder>
   ```

2. **Install required dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

3. **(Optional) Create a virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Mac/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

## 🚀 How to Run

1. **Run the main script**
   ```bash
   python iris_classifier.py
   ```

2. **Or run in Jupyter Notebook**
   ```bash
   jupyter notebook iris_classifier.ipynb
   ```

## 📊 Dataset

**Iris Flower Dataset**
- **Samples**: 150 instances
- **Features**: 4 (Sepal Length, Sepal Width, Petal Length, Petal Width)
- **Classes**: 3 (Setosa, Versicolor, Virginica)
- **Source**: Built-in scikit-learn dataset

## 📈 Project Workflow

1. **Data Loading**: Import Iris dataset using scikit-learn
2. **Exploratory Data Analysis (EDA)**: Understand data distribution and relationships
3. **Data Preprocessing**: Scale features using StandardScaler
4. **Data Splitting**: 80% training, 20% testing with stratification
5. **Model Training**: Train Logistic Regression classifier
6. **Model Evaluation**: Assess performance on test data
7. **Model Comparison**: Compare multiple algorithms
8. **Hyperparameter Tuning**: Optimize KNN using GridSearchCV
9. **Prediction**: Test model on new, unseen data

## 🎓 Learning Outcomes

This project helped develop the following skills:

✅ **Data Handling**: Loading, cleaning, and preprocessing datasets  
✅ **Supervised Learning**: Understanding classification algorithms  
✅ **Model Training**: Fitting models to training data  
✅ **Model Evaluation**: Using metrics to assess performance  
✅ **Data Visualization**: Creating insightful plots and graphs  
✅ **Python Programming**: Applying libraries for ML tasks  
✅ **Best Practices**: Train-test split, cross-validation, hyperparameter tuning

## 📁 Project Structure

```
project-2-data-classification/
│
├── iris_classifier.py          # Main Python script
── README.md                    # Project documentation
├── requirements.txt             # Dependencies (optional)
└── visuals/                     # Generated plots (optional)
    ├── distribution.png
    ├── pairplot.png
    └── confusion_matrix.png
```

## 🔧 Key Components

### Classification Algorithms Used

1. **Logistic Regression**
   - Fast and interpretable
   - Good baseline model
   - Multi-class support (One-vs-Rest)

2. **Decision Tree**
   - Easy to visualize and understand
   - No feature scaling required
   - Captures non-linear relationships

3. **K-Nearest Neighbors (KNN)**
   - Simple and effective
   - Distance-based classification
   - Hyperparameter tuning for optimal k

### Evaluation Matrics

- **Accuracy**: Overall correctness
- **Precision**: Quality of positive predictions
- **Recall**: Ability to find all positive instances
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions

## 📝 Sample Output

```
✅ Model trained successfully!
🎯 Performance Metrics:
   Accuracy:  0.967  ← % of correct predictions
   Precision: 0.967  ← Of predicted class X, how many were truly X?
   Recall:    0.967  ← Of actual class X, how many did we find?
   F1-Score:  0.967  ← Balanced measure of precision & recall

 Best performer: Logistic Regression (0.967 accuracy)
```

