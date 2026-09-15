# =============================================================================
# 🌸 IRIS FLOWER CLASSIFICATION - BEGINNER-FRIENDLY ML PROJECT
# =============================================================================
# Goal: Predict iris species using petal/sepal measurements
# Skills practiced: data loading, exploration, preprocessing, training, evaluation
# Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn
# =============================================================================

# -------------------------
# STEP 1: IMPORT LIBRARIES
# -------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')  # Keep output clean

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (8, 5)

print("✅ Libraries imported successfully!\n")


# -------------------------
# STEP 2: LOAD & EXPLORE DATA
# -------------------------
print("🔍 Loading Iris dataset...")
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Quick overview
print(f"📊 Dataset shape: {df.shape} (150 samples, 6 columns)")
print(f"\n📋 Columns: {list(df.columns)}")
print(f"\n🎯 Target classes: {df['species'].value_counts().to_dict()}")
print(f"\n🔎 First 3 rows:\n{df.head(3)}")

# Check data quality
print(f"\n⚠️  Missing values: {df.isnull().sum().sum()}")  # Should be 0

# Visual exploration
print("\n📈 Generating exploration plots...")

# Plot 1: Class distribution
plt.figure()
sns.countplot(x='species', data=df, palette='Set2')
plt.title('Iris Species Distribution (Balanced!)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Plot 2: Feature relationships (pairplot - shows why this problem is learnable)
sns.pairplot(df, hue='species', vars=iris.feature_names, 
             plot_kws={'alpha': 0.6, 's': 40}, corner=True)
plt.suptitle('Feature Relationships by Species', y=1.02, fontsize=14)
plt.tight_layout()
plt.show()

print("✅ Data exploration complete!\n")


# -------------------------
# STEP 3: PREPROCESS DATA
# -------------------------
print("⚙️  Preprocessing data...")

# Prepare features (X) and target (y)
X = df[iris.feature_names]  # 4 numerical features
y = df['target']            # Already numeric (0, 1, 2) - no encoding needed!

# Scale features (important for Logistic Regression & KNN)
# Trees don't need scaling, but scaling doesn't hurt and helps compare algorithms
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"✅ Features scaled (mean≈0, std≈1 for each)")
print(f"   Example - Sepal Length: original range [{X['sepal length (cm)'].min():.1f}, {X['sepal length (cm)'].max():.1f}] cm")
print(f"                     scaled range [{X_scaled[:, 0].min():.2f}, {X_scaled[:, 0].max():.2f}] (unitless)\n")


# -------------------------
# STEP 4: TRAIN/TEST SPLIT
# -------------------------
print("✂️  Splitting data: 80% train, 20% test...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,          # 20% held out for final evaluation
    random_state=42,        # Reproducible results
    stratify=y              # Preserve class balance in both sets
)

print(f"📚 Training set: {X_train.shape[0]} samples")
print(f"🧪 Testing set:  {X_test.shape[0]} samples")
print(f"✅ Split complete - model will learn on train, be evaluated on unseen test data\n")


# -------------------------
# STEP 5: TRAIN MODEL
# -------------------------
print("🤖 Training Logistic Regression model...")

model = LogisticRegression(
    max_iter=200,           # Ensure convergence
    random_state=42,
    multi_class='ovr'       # One-vs-Rest for 3 classes
)

model.fit(X_train, y_train)
print("✅ Model trained successfully!")

# Show what the model learned (feature importance via coefficients)
coef_df = pd.DataFrame({
    'feature': iris.feature_names,
    'avg_coefficient': model.coef_.mean(axis=0)
}).sort_values('avg_coefficient', key=abs, ascending=False)

print(f"\n🔑 Top predictive features (by coefficient magnitude):")
for _, row in coef_df.iterrows():
    print(f"   • {row['feature']}: {row['avg_coefficient']:.3f}")
print()


# -------------------------
# STEP 6: EVALUATE MODEL
# -------------------------
print("📊 Evaluating model on TEST set (unseen data)...")

# Make predictions
y_pred = model.predict(X_test)

# Calculate key metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n🎯 Performance Metrics:")
print(f"   Accuracy:  {acc:.3f}  ← % of correct predictions")
print(f"   Precision: {prec:.3f}  ← Of predicted class X, how many were truly X?")
print(f"   Recall:    {rec:.3f}  ← Of actual class X, how many did we find?")
print(f"   F1-Score:  {f1:.3f}  ← Balanced measure of precision & recall")

# Confusion Matrix visualization
plt.figure()
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Detailed report
print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("✅ Evaluation complete!\n")


# -------------------------
# STEP 7: COMPARE ALGORITHMS (IMPROVEMENT #1)
# -------------------------
print("🚀 BONUS: Comparing multiple algorithms with cross-validation...")

models_to_compare = {
    'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'K-Nearest Neighbors (k=3)': KNeighborsClassifier(n_neighbors=3)
}

print(f"\n📈 5-Fold Cross-Validated Accuracy (more reliable than single split):")
results = {}
for name, clf in models_to_compare.items():
    scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')
    results[name] = scores.mean()
    print(f"   • {name:25s}: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Highlight best performer
best_model = max(results, key=results.get)
print(f"\n🏆 Best performer: {best_model} ({results[best_model]:.3f} accuracy)")
print("💡 Tip: Try the winning model for your final version!\n")


# -------------------------
# STEP 8: HYPERPARAMETER TUNING (IMPROVEMENT #2)
# -------------------------
print("🔧 BONUS: Tuning K-Nearest Neighbors hyperparameters...")

param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11]}
grid = GridSearchCV(
    KNeighborsClassifier(), 
    param_grid, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1  # Use all CPU cores
)
grid.fit(X_train, y_train)

print(f"\n✅ Best k value: {grid.best_params_['n_neighbors']}")
print(f"✅ Best cross-validated accuracy: {grid.best_score_:.3f}")
print(f"💡 You can now retrain KNN with n_neighbors={grid.best_params_['n_neighbors']} for optimal performance\n")


# -------------------------
# STEP 9: PREDICT NEW DATA (REAL-WORLD USAGE)py
# -------------------------
print("🔮 Demo: Predicting a new, unseen flower...")

# Example new measurement (in cm) - let's predict its species
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])  # Likely setosa
new_flower_scaled = scaler.transform(new_flower)
prediction = model.predict(new_flower_scaled)
prediction_proba = model.predict_proba(new_flower_scaled)[0]

print(f"\n🌸 New flower measurements: {new_flower[0]} cm")
print(f"🎯 Predicted species: {iris.target_names[prediction[0]].upper()}")
print(f"📊 Confidence by class:")
for i, class_name in enumerate(iris.target_names):
    print(f"   • {class_name:10s}: {prediction_proba[i]:.1%}")

print("\n" + "="*70)
print("🎉 PROJECT COMPLETE! You've built a full ML classification pipeline:")
print("   ✓ Loaded & explored data")
print("   ✓ Preprocessed features") 
print("   ✓ Split train/test properly")
print("   ✓ Trained & evaluated a model")
print("   ✓ Compared algorithms & tuned hyperparameters")
print("   ✓ Made predictions on new data")
print("="*70)
print("\n💡 Next steps you could try:")
print("   • Save model: joblib.dump(model, 'iris_model.pkl')")
print("   • Load custom CSV: pd.read_csv('your_data.csv')")
print("   • Add more features: e.g., petal_area = length * width")
print("   • Deploy as API: Flask/FastAPI wrapper around predict()")
print("\n✨ Remember: Great ML is iterative. Start simple, then refine!")