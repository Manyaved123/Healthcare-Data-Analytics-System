import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

FEATURES = ['Age', 'BMI', 'BloodPressure', 'Glucose', 'Insulin']

# Load dataset
data = pd.read_csv("dataset.csv")
X = data[FEATURES]
y = data['Outcome']

model = RandomForestClassifier(n_estimators=200, random_state=42)

# With only 15 rows, a single 80/20 split tests on just 3 patients, so the
# accuracy would be mostly luck. 5-fold stratified cross-validation uses
# every row for testing exactly once and keeps the class balance in each fold.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv)
print(f"5-fold cross-validated accuracy: {scores.mean():.0%} (+/- {scores.std():.0%})")

# Which indicators matter most?
model.fit(X, y)
importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nFeature importance (Random Forest):")
for name, value in importance.items():
    print(f"  {name:<14} {value:.2f}")
