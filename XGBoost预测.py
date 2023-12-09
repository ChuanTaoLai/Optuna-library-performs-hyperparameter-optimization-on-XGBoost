import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Load the data from Excel
data = pd.read_excel(r'D:\0文献整理\网络入侵检测\KDD99\vif_after_alpha0.04.xlsx')

# Define the hyperparameters
hyperparameters = {'n_estimators': 150, 'max_depth': 5, 'learning_rate': 0.22807851936316503, 'subsample': 0.8337624693024694}

# Split the dataset into training and testing sets
X = data.iloc[:, 0:-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1412)

# Build and train the XGBoost Classifier
# clf = xgb.XGBClassifier(**hyperparameters, random_state=1412)
clf = xgb.XGBClassifier(random_state=1412)
clf.fit(X_train, y_train)

# Predictions on the training set
y_train_pred = clf.predict(X_train)
accuracy_train = accuracy_score(y_train, y_train_pred)

# Predictions on the testing set
y_test_pred = clf.predict(X_test)
accuracy_test = accuracy_score(y_test, y_test_pred)

# Print accuracies
print(f"训练集准确率: {accuracy_train}")
print(f"测试集准确率: {accuracy_test}")
