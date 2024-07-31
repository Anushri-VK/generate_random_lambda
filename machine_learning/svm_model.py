import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the generated data from the JSON file
filename = 'lambda_function_data.json'
with open(filename, 'r') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Select the features and target variable
X = df[["execution_time_ms", "memory_usage_mb", "invocation_count", "error_rate", "cold_start_count"]]
y = df["faulty"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM model
clf = SVC(kernel='linear', random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Non-Faulty", "Faulty"], output_dict=True)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Non-Faulty", "Faulty"]))

# Save the trained model to a .pt file
joblib.dump(clf, 'svm_model.pt')

# Plot the metrics
def plot_metrics(report):
    metrics = ['precision', 'recall', 'f1-score']
    classes = ['Non-Faulty', 'Faulty']
    values = [[report[cls][metric] for cls in classes] for metric in metrics]
    
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, metric in enumerate(metrics):
        sns.barplot(x=classes, y=values[i], ax=ax[i])
        ax[i].set_ylim(0, 1)
        ax[i].set_title(metric.capitalize())
        ax[i].set_ylabel(metric.capitalize())
    
    plt.suptitle('Classification Metrics')
    plt.show()

plot_metrics(report)

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

plot_confusion_matrix(y_test, y_pred, ['Non-Faulty', 'Faulty'])
