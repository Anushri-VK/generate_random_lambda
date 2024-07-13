import random
import json
import time
import logging
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize S3 client
s3 = boto3.client('s3')

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Generate the data
def generate_lambda_data(num_records):
    data = []
    for _ in range(num_records):
        execution_time_ms = random.randint(1, 15000)  # Execution time between 1 ms and 15000 ms
        memory_usage_mb = random.randint(128, 10240)  # Memory usage between 128 MB and 10240 MB
        invocation_count = random.randint(1, 1000)    # Invocation count between 1 and 1000
        error_rate= round(random.uniform(0, 5), 2)   # Error rate between 0.0% and 5.0%
        cold_start_count= random.randint(0, 10)       # Cold start count between 0 and 10

        # Introduce correlations
        if execution_time_ms > 10000:
            error_rate = min(5.0, error_rate + random.uniform(0, 2))
        if memory_usage_mb > 5000:
            invocation_count = min(10000, invocation_count + random.randint(0, 5000))
        if error_rate > 2:
            cold_start_count = min(10, cold_start_count + random.randint(0, 5))

             # Edge cases for hard-to-recognize faulty data
        if random.random() < 0.05:  # 5% chance to create edge case
            if random.random() < 0.5:
                # Slightly above threshold
                error_rate = 1.01
            else:
                # Slightly below threshold
                memory_usage_mb = 4999
                cold_start_count = 8
                invocation_count = 6999
        
        record = {
            "timestamp": time.time(),
            "execution_time_ms": execution_time_ms,
            "memory_usage_mb": memory_usage_mb,
            "invocation_count": invocation_count,
            "error_rate": error_rate,
            "cold_start_count": cold_start_count
        }

        # Detect if the data is faulty based on given conditions
        if (record["error_rate"] > 1 or
            record["memory_usage_mb"] > 5000 or
            record["cold_start_count"] > 8 or
            record["invocation_count"] > 7000):
            record["faulty"] = True
        else:
            record["faulty"] = False
        data.append(record)
    return data

def save_data_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f"File saved to {filename}")

def upload_to_s3(file_path, bucket_name, object_name):
    try:
        s3.upload_file(file_path, bucket_name, object_name)
        logger.info(f"File uploaded to S3: s3://synthetic-lambda-data/test-data/")
    except NoCredentialsError:
        logger.error("Credentials not available")
    except PartialCredentialsError:
        logger.error("Incomplete credentials")

def lambda_handler(event, context):
    num_records = event.get("num_records", 100)  # Default to 100 if not specified in the event
    filename = "/tmp/lambda_function_data.json"  # Lambda has limited writable storage in /tmp directory
    lambda_data = generate_lambda_data(num_records)
    save_data_to_json(lambda_data, filename)
    logger.info(f"Generated {num_records} records of Lambda function data and saved to {filename}")

    # Upload to S3
    bucket_name = "synthetic-lambda-data"  # Replace with your S3 bucket name
    object_name = "lambda_function_data.json"  # Name of the file in S3
    upload_to_s3(filename, bucket_name, object_name)

    return {
        'statusCode': 200,
        'body': json.dumps(f"Generated {num_records} records and saved to {filename}, uploaded to S3")
    }

if __name__ == "__main__":
    num_records = 1000  # Number of data records to generate
    filename = "lambda_function_data.json"
    lambda_data = generate_lambda_data(num_records)
    save_data_to_json(lambda_data, filename)
    print(f"Generated {num_records} records of Lambda function data and saved to {filename}")


    # Load the generated data for training
    with open(filename, 'r') as f:
        data = json.load(f)


    # Convert to DataFrame
    df = pd.DataFrame(data)

   
   # Select the features and target variable
    X = df[["execution_time_ms", "memory_usage_mb", "invocation_count", "error_rate", "cold_start_count"]]
    y = df["faulty"]

    # Split the data multiple times with different splits and train/test the SVM model
    splits = [
        (0, 700, 700, 1000),
        (200, 900, 0, 200),
        (500, 1200, 0, 500),
        (300, 1000, 0, 300)
    ]

    for split in splits:
        train_start, train_end, test_start, test_end = split
        X_train, X_test = pd.concat([X.iloc[train_start:train_end], X.iloc[test_start:test_end]]), pd.concat([X.iloc[test_end:], X.iloc[:test_start]])
        y_train, y_test = pd.concat([y.iloc[train_start:train_end], y.iloc[test_start:test_end]]), pd.concat([y.iloc[test_end:], y.iloc[:test_start]])

        # Ensure there are no empty test sets
        if X_test.shape[0] == 0 or y_test.shape[0] == 0:
            continue

        # Train the SVM model
        clf = SVC(kernel='linear', random_state=42)
        clf.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=["Non-Faulty", "Faulty"], output_dict=True)

        # Print the results
        print(f"Split {splits.index(split) + 1}: Training from {train_start} to {train_end}, Testing from {test_start} to {test_end}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("Classification Report:")
        for label, metrics in report.items():
            if label in ["Non-Faulty", "Faulty"]:
                print(f"Class {label}:")
                print(f"  Precision: {metrics['precision'] * 100:.2f}%")
                print(f"  Recall: {metrics['recall'] * 100:.2f}%")
                print(f"  F1-score: {metrics['f1-score'] * 100:.2f}%")
                print(f"  Support: {metrics['support']}")
        print("\n==================================================\n")