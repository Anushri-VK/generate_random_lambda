import random
import json
import time
import logging
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize S3 client
s3 = boto3.client('s3')

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
    num_records = 100  # Number of data records to generate
    filename = "lambda_function_data.json"
    lambda_data = generate_lambda_data(num_records)
    save_data_to_json(lambda_data, filename)
    print(f"Generated {num_records} records of Lambda function data and saved to {filename}")


    # Load the generated data for training
    with open(filename, 'r') as f:
        data = json.load(f)


    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Separate features and labels
    X = df.drop(columns=["timestamp", "faulty"])
    y = df["faulty"]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Decision Tree model
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred, output_dict=True)

    # Convert classification report to percentage
    for key in report:
        if key not in ["accuracy", "macro avg", "weighted avg"]:
            for metric in report[key]:
                report[key][metric] *= 100

    # Display results
    print(f"Accuracy: {accuracy:.2f}%")
    print("Classification Report:")
    for label, metrics in report.items():
        if label not in ["accuracy", "macro avg", "weighted avg"]:
            print(f"Class {label}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.2f}%")