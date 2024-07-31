import json
import pandas as pd
import joblib
import sys

# Load the trained SVM model
model = joblib.load('svm_model.pt')

# Define a function to predict faulty or non-faulty
def predict_faulty(data):
    df = pd.DataFrame(data)
    X = df[["execution_time_ms", "memory_usage_mb", "invocation_count", "error_rate", "cold_start_count"]]
    predictions = model.predict(X)
    df['prediction'] = predictions
    return df

# Example usage
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python interface.py <input_data_file.json> <output_data_file.json>")
        sys.exit(1)
    
    input_data_file = sys.argv[1]
    output_data_file = sys.argv[2]

    with open(input_data_file, 'r') as f:
        data = json.load(f)

    result_df = predict_faulty(data)
    result = result_df.to_dict(orient='records')

    with open(output_data_file, 'w') as f:
        json.dump(result, f, indent=4)
    
    print(f"Predictions saved to {output_data_file}")
