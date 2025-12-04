import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def preprocess_data(input_path, output_dir, target_col='target'):
    """
    Preprocess the data: clean, scale, split.
    """
    print("Starting preprocessing...")
    
    # Load data
    df = pd.read_csv(input_path)
    print(f"Loaded data: {df.shape}")
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split data (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Split data: Train={len(X_train)}, Test={len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"Scaled features")
    
    # Convert back to DataFrames
    train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    train_df[target_col] = y_train.values
    
    test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    test_df[target_col] = y_test.values
    
    # Save processed data
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)
    print(f"Saved train.csv and test.csv to {output_dir}")
    
    # Save scaler
    joblib.dump(scaler, f"{output_dir}/scaler.pkl")
    print(f"Saved scaler.pkl")
    
    return train_df, test_df

if __name__ == "__main__":
    train_df, test_df = preprocess_data(
        input_path="dataraw/dataset.csv",
        output_dir="dataprocessed"
    )
    print("\n Preprocessing complete!")
