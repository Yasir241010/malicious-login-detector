# train.py
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
from app import generate_synthetic_data

def main():
    df = generate_synthetic_data(5000)
    X = df[['failed_attempts','time_of_login','geo_distance','device_known']]
    y = df['malicious']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    joblib.dump((model, X_train, X_test, y_train, y_test), 'malicious_login_lr.joblib')
    print("Saved malicious_login_lr.joblib")

if __name__ == "__main__":
    main()
