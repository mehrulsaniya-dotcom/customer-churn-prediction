
import streamlit as st
import pandas as pd
file_path = "European_Bank.csv"
os.path.join(os.path.dirname(__file__),
"European_Bank.csv")
df = pd.read_csv(file_path)
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Bank Churn Analytics", layout="wide")
st.title("Bank Customer Churn Prediction")
st.write("Financial Analytics | Machine Learning Project")

@st.cache_data
def load_data():
    return pd.read_csv("European_Bank.csv")

try:
    df = load_data()
    st.subheader("Dataset Overview")
    st.write("Dataset shape:", df.shape)
    st.dataframe(df.head(10), use_container_width=True)

    targets = [
        c for c in df.columns
        if c.strip().lower() in
        ["exited", "churn", "churned", "attrition", "target"]
    ]

    if not targets:
        st.error(
            "Churn target column not found. "
            "Check the column names in your CSV."
        )
    else:
        target = targets[0]
        data = df.dropna(subset=[target]).copy()
        y = data[target]
        X = data.drop(columns=[target])

        # Remove common customer identifiers
        ids = [
            c for c in X.columns
            if c.strip().lower() in
            ["rownumber", "customerid", "surname", "customer_id", "id"]
        ]
        X = X.drop(columns=ids, errors="ignore")

        if y.nunique() < 2:
            st.error("The target must contain at least two classes.")
        else:
            numeric = X.select_dtypes(include="number").columns.tolist()
            categorical = X.select_dtypes(
                exclude="number"
            ).columns.tolist()

            preprocessing = ColumnTransformer([
                ("numeric", SimpleImputer(strategy="median"), numeric),
                ("categorical", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ]), categorical)
            ])

            model = Pipeline([
                ("preprocessing", preprocessing),
                ("classifier", RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    class_weight="balanced"
                ))
            ])

            stratify = y if y.value_counts().min() >= 2 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2,
                random_state=42, stratify=stratify
            )

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            st.subheader("Model Performance")
            st.metric("Test Accuracy", f"{accuracy:.2%}")
            st.caption(
                "Accuracy is measured on a held-out test set. "
                "It is not a guarantee of real-world performance."
            )

            st.subheader("Sample Churn Predictions")
            results = X_test.copy()
            results["Actual Churn"] = y_test
            results["Predicted Churn"] = predictions
            st.dataframe(results.head(20), use_container_width=True)

except FileNotFoundError:
    st.error(
        "Dataset file not found. Ensure European Bank.csv "
        "is uploaded to the repository root."
    )
except Exception as e:
    st.error(f"Please check the dataset and columns: {e}")
