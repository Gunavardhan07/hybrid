import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay

st.title("Hybrid Stacking Model for Crowdfunding Prediction")
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of your data", df.head())

    target_col = st.selectbox("Pick your target column", df.columns)
    df = df.fillna(0)
    y = df[target_col].astype(int)
    X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)

    stratify_param = y if y.nunique() > 1 and y.value_counts().min() > 1 else None
    if stratify_param is None:
        st.warning("Target has one class or very few samples. Stratification turned off.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=stratify_param, random_state=42)

    base_models = [
        ('lr', LogisticRegression(solver='liblinear')),
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
        ('xgb', XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
            eval_metric='logloss', random_state=42
        ))
    ]

    stack_model = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(solver='liblinear'),
        cv=5,
        stack_method='predict_proba'
    )

    stack_model.fit(X_train, y_train)
    y_pred = stack_model.predict(X_test)
    y_pred_proba = stack_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba) if y.nunique() > 1 else 0.0

    st.subheader("Metrics")
    st.dataframe(pd.DataFrame({
        "Accuracy":[acc*100], "Precision":[prec*100],
        "Recall":[rec*100], "F1-score":[f1*100], "ROC-AUC":[auc]
    }).style.format("{:.2f}"))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=stack_model.classes_)
    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=stack_model.classes_).plot(ax=ax_cm, cmap="Blues", colorbar=False)
    st.pyplot(fig_cm)

    if y.nunique() > 1:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        ax_roc.plot([0,1],[0,1],"k--")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)

    st.success("Hybrid model trained successfully!")
else:
    st.info("Upload a CSV to get started.")
