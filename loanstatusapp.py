import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

model = joblib.load('loan_status_predictor.pkl')
scaler = joblib.load('scaler.pkl')
encoders = joblib.load('label_encoders.pkl')
df = pd.read_csv('loan_train.csv')
df_visualiztion = df.copy()
df.drop(columns=['Loan_ID'], inplace=True)
df_visualiztion.drop(columns=['Loan_ID'], inplace=True)

null_cols = ['Credit_History', 'Self_Employed', 'LoanAmount', 'Dependents', 'Loan_Amount_Term', 'Gender', 'Married']
for col in null_cols:
    df[col] = df[col].fillna(df[col].dropna().mode().values[0])

num_cols = df.select_dtypes('number').columns.to_list()
cat_cols = df.select_dtypes('object').columns.to_list()

if 'Loan_Status' in cat_cols:
    cat_cols.remove('Loan_Status')

for col in cat_cols:
    df[col] = encoders[col].transform(df[col])

credit = df['Credit_History']
loan = df['Loan_Status']
df.drop(columns=['Loan_Status', 'Credit_History'], inplace=True)
credit = credit * 10

scaled = scaler.transform(df)
df_scaled = pd.DataFrame(scaled, columns=df.columns)
df_scaled['Credit_History'] = credit
df_scaled['Loan_Status'] = loan
df_scaled.columns = df_scaled.columns.astype(str)

X = df_scaled.drop(columns=['Loan_Status'])
y = df_scaled['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
y_pred = model.predict(X_test)

def about():
    st.title("💰 Loan Approval Predictor")
    st.markdown('''
        Ever wondered if your loan application might get approved? This Loan Predictor gives you a quick, data-driven estimate based on your details — like income, loan amount, credit history, and more.
                   
         It predicts whether a loan might be approved — just like a mini virtual loan officer (without the paperwork)!
    ''')

    st.header("About the Model")
    st.markdown('''
        This machine learning model was built to predict whether a loan application is likely to be **approved** or **rejected**, based on real-world historical data. It’s designed to mimic the kind of evaluation that lenders typically perform — using numbers, not guesswork.

        ### 🧠 Behind the Scenes
        - **Algorithm:** [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)  
        - **Learning Type:** Supervised Binary Classification  
        - **Purpose:** Predict loan approval (Y) or rejection (N) based on applicant and financial information.

        I chose Logistic Regression because it’s simple, interpretable, and effective for yes/no prediction problems — perfect for a beginner-friendly project like this.

        ### 📋 Dataset Used

        The model was trained using the **[Loan Prediction Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)** from Kaggle. All features from the dataset were used except for Loan_ID, which is just a unique identifier.

        Here’s what I used:

        - `Gender` – Applicant's gender  
        - `Married` – Marital status  
        - `Dependents` – Number of dependents  
        - `Education` – Whether the applicant is a graduate  
        - `Self_Employed` – Employment type  
        - `ApplicantIncome` – Income of the primary applicant  
        - `CoapplicantIncome` – Income from any co-applicant  
        - `LoanAmount` – Requested loan amount  
        - `Loan_Amount_Term` – Repayment period in months  
        - `Credit_History` – Whether the applicant has a history of repaying credit  
        - `Property_Area` – Urban, Semi-urban, or Rural classification of the property

        > By combining both numerical and categorical features, the model gets a well-rounded view of each loan application.
    ''')
    y_pred_decoded = encoders['Loan_Status'].inverse_transform(y_pred)
    report = classification_report(y_test, y_pred_decoded)
    st.header("📄 Classification Report")
    st.code(report, language='text')

    st.header("📊 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred_decoded)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(['Rejected', 'Approved'])
    ax.set_yticklabels(['Rejected', 'Approved'])
    st.pyplot(fig)

def visualization():
    st.header('Data Visualization')
    st.dataframe(df_visualiztion.head())

    st.subheader("**For Numerical features:**")
    feature = st.selectbox("Select Numerical Feature", df_visualiztion.select_dtypes(include=['int64', 'float64']).columns)
    fig, ax = plt.subplots()
    sns.histplot(df_visualiztion[feature], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("**For Categorical features:**")
    feature = st.selectbox("Select Categorical Feature", df_visualiztion.select_dtypes(include=['object']).columns)
    fig, ax = plt.subplots()
    sns.countplot(data=df_visualiztion, x=feature, ax=ax, hue='Loan_Status', palette='plasma')
    plt.grid(axis='y')
    st.pyplot(fig)

def predict():
    st.title("Predict Loan Approval")

    gender = st.selectbox("Gender", ['Male', 'Female'])
    married = st.selectbox("Married?", ['Yes', 'No'])
    dependents = st.selectbox("No. of dependents", ['0', '1', '2', '3+'])
    education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
    employed = st.selectbox("Self Employed?", ['Yes', 'No'])
    area = st.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural'])
    appincome = st.slider("Applicant's Income", 0, 80000, 5000)
    coappincome = st.slider("CoApplicant's Income", 0, 40000, 5000)
    amount = st.slider("Loan Amount(in 1000s)", 0, 700, 180)
    term = st.slider("Loan Term(in months)", 0, 500, 360)
    credit = st.selectbox("Credit History (1.0->Satisfactory, 0.0-> Not satisfactory)", [1.0, 0.0])

    input_dict = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': employed,
        'Property_Area': area,
        'ApplicantIncome': appincome,
        'CoapplicantIncome': coappincome,
        'LoanAmount': amount,
        'Loan_Amount_Term': term,
        'Credit_History': credit
    }

    features = pd.DataFrame([input_dict])

    for col in cat_cols:
        features[col] = encoders[col].transform(features[col])



    credit_hist = features['Credit_History'] * 10
    features = features.drop(columns=['Credit_History'])

    features = features[X.drop(columns=['Credit_History']).columns]

    scaled_features = scaler.transform(features)
    features = pd.DataFrame(scaled_features, columns=features.columns)

    features['Credit_History'] = credit_hist
    features = features[X.columns]

    if st.button("Predict"):
        prediction = model.predict(features)
        if prediction[0] == 1:
            st.success("✅ Predicted: Approved!")
        else:
            st.error("❌ Predicted: Rejected")

pg = st.navigation([
    st.Page(about, title="Welcome to the app!"),
    st.Page(visualization, title="Dataset Visualization"),
    st.Page(predict, title="Let's Predict"),
])
pg.run()
