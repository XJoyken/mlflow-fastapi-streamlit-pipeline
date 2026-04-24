import streamlit as st
import requests
import os
# API_URL = 'http://backend:666/predict'
# API_URL = "http://127.0.0.1:666/predict"
API_URL = os.getenv("API_URL", "http://127.0.0.1:666/predict")

st.set_page_config(
    page_title="German Credit Scoring API", 
    page_icon='random', 
    layout='centered', 
    initial_sidebar_state=200, 
    menu_items= {
        "About": "This is the use of MLFlow, Docker, FastAPI and Streamlit"
    }
)
st.title("German Credit Scoring")
st.write("Fill in the information to get the probability of a loan being issued.")

dict_A1 = {"A11": "< 0 DM", "A12": "0 <= ... < 200 DM", "A13": ">= 200 DM / salary assignments for at least 1 year", "A14": "no checking account"}
dict_A3 = {"A30": "no credits taken / all paid back duly", "A31": "all credits at this bank paid back duly", "A32": "existing credits paid back duly till now", "A33": "delay in paying off in the past", "A34": "critical account / other credits existing"}
dict_A4 = {"A40": "car (new)", "A41": "car (used)", "A42": "furniture/equipment", "A43": "radio/television", "A44": "domestic appliances", "A45": "repairs", "A46": "education", "A47": "vacation", "A48": "retraining", "A49": "business", "A410": "others"}
dict_A6 = {"A61": "< 100 DM", "A62": "100 <= ... < 500 DM", "A63": "500 <= ... < 1000 DM", "A64": ">= 1000 DM", "A65": "unknown / no savings account"}
dict_A7 = {"A71": "unemployed", "A72": "< 1 year", "A73": "1 <= ... < 4 years", "A74": "4 <= ... < 7 years", "A75": ">= 7 years"}
dict_A9 = {"A91": "male: divorced/separated", "A92": "female: divorced/separated/married", "A93": "male: single", "A94": "male: married/widowed", "A95": "female: single"}
dict_A10 = {"A101": "none", "A102": "co-applicant", "A103": "guarantor"}
dict_A12 = {"A121": "real estate", "A122": "building society savings / life insurance", "A123": "car or other", "A124": "unknown / no property"}
dict_A14 = {"A141": "bank", "A142": "stores", "A143": "none"}
dict_A15 = {"A151": "rent", "A152": "own", "A153": "for free"}
dict_A17 = {"A171": "unemployed/unskilled - non-resident", "A172": "unskilled - resident", "A173": "skilled employee / official", "A174": "management / self-employed / highly qualified"}
dict_A19 = {"A191": "none", "A192": "yes, registered under the customers name"}
dict_A20 = {"A201": "yes", "A202": "no"}

header_col1, header_col2 = st.columns(2)
header_col1.subheader("Financial information")
header_col2.subheader("Personal information")

col1, col2 = st.columns(2)

with col1:
    at1 = st.selectbox("Status of existing checking account (A1)", options=list(dict_A1.keys()), placeholder='A11', format_func=lambda x: f"{x}: {dict_A1[x]}")
    at3 = st.selectbox("Credit history (A3)", options=list(dict_A3.keys()), placeholder='A30', format_func=lambda x: f"{x}: {dict_A3[x]}")
    at4 = st.selectbox("Purpose (A4)", options=list(dict_A4.keys()), placeholder='A40', format_func=lambda x: f"{x}: {dict_A4[x]}")
    at5 = st.number_input("Credit amount (A5)", min_value=100, value=20000)
    at6 = st.selectbox("Savings account/bonds (A6)", options=list(dict_A6.keys()), placeholder='A61', format_func=lambda x: f"{x}: {dict_A6[x]}")
    at8 = st.number_input("Installment rate in percentage of disposable income (A8)", min_value=1, max_value=4, value=2)
    at10 = st.selectbox("Other debtors / guarantors (A10)", options=list(dict_A10.keys()), placeholder='A101', format_func=lambda x: f"{x}: {dict_A10[x]}")
    at12 = st.selectbox("Property (A12)", options=list(dict_A12.keys()), placeholder='A121', format_func=lambda x: f"{x}: {dict_A12[x]}")
    at14 = st.selectbox("Other installment plans (A14)", options=list(dict_A14.keys()), placeholder='A141',  format_func=lambda x: f"{x}: {dict_A14[x]}")
    at16 = st.number_input("Number of existing credits at this bank (A16)", min_value=1, max_value=4, value=1)

with col2:
    at2 = st.number_input("Duration in month (A2)", min_value=1, max_value=72, value=12)
    at7 = st.selectbox("Present employment since (A7)", options=list(dict_A7.keys()), placeholder='A71', format_func=lambda x: f"{x}: {dict_A7[x]}")
    at9 = st.selectbox("Personal status and sex (A9)", options=list(dict_A9.keys()), placeholder='A91', format_func=lambda x: f"{x}: {dict_A9[x]}")
    at11 = st.number_input("Present residence since (A11)", min_value=1, max_value=4, value=2)
    at13_age = st.number_input("Age in years (A13)", min_value=18, max_value=100, value=30)
    at15 = st.selectbox("Housing (A15)", options=list(dict_A15.keys()), placeholder='A151', format_func=lambda x: f"{x}: {dict_A15[x]}")
    at17 = st.selectbox("Job (A17)", options=list(dict_A17.keys()), placeholder='A171', format_func=lambda x: f"{x}: {dict_A17[x]}")
    at18 = st.number_input("Number of people being liable to provide maintenance for (A18)", min_value=1, max_value=2, value=1)
    at19 = st.selectbox("Telephone (A19)", options=list(dict_A19.keys()), placeholder='A191', format_func=lambda x: f"{x}: {dict_A19[x]}")
    at20 = st.selectbox("Foreign worker (A20)", options=list(dict_A20.keys()), placeholder='A201', format_func=lambda x: f"{x}: {dict_A20[x]}")

st.markdown('---')

if st.button('Predict', type='primary'):
    features_dict = {
        "Attribute1": at1,
        "Attribute2": at2,
        "Attribute3": at3,
        "Attribute4": at4,
        "Attribute5": at5,
        "Attribute6": at6,
        "Attribute7": at7,
        "Attribute8": at8,
        "Attribute9": at9,
        "Attribute10": at10,
        "Attribute11": at11,
        "Attribute12": at12,
        "Attribute14": at14,
        "Attribute15": at15,
        "Attribute16": at16,
        "Attribute17": at17,
        "Attribute18": at18,
        "Attribute19": at19,
        "Attribute20": at20,
        "age_group": 0 if at13_age < 30 else 1
    }
    payload = {'features': features_dict}

    try:
        with st.spinner("Connect with FastAPI..."):
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            result = response.json()
            prediction = result.get('prediction')

            if prediction == 1:
                st.success("Good credit.")
            else:
                st.error("Bad credit.")
    except requests.exceptions.ConnectionError:
        st.error("Error: unable to connect to FastAPI")
    except Exception as e:
        st.error(f"Error: {e}")