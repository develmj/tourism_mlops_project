import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Wellness Tourism Package Predictor", layout="centered")

st.title("Wellness Tourism Package Predictor")
st.write("Enter customer details below to predict the likelihood of purchasing the Wellness Package.")

# Load model from Hugging Face
MODEL_REPO = "mjiyer/tourism-wellness-gb-model"
MODEL_FILENAME = "gb_best_model.joblib"

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        repo_type="model"
    )
    return joblib.load(model_path)

model = load_model()

# UI Inputs
def get_user_inputs():
    Age = st.number_input("Age", 18, 100, 35)
    TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    DurationOfPitch = st.number_input("Duration of Pitch (minutes)", 0, 60, 10)
    Occupation = st.selectbox("Occupation", ["Salaried", "Self Employed", "Student", "Freelancer", "Other"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    NumberOfPersonVisiting = st.number_input("Number of People Visiting", 1, 10, 2)
    NumberOfFollowups = st.number_input("Number of Followups", 0, 20, 2)
    ProductPitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Premium", "Super Deluxe", "King"])
    PreferredPropertyStar = st.selectbox("Preferred Property Star Rating", [3, 4, 5])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    NumberOfTrips = st.number_input("Trips per Year", 0, 20, 3)
    Passport = st.selectbox("Passport", [0, 1])
    PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
    OwnCar = st.selectbox("Own Car", [0, 1])
    NumberOfChildrenVisiting = st.number_input("Children Visiting", 0, 5, 0)
    Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    MonthlyIncome = st.number_input("Monthly Income (INR)", 10000, 500000, 50000)

    df = pd.DataFrame([{
        "Age": Age,
        "TypeofContact": TypeofContact,
        "CityTier": CityTier,
        "DurationOfPitch": DurationOfPitch,
        "Occupation": Occupation,
        "Gender": Gender,
        "NumberOfPersonVisiting": NumberOfPersonVisiting,
        "NumberOfFollowups": NumberOfFollowups,
        "ProductPitched": ProductPitched,
        "PreferredPropertyStar": PreferredPropertyStar,
        "MaritalStatus": MaritalStatus,
        "NumberOfTrips": NumberOfTrips,
        "Passport": Passport,
        "PitchSatisfactionScore": PitchSatisfactionScore,
        "OwnCar": OwnCar,
        "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
        "Designation": Designation,
        "MonthlyIncome": MonthlyIncome
    }])

    return df

input_df = get_user_inputs()

if st.button("Predict"):
    probability = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]

    st.subheader(f"Prediction Probability: {probability:.2f}")
    if prediction == 1:
        st.success("Customer is LIKELY to purchase the package.")
    else:
        st.warning("Customer is UNLIKELY to purchase the package.")

