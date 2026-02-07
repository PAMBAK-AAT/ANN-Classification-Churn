







# import streamlit as st
# import numpy as np
# import tensorflow as tf
# import sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
# import pandas as pd
# import pickle

# model = tf.keras.models.load_model('regression_model.h5')

# with open('label_encoder_gender.pkl', 'rb') as file:
#     label_encoder_gender = pickle.load(file)

# with open('onehot_encoder_geo.pkl', 'rb') as file:
#     onehot_encoder_geo = pickle.load(file)

# with open('scaler.pkl', 'rb') as file:
#     scaler = pickle.load(file)


# st.title('Estimated Salary prediction')

# #User input
# geography = st.selectbox('Geography', onehot_encoder_geo.categories_[1])
# gender = st.selectbox('Gender', label_encoder_gender.classes_)
# age = st.slider('Age', 18, 92)
# balance = st.number_input('Balance')
# credit_score = 
# exited = st.selectbox('Exited', [0, 1])
# tenure = 
# num_products = 
# has_cr_card = 
# is_active_member = 

# #Prepare the input data
# input_data = pd.DataFrame({

# })

# # One hot encode the geography
# geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

# ## We concatenate numerical and one-hot encoded categorical features to create a single
# #  feature matrix. Resetting the index prevents misalignment during concatenation. We then apply the same scaler used 
# # during training to ensure consistency, since neural networks are sensitive to feature scale.
# input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded], axis = 1)
# input_data_scaled = scaler.transform(input_data)
# prediction =  model.predict(input_data_scaled)

# predicted_salary = prediction[0][0]

# st.write(f'Predicted Salary: ${predicted_salary:.2f}')








import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# Page configuration
st.set_page_config(
    page_title="Estimated Salary Prediction",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (same style as churn app)
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #43cea2 0%, #185a9d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #43cea2 0%, #185a9d 100%);
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border: none;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(67, 206, 162, 0.3);
    }
    .result-card {
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    margin-top: 1rem;

    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    border: 2px solid #00e5ff;
    color: #ffffff;
            
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);

    }
</style>
""", unsafe_allow_html=True)

# Load model and artifacts
@st.cache_resource
def load_artifacts():
    model = load_model("regression_model.h5")
    with open("label_encoder_gender.pkl", "rb") as f:
        label_encoder_gender = pickle.load(f)
    with open("onehot_encoded_geo.pkl", "rb") as f:
        onehot_encoder_geo = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, label_encoder_gender, onehot_encoder_geo, scaler

model, label_encoder_gender, onehot_encoder_geo, scaler = load_artifacts()

# Header
st.markdown('<h1 class="main-header">💼 Estimated Salary Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">ANN Regression Model | Predict customer salary using AI</p>', unsafe_allow_html=True)
st.markdown("---")

# Input layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 👤 Personal Information")
    geography = st.selectbox("🌍 Geography", onehot_encoder_geo.categories_[0])
    gender = st.selectbox("👫 Gender", label_encoder_gender.classes_)
    age = st.slider("🎂 Age", 18, 92, 35)

with col2:
    st.markdown("### 💳 Financial Details")
    credit_score = st.number_input("📊 Credit Score", min_value=300, max_value=850, value=650)
    balance = st.number_input("💰 Account Balance ($)", min_value=0.0, value=50000.0, format="%.2f")

st.markdown("---")

col3, col4 = st.columns(2)

with col3:
    st.markdown("### 📋 Account Information")
    tenure = st.slider("📅 Tenure (Years)", 0, 10, 5)
    num_products = st.slider("📦 Number of Products", 1, 4, 2)

with col4:
    st.markdown("### ✅ Status")
    has_cr_card = st.selectbox("💳 Has Credit Card", ["Yes", "No"])
    is_active_member = st.selectbox("🔔 Is Active Member", ["Yes", "No"])
    exited = st.selectbox("🚪 Previously Exited", [0, 1])

has_cr_card_val = 1 if has_cr_card == "Yes" else 0
is_active_member_val = 1 if is_active_member == "Yes" else 0
gender_encoded = label_encoder_gender.transform([gender])[0]

st.markdown("---")

# Predict button
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_button = st.button("🔮 Predict Estimated Salary", use_container_width=True)

if predict_button:
    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Gender": [gender_encoded],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_products],
        "HasCrCard": [has_cr_card_val],
        "IsActiveMember": [is_active_member_val],
        "Exited": [exited]
    })

    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
    )

    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
    input_data_scaled = scaler.transform(input_data)

    with st.spinner("🔄 Predicting salary..."):
        prediction = model.predict(input_data_scaled)
        predicted_salary = prediction[0][0]

    st.markdown("---")
    st.markdown("## 📈 Prediction Result")

    st.markdown(f"""
    <div class="result-card">
        <h2>💵 Estimated Annual Salary</h2>
        <h1>${predicted_salary:,.2f}</h1>
    </div>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🧠 Model Information")
    st.markdown("""
    - **Model Type:** ANN Regression  
    - **Framework:** TensorFlow / Keras  
    - **Input Features:** 11  
    - **Output:** Continuous Salary Value  
    """)

    st.markdown("---")
    st.markdown("### 📋 Features Used")
    st.markdown("""
    - Credit Score  
    - Geography (One-Hot)  
    - Gender  
    - Age  
    - Tenure  
    - Balance  
    - Number of Products  
    - Credit Card  
    - Active Member  
    - Exited  
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 1rem;">
    <p>© 2024 Estimated Salary Prediction | Powered by ANN Regression</p>
</div>
""", unsafe_allow_html=True)




