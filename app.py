


# import streamlit as st
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import pickle
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder



# ## This block is used to make predictions using our trained model on new data.
# model=load_model("model.h5") # Loads your trained neural network from disk

# ## During training we have map Male->0 and Female->1, so in order to maintain same mapping we have import the encoder
# with open('label_encoder_gender.pkl', 'rb') as file:
#     label_encoder_gender=pickle.load(file)


# with open('onehot_encoded_geo.pkl', 'rb') as file:
#     onehot_encoded_geo=pickle.load(file)

# # During training our model learns on the scaled value not raw ones, so we have to use the same scaler
# with open('scaler.pkl', 'rb') as file:
#     scaler=pickle.load(file)


# ## streamlit app
# st.title('Customer Churn Prediction')

# ## User input
# # onehot_encoded_geo.categories_ : [array(['France', 'Germany', 'Spain'], dtype=object)]
# # categories_[0] : ['France', 'Germany', 'Spain']

# geography =st.selectbox('Geography', onehot_encoded_geo.categories_[0]) # categories_ stores all unique categories learned during training
# gender = st.selectbox('Gender', label_encoder_gender.classes_)
# age = st.slider('Age', 18, 92)
# balance = st.number_input('Balance')
# estimated_salary = st.number_input('Estimated Salary')
# credit_score = st.number_input('Credit Score')
# tenure = st.slider('Tenure', 0, 10)
# num_of_products = st.slider('Number of Products', 1, 4)
# has_cr_card = st.selectbox('Has Credit card', [0, 1])
# is_active_member = st.selectbox('Is Active member', [0, 1])


# ## Prepare the input data
# input_data = pd.DataFrame({
#     'CreditScore': [credit_score],
#     'Gender': [label_encoder_gender.transform([gender])[0]],
#     'Age': [age],
#     'Tenure': [tenure],
#     'Balance': [balance],
#     'NumOfProducts': [num_of_products],
#     'HasCrCard': [has_cr_card],
#     'IsActiveMember': [is_active_member],
#     'EstimatedSalary': [estimated_salary]
# })


# geo_encoded = onehot_encoded_geo.transform([[geography]]).toarray()
# geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoded_geo.get_feature_names_out(['Geography']))

# # Combine one hot encoded columns with input_data
# input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# # Scale the input data
# input_data_scaled = scaler.transform(input_data)

# ## Prediction Churn
# prediction = model.predict(input_data_scaled)
# prediction_prob = prediction[0][0] # For neural networks / Keras / TensorFlow models, predict() always returns a 2D array, even for a single input.

# st.write(f"Churn probability: {prediction_prob:.2f}.")

# if prediction_prob > 0.5:
#     st.write('The customer is likely to churn.')
# else:
#     st.write('The customer is not likely to churn.')







import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
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
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
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
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
    .result-card {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 1rem;
    }
    .churn-high {
        background: linear-gradient(135deg, #ff6b6b20 0%, #ee5a5a20 100%);
        border: 2px solid #ff6b6b;
    }
    .churn-low {
        background: linear-gradient(135deg, #51cf6620 0%, #3cb37120 100%);
        border: 2px solid #51cf66;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .info-box {
        background: linear-gradient(135deg, #e5e6de 5%, #c2d2ge 20%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model and encoders
@st.cache_resource
def load_artifacts():
    model = load_model("model.h5")
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)
    with open('onehot_encoded_geo.pkl', 'rb') as file:
        onehot_encoded_geo = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, label_encoder_gender, onehot_encoded_geo, scaler

model, label_encoder_gender, onehot_encoded_geo, scaler = load_artifacts()

# Header Section
st.markdown('<h1 class="main-header">🎯 Customer Churn Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powered by Artificial Neural Network | Predict customer retention with AI</p>', unsafe_allow_html=True)

# Info box
st.markdown("""
<div class="info-box">
    <strong>ℹ️ About This Tool:</strong> This AI-powered tool uses a trained neural network to predict the likelihood 
    of a customer leaving your service. Enter customer details below to get instant predictions.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 👤 Customer Demographics")
    geography = st.selectbox('🌍 Geography', onehot_encoded_geo.categories_[0], help="Select the customer's country")
    gender = st.selectbox('👫 Gender', label_encoder_gender.classes_, help="Select the customer's gender")
    age = st.slider('🎂 Age', 18, 92, 35, help="Customer's age in years")

with col2:
    st.markdown("### 💰 Financial Information")
    credit_score = st.number_input('📊 Credit Score', min_value=300, max_value=850, value=650, help="Customer's credit score (300-850)")
    balance = st.number_input('💵 Account Balance ($)', min_value=0.0, value=50000.0, format="%.2f", help="Current account balance")
    estimated_salary = st.number_input('💼 Estimated Salary ($)', min_value=0.0, value=75000.0, format="%.2f", help="Annual estimated salary")

st.markdown("---")

# Create another row
col3, col4 = st.columns(2)

with col3:
    st.markdown("### 📋 Account Details")
    tenure = st.slider('📅 Tenure (Years)', 0, 10, 5, help="Years as a customer")
    num_of_products = st.slider('📦 Number of Products', 1, 4, 2, help="Number of products held")

with col4:
    st.markdown("### ✅ Account Status")
    has_cr_card = st.selectbox('💳 Has Credit Card', ['Yes', 'No'], help="Does customer have a credit card?")
    is_active_member = st.selectbox('🔔 Is Active Member', ['Yes', 'No'], help="Is customer an active member?")

# Convert Yes/No to 1/0
has_cr_card_val = 1 if has_cr_card == 'Yes' else 0
is_active_member_val = 1 if is_active_member == 'Yes' else 0

st.markdown("---")

# Prediction button centered
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_button = st.button('🔮 Predict Churn Probability', use_container_width=True)

if predict_button:
    # Prepare the input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card_val],
        'IsActiveMember': [is_active_member_val],
        'EstimatedSalary': [estimated_salary]
    })

    geo_encoded = onehot_encoded_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoded_geo.get_feature_names_out(['Geography']))
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
    input_data_scaled = scaler.transform(input_data)

    # Prediction
    with st.spinner('🔄 Analyzing customer data...'):
        prediction = model.predict(input_data_scaled)
        prediction_prob = prediction[0][0]

    st.markdown("---")
    
    # Results section
    st.markdown("## 📈 Prediction Results")
    
    # Metrics row
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.metric(label="Churn Probability", value=f"{prediction_prob:.1%}", delta=None)
    
    with metric_col2:
        risk_level = "High Risk" if prediction_prob > 0.5 else "Low Risk"
        st.metric(label="Risk Level", value=risk_level)
    
    with metric_col3:
        confidence = max(prediction_prob, 1 - prediction_prob)
        st.metric(label="Model Confidence", value=f"{confidence:.1%}")

    # Progress bar for probability
    st.markdown("### Churn Risk Meter")
    st.progress(float(prediction_prob))

    # Result card
    if prediction_prob > 0.5:
        st.error(f"""
        ### ⚠️ High Churn Risk Detected!
        
        **Probability: {prediction_prob:.1%}**
        
        This customer shows a high likelihood of churning. Consider:
        - 📞 Proactive outreach and engagement
        - 🎁 Personalized retention offers
        - 💬 Satisfaction survey to understand concerns
        - 🌟 Loyalty program enrollment
        """)
    else:
        st.success(f"""
        ### ✅ Low Churn Risk
        
        **Probability: {prediction_prob:.1%}**
        
        This customer appears likely to stay. Recommendations:
        - 📈 Consider upselling opportunities
        - 🏆 Recognize loyalty with rewards
        - 📊 Monitor engagement metrics
        - 💡 Gather feedback for improvements
        """)

# Sidebar with additional info
with st.sidebar:
    st.markdown("## 📊 Model Information")
    st.markdown("---")
    
    st.markdown("""
    ### 🧠 Neural Network Details
    - **Model Type:** Artificial Neural Network
    - **Framework:** TensorFlow/Keras
    - **Input Features:** 12
    - **Output:** Binary Classification
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 📋 Input Features Used
    - Credit Score
    - Geography (One-Hot)
    - Gender
    - Age
    - Tenure
    - Balance
    - Number of Products
    - Has Credit Card
    - Is Active Member
    - Estimated Salary
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 📝 How to Use
    1. Enter customer details
    2. Click **Predict** button
    3. View churn probability
    4. Take action based on results
    """)
    
    st.markdown("---")
    st.markdown("*Built with ❤️ using Streamlit*")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 1rem;">
    <p>© 2024 Customer Churn Prediction System | Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)



