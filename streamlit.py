import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model
model = tf.keras.models.load_model('Hyperparameter_HybridBCNN_model.h5')  # Update with your model path

# Initialize the StandardScaler
scaler_mean = np.load('scaler_mean.npy')  # Update with your scaler mean path
scaler_scale = np.load('scaler_scale.npy')  # Update with your scaler scale path
scaler = StandardScaler()
scaler.mean_ = scaler_mean
scaler.scale_ = scaler_scale

# Define class labels
class_labels = {
    0: "No",
    1: "Yes",
    2: "No, borderline diabetes",
    3: "Yes (during pregnancy)"
}

# Define the Streamlit app
def main():
    st.title('Diabetes Prediction App')
    st.markdown('Enter the values for the following features to predict diabetes.')

    # Initialize input fields for features
    features = [
        "HeartDisease", "BMI", "Smoking", "AlcoholDrinking", "Stroke", 
        "PhysicalHealth", "MentalHealth", "DiffWalking", "Sex", "AgeCategory", 
        "Race", "PhysicalActivity", "GenHealth", "SleepTime", "Asthma", 
        "KidneyDisease", "SkinCancer"
    ]

    input_data = []
    for feature in features:
        value = st.number_input(f"Enter {feature}", step=0.01)
        input_data.append(value)

    # Predict button
    if st.button('Predict'):
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        input_scaled_reshaped = np.expand_dims(input_scaled, axis=2)  # Adjust shape for model input
        
        # Get predictions
        prediction = model.predict(input_scaled_reshaped)[0]
        predicted_class = np.argmax(prediction)  # Get the index of the highest probability
        
        st.success(f'Predicted Outcome: {class_labels[predicted_class]}')

if __name__ == '__main__':
    main()