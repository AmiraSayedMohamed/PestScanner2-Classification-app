import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import time
import pandas as pd
import requests
from datetime import datetime, timedelta

# App title and description
st.set_page_config(page_title="PestScanner Classification App", page_icon=":herb:")
st.title("🌱 PestScanner Disease Classification App")

# Sidebar info - Team Members
st.sidebar.header("Team Members")
st.sidebar.info(
    """
    **Abdelrahman**  
    Team Leader 
    
    **Amira**  
    Software Member
    
    **Omar**  
    Mechanical Member  
    """
)

# Green info message about what the app does - now in sidebar
st.sidebar.success("""
This app uses AI to analyze citrus leaf images and detect diseases.
Currently identifies:
- Black spot
- Citrus canker
""")

# Location selection in sidebar
st.sidebar.header("Location Information")
# Fixed country
country = "Egypt"

# List of major Egyptian cities
egyptian_cities = [
    "Cairo", "Alexandria", "Giza", "Shubra El-Kheima", "Port Said", 
    "Suez", "Luxor", "Mansoura", "El-Mahalla El-Kubra", "Tanta", 
    "Asyut", "Ismailia", "Faiyum", "Zagazig", "Aswan", 
    "Damietta", "Damanhur", "Minya", "Beni Suef", "Qena",
    "Sohag", "Hurghada", "6th of October City", "Shibin El Kom",
    "Banha", "Arish", "10th of Ramadan City", "Kafr El Sheikh",
    "Marsa Matruh", "Idfu", "Mit Ghamr", "Al-Hamidiyya",
    "Desouk", "Qalyub", "Abu Kabir", "Kafr El Dawwar",
    "Girga", "Akhmim", "Matareya"
]

# City selection dropdown
selected_city = st.sidebar.selectbox("Select your city in Egypt:", egyptian_cities)

# Display selected location in sidebar
st.sidebar.success(f"📍 Selected Location: {selected_city}, {country}")

# Combine into location string for weather API
location = f"{selected_city}, {country}"

# Weather API Configuration
OPENWEATHER_API_KEY = "77e0c3ce19e37b2f9c0a39cea77a7d19"

# Load your trained model (silently)
@st.cache_resource
def load_model():
    try:
        # Load TFLite model and allocate tensors
        interpreter = tf.lite.Interpreter(model_path='plant_disease_classifier_quant.tflite')
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

model = load_model()

# Define your class names
class_names = ['black-spot', 'citrus-canker']

# Recommendation database based on your uploaded images
recommendation_db = {
    'black-spot': {
        'chemical': [
            {'name': 'Ortus super', 'active': 'Fenpyroximate 5% EC', 'type': 'Contact', 'safety': 'Wear gloves and mask. Avoid application in windy conditions.'},
            {'name': 'TAK', 'active': 'Chlorpyrifos 48% EC', 'type': 'Systemic', 'safety': 'Highly toxic to bees. Apply in evening when bees are less active.'}
        ],
        'organic': [
            'Neem oil spray (apply every 7-10 days)',
            'Baking soda solution (1 tbsp baking soda + 1 tsp vegetable oil + 1 gallon water)',
            'Copper-based fungicides'
        ],
        'cultural': [
            'Remove and destroy infected leaves',
            'Improve air circulation by pruning',
            'Avoid overhead watering',
            'Rotate with non-citrus crops for 2 seasons'
        ],
        'description': 'Black spot is a fungal disease that causes dark spots on leaves and fruit. It thrives in warm, wet conditions.',
        'weather_risk': {
            'high_humidity': 70,
            'optimal_temp_range': (20, 30),
            'rain_risk': True
        }
    },
    'citrus-canker': {
        'chemical': [
            {'name': 'Biomectin', 'active': 'Abamectin 3% EC', 'type': 'Systemic', 'safety': 'Use protective clothing. Do not apply near water sources.'},
            {'name': 'AVENUE', 'active': 'Imidacloprid 70% SC', 'type': 'Systemic', 'safety': 'Toxic to aquatic organisms. Keep away from waterways.'}
        ],
        'organic': [
            'Copper-based bactericides',
            'Streptomycin sulfate (antibiotic spray)',
            'Garlic and chili pepper extract sprays'
        ],
        'cultural': [
            'Remove and burn infected plants',
            'Disinfect tools with 10% bleach solution',
            'Plant resistant varieties when available',
            'Implement strict quarantine measures for new plants'
        ],
        'description': 'Citrus canker is a bacterial disease causing raised lesions on leaves, stems, and fruit. Highly contagious.',
        'weather_risk': {
            'high_humidity': 75,
            'optimal_temp_range': (25, 35),
            'rain_risk': True
        }
    }
}

def get_weather_forecast(location):
    """Get 5-day weather forecast from OpenWeatherMap API"""
    try:
        # First get coordinates
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={location}&limit=1&appid={OPENWEATHER_API_KEY}"
        geo_response = requests.get(geo_url).json()
        
        if not geo_response:
            st.error("Location not found. Please try again.")
            return None
            
        lat = geo_response[0]['lat']
        lon = geo_response[0]['lon']
        
        # Then get forecast
        forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(forecast_url).json()
        
        if response.get('cod') != '200':
            st.error("Error fetching weather data. Please try again later.")
            return None
            
        # Process forecast data
        forecast_data = []
        for item in response['list']:
            date = datetime.fromtimestamp(item['dt'])
            forecast_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'time': date.strftime('%H:%M'),
                'temp': item['main']['temp'],
                'humidity': item['main']['humidity'],
                'weather': item['weather'][0]['main'],
                'rain': item.get('rain', {}).get('3h', 0)
            })
        
        return forecast_data
    except Exception as e:
        st.error(f"Error fetching weather data: {str(e)}")
        return None

def analyze_weather_risk(disease, forecast_data):
    """Analyze weather conditions for disease risk with more detailed output"""
    if not forecast_data or disease not in recommendation_db:
        return None
    
    weather_risk = recommendation_db[disease]['weather_risk']
    risk_factors = {
        'high_humidity': {'count': 0, 'max': 0, 'days': set()},
        'optimal_temp': {'count': 0, 'days': set(), 'min': float('inf'), 'max': -float('inf')},
        'rain': {'count': 0, 'days': set()}
    }
    
    # Analyze each forecast period
    for day in forecast_data:
        date = day['date']
        
        # Check humidity
        if day['humidity'] > weather_risk['high_humidity']:
            risk_factors['high_humidity']['count'] += 1
            risk_factors['high_humidity']['days'].add(date)
            if day['humidity'] > risk_factors['high_humidity']['max']:
                risk_factors['high_humidity']['max'] = day['humidity']
        
        # Check temperature
        temp_range = weather_risk['optimal_temp_range']
        if temp_range[0] <= day['temp'] <= temp_range[1]:
            risk_factors['optimal_temp']['count'] += 1
            risk_factors['optimal_temp']['days'].add(date)
            if day['temp'] < risk_factors['optimal_temp']['min']:
                risk_factors['optimal_temp']['min'] = day['temp']
            if day['temp'] > risk_factors['optimal_temp']['max']:
                risk_factors['optimal_temp']['max'] = day['temp']
        
        # Check for rain
        if weather_risk['rain_risk'] and day.get('rain', 0) > 0:
            risk_factors['rain']['count'] += 1
            risk_factors['rain']['days'].add(date)
    
    # Generate detailed risk messages
    messages = []
    temp_range = weather_risk['optimal_temp_range']
    
    if risk_factors['optimal_temp']['count'] > 0:
        day_count = len(risk_factors['optimal_temp']['days'])
        temp_min = risk_factors['optimal_temp']['min']
        temp_max = risk_factors['optimal_temp']['max']
        messages.append(
            f"🌡️ Ideal temperatures ({temp_range[0]}°C-{temp_range[1]}°C) expected on {day_count} days "
            f"(actual: {temp_min}°C to {temp_max}°C)"
        )
    
    if risk_factors['high_humidity']['count'] > 0:
        day_count = len(risk_factors['high_humidity']['days'])
        max_humidity = risk_factors['high_humidity']['max']
        messages.append(
            f"💧 High humidity (> {weather_risk['high_humidity']}%) expected on {day_count} days "
            f"(peaking at {max_humidity}%)"
        )
    
    if risk_factors['rain']['count'] > 0:
        day_count = len(risk_factors['rain']['days'])
        messages.append(f"☔ Rain expected on {day_count} days - will favor fungal growth")
    
    return messages if messages else None

def display_recommendations(disease, location=None):
    """Display recommendations based on the detected disease"""
    if disease not in recommendation_db:
        st.warning("No recommendations available for this disease.")
        return
    
    data = recommendation_db[disease]
    
    st.subheader(f"🌱 {disease.replace('-', ' ').title()} Information")
    st.info(data['description'])
    
    st.markdown("---")
    
    # Chemical recommendations
    st.subheader("🧪 Chemical Control Options")
    if data['chemical']:
        chem_df = pd.DataFrame(data['chemical'])
        st.table(chem_df)
        st.warning("⚠️ Always follow pesticide label instructions and local regulations")
    else:
        st.info("No chemical recommendations available")
    
    # Organic recommendations
    st.subheader("🍃 Organic/Natural Remedies")
    for remedy in data['organic']:
        st.markdown(f"- {remedy}")
    
    # Cultural practices
    st.subheader("🌿 Cultural Practices")
    for practice in data['cultural']:
        st.markdown(f"- {practice}")
    
    # Regulatory info
    st.markdown("---")
    st.subheader("📜 Regulatory Information")
    st.info("""
    - Always check with your local agricultural extension office
    - Some pesticides may be restricted in your area
    - Follow recommended pre-harvest intervals
    """)

# Image preprocessing function
def preprocess_image(image):
    img = np.array(image)
    if img.shape[-1] == 4:  # RGBA case
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif len(img.shape) == 2:  # Grayscale case
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  # Normalize to [0,1] range
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img.astype(np.float32)  # Convert to float32

# File uploader
uploaded_file = st.file_uploader(
    "Choose a citrus leaf image...", 
    type=["jpg", "jpeg", "png"]
)

# Add guidance when no file is uploaded
if model is not None and uploaded_file is None:
    st.info("ℹ️ Please upload a citrus leaf image to get a diagnosis")

if model is not None and uploaded_file is not None:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        # Preprocess and predict
        with st.spinner('Analyzing the leaf...'):
            processed_image = preprocess_image(image)
            
            # Get input and output tensors
            input_details = model.get_input_details()
            output_details = model.get_output_details()
            
            # Verify input shape and type
            if processed_image.shape != tuple(input_details[0]['shape']):
                processed_image = np.resize(processed_image, input_details[0]['shape'])
            
            # Set input tensor
            model.set_tensor(input_details[0]['index'], processed_image)
            
            # Run inference
            model.invoke()
            
            # Get predictions
            predictions = model.get_tensor(output_details[0]['index'])
            
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            time.sleep(1)
        
        # Display results
        st.success("Analysis Complete!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Diagnosis")
            disease = class_names[predicted_class]
            st.write(f"**Detected Disease:** {disease}")
            st.write(f"**Confidence:** {confidence:.2%}")
            st.warning("⚠️ Disease detected!")
            
            # Add expander for recommendations
            with st.expander("🛠️ View Treatment Recommendations", expanded=True):
                display_recommendations(disease, location)
        
        with col2:
            st.subheader("Probability Distribution")
            chart_data = pd.DataFrame({
                "Disease": class_names,
                "Probability": predictions[0]
            })
            st.bar_chart(chart_data.set_index('Disease'))
            
            # Forecasting Section under the graph
            st.subheader("🌤️ Disease Risk Forecasting")
            st.write("""
            This section analyzes local weather conditions to assess disease spread risk.
            The forecast checks for:
            - Ideal temperature ranges for disease development
            - High humidity levels that favor infection
            - Rain events that can spread pathogens
            """)
            
            if location:
                with st.spinner('Fetching weather forecast...'):
                    forecast_data = get_weather_forecast(location)
                    
                    if forecast_data:
                        risk_factors = analyze_weather_risk(disease, forecast_data)
                        
                        if risk_factors:
                            st.warning("⚠️ High Risk Conditions Detected")
                            for factor in risk_factors:
                                st.write(f"- {factor}")
                            
                            st.info("""
                            **Recommended Actions:**
                            - Increase monitoring of plants
                            - Apply preventive treatments
                            - Improve air circulation
                            - Remove infected plant material
                            """)
                        else:
                            st.success("✅ Current weather shows low disease risk")
                    else:
                        st.warning("Weather data unavailable - using general recommendations")
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <style>
    .footer {
        font-size: 12px;
        color: #777;
        text-align: center;
    }
    </style>
    <div class="footer">
        Citrus Disease Classifier | Made with Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
