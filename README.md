# BetelCare ML Backend

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

## Overview

BetelCare ML Backend is a comprehensive Flask-based API server that powers the AI-driven agricultural intelligence platform for Sri Lankan betel farmers. The backend integrates multiple machine learning models to provide disease detection, harvest prediction, market forecasting, and weather-based recommendations.

## 🧠 ML Model Architecture

### 1. Disease Detection Engine
- **CNN-based pathogen identification** for brown spots and bacterial leaf blight  
- **Pest recognition system** for firefly and two-spotted red spider mites  
- **Image preprocessing pipeline** with segmentation and feature extraction  
- **Treatment recommendation engine** with rule-based advisory system  

### 2. Harvest Prediction Framework
- **Ensemble methodology** combining multiple ML algorithms  
- **Multi-type prediction** for P-Type, KT-Type, and RKT-Type betel varieties  
- **Environmental factor integration** (weather, soil, density analysis)  
- **Temporal pattern recognition** for optimal timing predictions  

### 3. Market Intelligence System
- **Price forecasting models** with regional trend analysis  
- **Profitability optimization** algorithms for market selection  
- **WhatsApp chatbot backend** (BetelBrio integration)  
- **Multilingual NLP processing** for Sinhala, Tamil, and English  

### 4. Weather Advisory Models
- **Three-tier recommendation system** for irrigation, fertilization, and protection  
- **Real-time weather integration** with OpenMeteo and Sri Lanka Meteorology APIs  
- **Location-specific algorithms** for Kurunegala, Puttalam, and Anamaduwa regions  
- **7-day forecasting pipeline** with agricultural decision support  

## 🏗️ Project Structure
```
betelcare-backend/
├── app/
│   ├── models/
│   │   ├── disease_detection/
│   │   ├── harvest_prediction/
│   │   ├── market_prediction/
│   │   └── weather_advisory/
│   ├── api/
│   │   ├── disease.py
│   │   ├── harvest.py
│   │   ├── market.py
│   │   └── weather.py
│   ├── services/
│   │   ├── image_processing.py
│   │   ├── weather_service.py
│   │   ├── chatbot_service.py
│   │   └── database_service.py
│   ├── utils/
│   └── config/
├── trained_models/
├── data/
├── tests/
└── deployment/
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+  
- pip package manager  
- Virtual environment (recommended)  
- PostgreSQL/SQLite for data storage  

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/betelcare-backend.git
   cd betelcare-backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Configure your API keys and database connections
   ```

5. **Initialize database**
   ```bash
   python manage.py init-db
   ```

6. **Run the application**
   ```bash
   python app.py
   ```

## 📦 Core Dependencies

```python
# Machine Learning & Data Science
tensorflow==2.13.0
scikit-learn==1.3.0
xgboost==1.7.6
opencv-python==4.8.0
numpy==1.24.3
pandas==2.0.3

# Web Framework & API
flask==2.3.3
flask-cors==4.0.0
flask-restful==0.3.10
gunicorn==21.2.0

# Database & Storage
sqlalchemy==2.0.20
psycopg2-binary==2.9.7
supabase==1.0.4

# External Integrations
requests==2.31.0
python-dotenv==1.0.0
pillow==10.0.0

# NLP & Chatbot
nltk==3.8.1
spacy==3.6.1
```

## 🔌 API Endpoints

### Disease Detection
```http
POST /api/disease/detect
Content-Type: multipart/form-data
# Upload image for disease/pest identification
# Returns: disease type, confidence, treatment recommendations
```

### Harvest Prediction
```http
POST /api/harvest/predict
Content-Type: application/json

{
  "field_area": 2.5,
  "leaf_type": "P-Type",
  "planting_date": "2024-01-15",
  "soil_type": "clay",
  "location": "Kurunegala"
}
```

### Market Intelligence
```http
POST /api/market/forecast
Content-Type: application/json

{
  "leaf_type": "KT-Type",
  "quantity": 100,
  "quality_grade": "premium",
  "harvest_date": "2024-03-20",
  "location": "Puttalam"
}
```

### Weather Advisory
```http
GET /api/weather/recommendations?location=Anamaduwa&days=7
# Returns: watering, fertilizing, and protection recommendations
```

### WhatsApp Chatbot Integration
```http
POST /api/chatbot/webhook
# Webhook endpoint for WhatsApp Business API integration
```

## 🤖 ML Model Details

### CNN Disease Detection
- Architecture: Custom CNN with transfer learning  
- Input: RGB images (224x224 pixels)  
- Output: Disease classification + confidence scores  
- Preprocessing: Segmentation, color normalization, augmentation  

### Ensemble Harvest Prediction
- Models: XGBoost, Random Forest, Gradient Boosting  
- Features: Weather patterns, soil data, historical yields  
- Output: Yield predictions with confidence intervals  

### Market Forecasting
- Algorithm: Time series analysis with feature engineering  
- Inputs: Historical prices, seasonal factors, quality metrics  
- Output: Price predictions and optimal market recommendations  

### Weather Advisory Models
- Watering Model: Random Forest (irrigation optimization)  
- Fertilizing Model: Random Forest (nutrient timing)  
- Protection Model: Random Forest (weather risk assessment)  

## 🗄️ Database Schema

```sql
-- Core tables for ML predictions and user data
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255),
    prediction_type VARCHAR(50),
    input_data JSONB,
    output_data JSONB,
    confidence_score FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE disease_detections (
    id SERIAL PRIMARY KEY,
    image_path VARCHAR(255),
    disease_type VARCHAR(100),
    confidence FLOAT,
    treatment_plan JSONB,
    location_data JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

## 🔧 Configuration

### Environment Variables
```env
# Flask Configuration
FLASK_ENV=development
SECRET_KEY=your_secret_key
DEBUG=True

# Database
DATABASE_URL=postgresql://user:password@localhost/betelcare
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# External APIs
OPENMETEO_API_KEY=your_openmeteo_key
WHATSAPP_API_TOKEN=your_whatsapp_token
GOOGLE_MAPS_API_KEY=your_google_maps_key

# ML Model Paths
DISEASE_MODEL_PATH=./trained_models/disease_cnn.h5
HARVEST_MODEL_PATH=./trained_models/harvest_ensemble.pkl
MARKET_MODEL_PATH=./trained_models/market_forecast.pkl
WEATHER_MODEL_PATH=./trained_models/weather_advisory.pkl
```

## 🧪 Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_disease_detection.py
python -m pytest tests/test_harvest_prediction.py
python -m pytest tests/test_api_endpoints.py

# Run with coverage
python -m pytest --cov=app tests/
```

## 🚀 Deployment

### Docker Deployment
```bash
# Build Docker image
docker build -t betelcare-backend .

# Run container
docker run -p 5000:5000 --env-file .env betelcare-backend
```

### Railway Deployment
```bash
# Install Railway CLI
npm install -g @railway/cli

# Deploy to Railway
railway login
railway deploy
```

## 📊 Performance Monitoring

- Model inference time tracking  
- API response time monitoring  
- Database query optimization  
- Memory usage analysis  
- Error logging and alerting  

## 🔒 Security Measures

- API rate limiting for endpoint protection  
- Input validation for all ML model inputs  
- Image file sanitization for disease detection uploads  
- Authentication tokens for mobile app integration  
- CORS configuration for secure cross-origin requests  

## 🤝 Contributing

1. Fork the repository  
2. Create a feature branch (`git checkout -b feature/ml-enhancement`)  
3. Implement changes with tests  
4. Commit changes (`git commit -am 'Add ML model improvement'`)  
5. Push to branch (`git push origin feature/ml-enhancement`)  
6. Create Pull Request  

## 📈 Model Training & Updates

```bash
# Disease detection model retraining
python scripts/train_disease_model.py --data-path ./data/diseases/

# Harvest prediction model update
python scripts/train_harvest_model.py --historical-data ./data/harvest/

# Market forecasting model retraining
python scripts/train_market_model.py --market-data ./data/prices/
```

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Research Team

- **Eshan Imesh** - Associate Software Engineer (ML Architecture)  
- **Saranga Siriwardhana** - Junior Full Stack Developer (API Development)  
- **Umesh Dewasinghe** - Trainee AI/ML Engineer (Model Training)  
- **Kavindi Fernando** - Trainee Business Analyst (Requirements & Testing)  

**Supervisors**:  
- Dr. Sanvitha Kasthuriarachchi - Assistant Professor, SLIIT  
- Ms. Lokesha Weerasinghe - Senior Lecturer, SLIIT  

## 📞 Support

For technical support, model questions, or API integration help, please open an issue in this repository or contact the development team.

**BetelCare ML Backend — Powering Intelligent Agriculture with AI 🤖🌱**
