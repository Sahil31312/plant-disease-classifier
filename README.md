# 🌱 Plant Disease Classifier

A professional web application for detecting plant diseases using Convolutional Neural Networks (CNN). Features bilingual support (English/Pashto), RTL/LTR switching, user authentication, admin dashboard, and comprehensive disease management.

## 📋 Features

### 🌐 Multi-Language Support
- **English & Pashto** interface with automatic RTL/LTR switching
- Real-time language switching without page reload
- Complete bilingual disease information

### 👥 User Management
- User registration and login system
- Role-based access control (Admin/User)
- User prediction history tracking
- Profile management

### 🎯 Disease Detection
- Real-time plant disease classification
- 8 PlantVillage dataset classes support
- Disease severity analysis
- Treatment recommendations
- Prevention tips

### 📊 Admin Dashboard
- Comprehensive system statistics
- User management interface
- Message inbox with reply functionality
- System logs with auto-deletion
- Disease information management

### 🔧 Technical Features
- **Flask** web framework with **SQLAlchemy** ORM
- **TensorFlow/Keras** model integration
- Responsive Bootstrap 5 design
- File upload with validation
- Real-time progress indicators
- Toast notifications
- Sidebar navigation with toggle
- Disclaimer and warning system

## 📁 Project Structure
# 🌱 Plant Disease Classifier

A professional web application for detecting plant diseases using Convolutional Neural Networks (CNN). Features bilingual support (English/Pashto), RTL/LTR switching, user authentication, admin dashboard, and comprehensive disease management.

## 📋 Features

### 🌐 Multi-Language Support
- **English & Pashto** interface with automatic RTL/LTR switching
- Real-time language switching without page reload
- Complete bilingual disease information

### 👥 User Management
- User registration and login system 
- Role-based access control (Admin/User)
- User prediction history tracking
- Profile management

### 🎯 Disease Detection
- Real-time plant disease classification
- 8 PlantVillage dataset classes support
- Disease severity analysis
- Treatment recommendations
- Prevention tips

### 📊 Admin Dashboard
- Comprehensive system statistics
- User management interface
- Message inbox with reply functionality
- System logs with auto-deletion
- Disease information management

### 🔧 Technical Features
- **Flask** web framework with **SQLAlchemy** ORM
- **TensorFlow/Keras** model integration
- Responsive Bootstrap 5 design
- File upload with validation
- Real-time progress indicators
- Toast notifications
- Sidebar navigation with toggle
- Disclaimer and warning system
## 📊 Supported Plant Diseases

The system detects 8 common plant diseases from the PlantVillage dataset:

**English Classes:**

- Pepper Bell Bacterial Spot
- Pepper Bell Healthy
- Potato Early Blight
- Potato Late Blight
- Potato Healthy
- Tomato Bacterial Spot
- Tomato Early Blight
- Tomato Late Blight

## 📁 Project Structure

---
```text
plant-disease-classifier/
│
├── app.py                 # Main Flask application
├── cnn.h5                 # Trained CNN model
├── requirements.txt       # Python dependencies
├── Procfile               # Heroku deployment
├── runtime.txt            # Python version
├── .gitignore             # Git ignore file
│
├── static/                # Static files
│   ├── css/
│   │   ├── style.css      # Main styles
│   │   └── rtl.css        # RTL styles
│   ├── js/
│   │   └── script.js      # JavaScript functions
│   └── uploads/           # User uploaded images
│
├── templates/             # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── profile.html
│   ├── prediction.html
│   ├── results.html
│   ├── analysis.html
│   ├── about.html
│   ├── contact.html
│   ├── admin_dashboard.html
│   ├── admin_users.html
│   ├── admin_messages.html
│   ├── admin_logs.html
│   ├── admin_diseases.html
│   ├── edit_disease.html
│   └── disease_info.html
│
└── plant_disease.db       # SQLite database
```

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/plant-disease-classifier.git
cd plant-disease-classifier
```


## 📦 `requirements.txt`

```txt
Flask==2.3.3
flask-sqlalchemy==3.0.5
flask-login==0.6.2
flask-bcrypt==1.0.1
tensorflow==2.13.0
keras==2.13.1
numpy==1.24.3
pillow==10.0.0
schedule==1.2.0
gunicorn==21.2.0
python-dotenv==1.0.0
```
## 🚢 Deployment

### Heroku Deployment

```bash
# Login to Heroku
heroku login

# Create Heroku app
heroku create your-app-name

# Set buildpack
heroku buildpacks:set heroku/python

# Deploy
git push heroku main

# Open application
heroku open
```

## Model Weights
Download the trained CNN model weights from
[huggingface.co](https://huggingface.co/ibrahimkhail/cnn-deployment/blob/main/cnn.h5).

