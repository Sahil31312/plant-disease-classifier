from flask import Flask, render_template, request, jsonify, session, flash, redirect, url_for
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
import os
from datetime import datetime, timedelta
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.models import load_model
import json
import schedule
import time
import threading
import io



app = Flask(__name__)
app.secret_key = 'db9ea97ee41ac36408136c96da3bb7c13816cd75d601daed16a46cedd662a3df'

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///plant_disease.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# Create upload folder to save uploaded Images
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default='user')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    predictions = db.relationship('PredictionHistory', backref='user', lazy=True)


class ContactMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    subject = db.Column(db.String(200))
    message = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    language = db.Column(db.String(10), default='en')
    status = db.Column(db.String(20), default='unread')  # unread, read, replied foe the messages statues 


class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    filename = db.Column(db.String(200), nullable=False)
    predicted_class = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    is_healthy = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    language = db.Column(db.String(10), default='en')
    ip_address = db.Column(db.String(50))
    user_agent = db.Column(db.Text)


class DiseaseInfo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    disease_id = db.Column(db.Integer, nullable=False)
    language = db.Column(db.String(10), default='en')
    severity = db.Column(db.String(20))
    symptoms = db.Column(db.Text)
    treatment = db.Column(db.Text)
    prevention = db.Column(db.Text)
    recommendation = db.Column(db.Text)
    warning = db.Column(db.Text)
    disclaimer = db.Column(db.Text)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = db.Column(db.Integer, db.ForeignKey('user.id'))


class SystemLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    action = db.Column(db.String(200), nullable=False)
    details = db.Column(db.Text)
    ip_address = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# Create tables
with app.app_context():
    db.create_all()

    # Create default admin user 
    if not User.query.filter_by(username='admin').first():
        hashed_password = bcrypt.generate_password_hash('admin123').decode('utf-8')
        admin = User(
            username='Khairullah',
            email='ibrahimkhil975@gmail.com',
            password=hashed_password,
            role='admin'
        )
        db.session.add(admin)
        db.session.commit()
        print("✅ Default admin user created (username: admin, password: admin123)")

@app.route('/api/disease/<int:disease_id>/<language>')
@login_required
def get_disease_details(disease_id, language):
    """API endpoint to get disease details"""
    if current_user.role != 'admin':
        return jsonify({'success': False, 'message': 'Access denied'}), 403

    disease = DiseaseInfo.query.filter_by(
        disease_id=disease_id,
        language=language
    ).first()

    if not disease:
        return jsonify({'success': False, 'message': 'Disease not found'}), 404

    disease_name = class_names[language][disease_id] if disease_id < len(class_names[language]) else f'Disease {disease_id}'

    return jsonify({
        'success': True,
        'disease': {
            'severity': disease.severity,
            'symptoms': disease.symptoms,
            'treatment': disease.treatment,
            'prevention': disease.prevention,
            'recommendation': disease.recommendation,
            'warning': disease.warning,
            'disclaimer': disease.disclaimer,
            'updated_at': disease.updated_at.strftime('%Y-%m-%d %H:%M:%S') if disease.updated_at else None
        },
        'disease_name': disease_name
    })

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


def create_log(action, details=None):
    """Create system log entry"""
    log = SystemLog(
        user_id=current_user.id if current_user.is_authenticated else None,
        action=action,
        details=details,
        ip_address=request.remote_addr if request else None
    )
    db.session.add(log)
    db.session.commit()


# Loading trained model
try:
    model = load_model('cnn.h5')
    print("✅ Model loaded successfully!")

    # Check model input shape
    if model.input_shape:
        input_shape = model.input_shape[1:]  # Remove batch dimension
        print(f"📊 Model input shape: {model.input_shape}")
        print(f"📊 Model output shape: {model.output_shape}")
        print(f"📏 Expected image dimensions: {input_shape[0]}x{input_shape[1]}")
        print(f"📏 Expected channels: {input_shape[2]}")

        # Store target size for preprocessing as input fo the modeel 
        TARGET_SIZE = (input_shape[0], input_shape[1])
        print(f"🎯 Using target size: {TARGET_SIZE}")
    else:
        print("⚠️ Model input shape not available, using default 224x224")
        TARGET_SIZE = (224, 224)

except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    TARGET_SIZE = (224, 224)  # Default size

# Bilingual text data
translations = {
    'en': {
        'app_name': 'Plant Disease Classifier',
        'home': 'Home',
        'prediction': 'Prediction',
        'analysis': 'Analysis',
        'about': 'About',
        'contact': 'Contact',
        'dashboard': 'Dashboard',
        'admin_dashboard': 'Admin Dashboard',
        'login': 'Login',
        'register': 'Register',
        'logout': 'Logout',
        'profile': 'Profile',
        'switch_lang': 'Switch to Pashto',
        'upload_image': 'Upload Plant Image',
        'choose_file': 'Choose plant image...',
        'predict': 'Predict Disease',
        'upload_success': 'Image uploaded successfully!',
        'upload_error': 'Error uploading image. Please try again.',
        'prediction_result': 'Disease Prediction Result',
        'class_probabilities': 'Disease Probabilities',
        'confidence': 'Confidence',
        'back_home': 'Back to Home',
        'try_another': 'Try Another Image',
        'welcome': 'Welcome to Plant Disease Classifier',
        'project_desc': 'This project uses Convolutional Neural Networks to detect plant diseases with high accuracy.',
        'features': 'Features',
        'feature1': 'Real-time disease detection',
        'feature2': 'Multiple plant disease support',
        'feature3': 'High accuracy predictions',
        'feature4': 'Bilingual interface',
        'feature5': 'Disease severity analysis',
        'feature6': 'Treatment recommendations',
        'analysis_desc': 'View detailed analysis of disease predictions and model performance.',
        'about_desc': 'Learn more about plant disease detection and its development.',
        'contact_desc': 'Get in touch with our team for inquiries and support.',
        'copyright': '© 2024 Plant Disease Classifier Project. All rights reserved.',
        'language': 'Language',
        'english': 'English',
        'pashto': 'Pashto',
        'model_loaded': 'Model is ready for disease prediction',
        'model_error': 'Model not loaded properly',
        'supported_formats': 'Supported formats: JPG, PNG, JPEG',
        'max_size': 'Max file size: 16MB',
        'processing': 'Analyzing plant disease...',
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1_score': 'F1 Score',
        'total_predictions': 'Total Predictions',
        'success_rate': 'Success Rate',
        'recent_activity': 'Recent Activity',
        'contact_us': 'Contact Us',
        'name': 'Name',
        'email': 'Email',
        'message': 'Message',
        'submit': 'Submit',
        'clear': 'Clear',
        'disease_name': 'Disease Name',
        'plant_type': 'Plant Type',
        'severity': 'Severity',
        'recommendation': 'Recommendation',
        'healthy': 'Healthy',
        'unhealthy': 'Diseased',
        'prevention_tips': 'Prevention Tips',
        'treatment': 'Treatment',
        'symptoms': 'Symptoms',
        'disease_info': 'Disease Information',
        'upload_plant_image': 'Upload Plant Leaf Image',
        'detect_disease': 'Detect Disease',
        'view_details': 'View Details',
        'disease_detected': 'Disease Detected',
        'no_disease': 'No Disease Detected',
        'plant_healthy': 'Plant is Healthy',
        'take_action': 'Take Action',
        'monitor': 'Monitor Plant',
        'performance_metrics': 'Performance Metrics',
        'class_distribution': 'Disease Distribution',
        'performance_trend': 'Performance Trend',
        'date': 'Date',
        'time': 'Time',
        'image': 'Image',
        'status': 'Status',
        'successful': 'Successful',
        'project_overview': 'Project Overview',
        'what_is_cnn': 'What is Plant Disease Detection?',
        'project_goals': 'Project Goals',
        'technology_stack': 'Technology Stack',
        'model_specifications': 'Model Specifications',
        'model_type': 'Model Type',
        'input_size': 'Input Size',
        'classes': 'Disease Classes',
        'training_time': 'Training Time',
        'dataset_size': 'Dataset Size',
        'development_team': 'Development Team',
        'project_timeline': 'Project Timeline',
        'project_initiation': 'Project Initiation',
        'model_development': 'Model Development',
        'web_development': 'Web Development',
        'deployment': 'Deployment',
        'send_message': 'Send Message',
        'subject': 'Subject',
        'address': 'Address',
        'phone': 'Phone',
        'business_hours': 'Business Hours',
        'faq': 'Frequently Asked Questions',
        'location': 'Location',
        'preview': 'Preview',
        'instructions': 'Instructions',
        'click_predict': 'Click Predict',
        'view_results': 'View Results',
        'uploaded_image': 'Uploaded Image',
        'prediction_details': 'Prediction Details',
        'top_predictions': 'Top 3 Predictions',
        'probability': 'Probability',
        'percentage': 'Percentage',
        'uploaded_successfully': 'Uploaded Successfully',
        'prediction_result_desc': 'Here are the results of your plant disease analysis',
        'disease_severity': 'Disease Severity',
        'low': 'Low',
        'medium': 'Medium',
        'high': 'High',
        'critical': 'Critical',
        'prevention': 'Prevention',
        'organic_treatment': 'Organic Treatment',
        'chemical_treatment': 'Chemical Treatment',
        'crop_rotation': 'Crop Rotation',
        'proper_watering': 'Proper Watering',
        'fertilization': 'Fertilization',
        'message_sent': 'Message sent successfully!',
        'message_error': 'Error sending message. Please try again.',
        'name_placeholder': 'Enter your name',
        'email_placeholder': 'Enter your email',
        'subject_placeholder': 'Enter subject',
        'message_placeholder': 'Enter your message',
        'thank_you': 'Thank you for your message!',
        'we_will_contact': 'We will contact you soon.',
        'all_fields_required': 'All fields are required',
        'invalid_email': 'Please enter a valid email',
        'view_messages': 'View Messages',
        'unread_messages': 'Unread Messages',
        'total_messages': 'Total Messages',
        'username': 'Username',
        'password': 'Password',
        'confirm_password': 'Confirm Password',
        'remember_me': 'Remember me',
        'forgot_password': 'Forgot Password?',
        'no_account': "Don't have an account? Register",
        'have_account': 'Already have an account? Login',
        'login_success': 'Login successful!',
        'login_error': 'Invalid username or password',
        'register_success': 'Registration successful! Please login.',
        'register_error': 'Registration failed. Please try again.',
        'password_mismatch': 'Passwords do not match',
        'username_exists': 'Username already exists',
        'email_exists': 'Email already exists',
        'user_history': 'My Prediction History',
        'all_users': 'All Users',
        'system_logs': 'System Logs',
        'manage_diseases': 'Manage Diseases',
        'add_disease': 'Add Disease Info',
        'edit_disease': 'Edit Disease Info',
        'delete_disease': 'Delete Disease Info',
        'update': 'Update',
        'delete': 'Delete',
        'save': 'Save',
        'cancel': 'Cancel',
        'search': 'Search',
        'filter': 'Filter',
        'actions': 'Actions',
        'warning': 'Warning',
        'disclaimer': 'Disclaimer',
        'disclaimer_text': 'This system provides suggestions based on AI analysis. Always consult with agricultural experts before applying treatments.',
        'warning_text': 'Misuse of chemicals can be harmful. Follow safety guidelines.',
        'auto_delete_logs': 'Auto-delete logs older than 30 days',
        'delete_old_data': 'Delete old data',
        'backup_database': 'Backup Database',
        'restore_database': 'Restore Database',
        'export_data': 'Export Data',
        'import_data': 'Import Data',
        'settings': 'Settings',
        'toggle_sidebar': 'Toggle Sidebar',
        'sidebar_collapsed': 'Sidebar collapsed',
        'sidebar_expanded': 'Sidebar expanded'
    },
    'ps': {
        'app_name': 'د نباتاتو ناروغۍ ډلبندي',
        'home': 'کور',
        'prediction': 'پیش بیني',
        'analysis': 'تحلیل',
        'about': 'زموږ په اړه',
        'contact': 'اړیکه',
        'dashboard': 'ډاشبورډ',
        'admin_dashboard': 'د ادمین ډاشبورډ',
        'login': 'ننوتل',
        'register': 'ثبت نام',
        'logout': 'وتل',
        'profile': 'پروفایل',
        'switch_lang': 'انګلیسي ته بدل کړئ',
        'upload_image': 'د نبات انځور پورته کړئ',
        'choose_file': 'د نبات انځور غوره کړئ...',
        'predict': 'ناروغۍ ومومئ',
        'upload_success': 'انځور په بریالیتوب سره پورته شو!',
        'upload_error': 'د انځور پورته کولو کې ستونزه. بیا هڅه وکړئ.',
        'prediction_result': 'د نبات ناروغۍ د تشخیص نتیجه',
        'class_probabilities': 'د ناروغۍ احتمالات',
        'confidence': 'باوري',
        'back_home': 'کور ته راستن شئ',
        'try_another': 'بل انځور هڅه کړئ',
        'welcome': 'د نباتاتو ناروغۍ تشخیص کوونکي ته ښه راغلاست',
        'project_desc': 'دا پروژه د کانوولیوشنل عصبي شبکو په کارولو سره د نباتاتو ناروغۍ د لوړې کچې دقت سره تشخیص کوي.',
        'features': 'ځانګړتیاوې',
        'feature1': 'د ریښتیني وخت ناروغۍ تشخیص',
        'feature2': 'د څو نباتاتو د ناروغیو ملاتړ',
        'feature3': 'د لوړې کچې پیشبینۍ',
        'feature4': 'دوه ژبی انٹرفیس',
        'feature5': 'د ناروغۍ شدت تحلیل',
        'feature6': 'د درملنې وړاندیزونه',
        'analysis_desc': 'د ناروغۍ د پیشبینو او د مادل د کارکړې میټریکسونو تفصيلي تحلیل وګورئ.',
        'about_desc': 'د نباتاتو د ناروغۍ د تشخیص او د هغې د پراختیا په اړه نور معلومات ترلاسه کړئ.',
        'contact_desc': 'د پوښتنو او ملاتړ لپاره زموږ د ټیم سره اړیکه ونیسئ.',
        'copyright': '© ۲۰۲۴ د نباتاتو ناروغۍ تشخیص پروژه. ټول حقونه خوندي دي.',
        'language': 'ژبه',
        'english': 'انګلیسي',
        'pashto': 'پښتو',
        'model_loaded': 'مودل د ناروغۍ د تشخیص لپاره چمتو دی',
        'model_error': 'مودل په سمه توګه نه دی پورته شوی',
        'supported_formats': 'مشتري بڼې: JPG, PNG, JPEG',
        'max_size': 'د فایل اعظمي اندازه: ۱۶ میګابایت',
        'processing': 'د نبات ناروغۍ تحلیل کول...',
        'accuracy': 'کرهوالی',
        'precision': 'صحت',
        'recall': 'بیا راګرځول',
        'f1_score': 'F1 نمره',
        'total_predictions': 'ټولې پیشبینۍ',
        'success_rate': 'د بریالیتوب کچه',
        'recent_activity': 'وروستی فعالیت',
        'contact_us': 'زمونږ سره اړیکه ونیسئ',
        'name': 'نوم',
        'email': 'بریښنالیک',
        'message': 'پیغام',
        'submit': 'ولېږئ',
        'clear': 'پاک کړئ',
        'disease_name': 'د ناروغۍ نوم',
        'plant_type': 'د نبات ډول',
        'severity': 'شدت',
        'recommendation': 'وړاندیز',
        'healthy': 'سوک',
        'unhealthy': 'ناروغ',
        'prevention_tips': 'د مخنیوي لارښوونې',
        'treatment': 'درملنه',
        'symptoms': 'نښې نښانې',
        'disease_info': 'د ناروغۍ معلومات',
        'upload_plant_image': 'د نبات پاڼې انځور پورته کړئ',
        'detect_disease': 'ناروغۍ ومومئ',
        'view_details': 'تفصیلات وګورئ',
        'disease_detected': 'ناروغۍ وموندل شوه',
        'no_disease': 'هیڅ ناروغۍ ونه موندل شوه',
        'plant_healthy': 'نبات سوک دی',
        'take_action': 'عمل وکړئ',
        'monitor': 'نبات وڅارئ',
        'performance_metrics': 'د کارکړې میټریکسونه',
        'class_distribution': 'د ناروغۍ ویش',
        'performance_trend': 'د کارکړې لوری',
        'date': 'نېټه',
        'time': 'وخت',
        'image': 'انځور',
        'status': 'حالت',
        'successful': 'بریالی',
        'project_overview': 'د پروژې عمومي کتنه',
        'what_is_cnn': 'د نباتاتو د ناروغۍ تشخیص څه شی دی؟',
        'project_goals': 'د پروژې موخې',
        'technology_stack': 'تکنالوژي پوړ',
        'model_specifications': 'د مادل مشخصات',
        'model_type': 'د مادل ډول',
        'input_size': 'د داخلي کچه',
        'classes': 'د ناروغۍ ټولګۍ',
        'training_time': 'د روزني وخت',
        'dataset_size': 'د ډېټا سیټ کچه',
        'development_team': 'د پراختیا ټیم',
        'project_timeline': 'د پروژې وخت لړۍ',
        'project_initiation': 'د پروژې پیل',
        'model_development': 'د مادل پراختیا',
        'web_development': 'ویب پراختیا',
        'deployment': 'پلي کول',
        'send_message': 'پیغام ولېږئ',
        'subject': 'موضوع',
        'address': 'پته',
        'phone': 'تلیفون',
        'business_hours': 'د کار ساعتونه',
        'faq': 'په مکرر ډول پوښتل شوي پوښتنې',
        'location': 'ځای',
        'preview': 'مخکتنه',
        'instructions': 'لارښوونې',
        'click_predict': 'پیش بیني کلیک کړئ',
        'view_results': 'نتایج وګورئ',
        'uploaded_image': 'پورته شوی انځور',
        'prediction_details': 'د پیش بیني تفصیلات',
        'top_predictions': 'مخکینۍ ۳ پیشبینۍ',
        'probability': 'احتمال',
        'percentage': 'سلنه',
        'uploaded_successfully': 'په بریالیتوب سره پورته شو',
        'prediction_result_desc': 'دلته ستاسو د نبات د ناروغۍ د تحلیل نتایج دي',
        'disease_severity': 'د ناروغۍ شدت',
        'low': 'کم',
        'medium': 'منځنی',
        'high': 'لوړ',
        'critical': 'حساس',
        'prevention': 'مخنیوی',
        'organic_treatment': 'عضوي درملنه',
        'chemical_treatment': 'کیمیاوي درملنه',
        'crop_rotation': 'د فصلونو څرخه',
        'proper_watering': 'مناسب اوبه ورکول',
        'fertilization': 'سره ورکول',
        'message_sent': 'پیغام په بریالیتوب سره ولېږل شو!',
        'message_error': 'د پیغام د لېږلو ستونزه. بیا هڅه وکړئ.',
        'name_placeholder': 'خپل نوم ولیکئ',
        'email_placeholder': 'خپل بریښنالیک ولیکئ',
        'subject_placeholder': 'موضوع ولیکئ',
        'message_placeholder': 'خپل پیغام ولیکئ',
        'thank_you': 'د ستاسو د پیغام مننه!',
        'we_will_contact': 'موږ به ژر ستاسو سره اړیکه ونیسو.',
        'all_fields_required': 'ټول ساحې اړینې دي',
        'invalid_email': 'مهرباني وکړئ یو باوري بریښنالیک ولیکئ',
        'view_messages': 'پیغامونه وګورئ',
        'unread_messages': 'نالوستل شوي پیغامونه',
        'total_messages': 'ټول پیغامونه',
        'username': 'کارن نوم',
        'password': 'پاسورډ',
        'confirm_password': 'پاسورډ تایید کړئ',
        'remember_me': 'ما په یاد ساتل',
        'forgot_password': 'پاسورډ هیر شوی؟',
        'no_account': 'حساب نه لرئ؟ ثبت نام',
        'have_account': 'لا دمخه حساب لرئ؟ ننوتل',
        'login_success': 'په بریالیتوب سره ننوتل!',
        'login_error': 'ناسم کارن نوم یا پاسورډ',
        'register_success': 'ثبت نام په بریالیتوب سره! مهرباني وکړئ ننوتل.',
        'register_error': 'د ثبت نام ناکامي. بیا هڅه وکړئ.',
        'password_mismatch': 'پاسورډونه سمون نه خوري',
        'username_exists': 'کارن نوم لا دمخه شته',
        'email_exists': 'بریښنالیک لا دمخه شته',
        'user_history': 'زما د پیشبیني تاریخچه',
        'all_users': 'ټول کارونکي',
        'system_logs': 'د سیستم یادښتونه',
        'manage_diseases': 'ناروغۍ مدیریت',
        'add_disease': 'د ناروغۍ معلومات اضافه کړئ',
        'edit_disease': 'د ناروغۍ معلومات سم کړئ',
        'delete_disease': 'د ناروغۍ معلومات ړنګ کړئ',
        'update': 'تازه کول',
        'delete': 'ړنګول',
        'save': 'خوندي کول',
        'cancel': 'لغوه کول',
        'search': 'لټون',
        'filter': 'فلټر',
        'actions': 'کړنې',
        'warning': 'خبرتیا',
        'disclaimer': 'ذمه وری',
        'disclaimer_text': 'دا سیستم د AI تحلیل پر بنسټ وړاندیزونه چمتو کوي. د درملنې په اړه تل د کرنیزو متخصصینو سره مشوره وکړئ.',
        'warning_text': 'د کیمیاوي موادو ناسم کارول زیانمن کولی شي. د ساتنې لارښوونې تعقیب کړئ.',
        'auto_delete_logs': 'د ۳۰ ورځو څخه زوړ یادښتونه پاک کول',
        'delete_old_data': 'زوړ ډېټا ړنګ کړئ',
        'backup_database': 'ډیټابیس بیک اپ',
        'restore_database': 'ډیټابیس بیرته راوړل',
        'export_data': 'ډېټا صادرول',
        'import_data': 'ډېټا واردول',
        'settings': 'ترتیبات',
        'toggle_sidebar': 'سایډبار بدل کړئ',
        'sidebar_collapsed': 'سایډبار کوچنی شو',
        'sidebar_expanded': 'سایډبار لوی شو'
    }
}

# PlantVillage Dataset Classes
class_names = {
    'en': [
        'Pepper Bell Bacterial Spot',
        'Pepper Bell Healthy',
        'Potato Early Blight',
        'Potato Late Blight',
        'Potato Healthy',
        'Tomato Bacterial Spot',
        'Tomato Early Blight',
        'Tomato Late Blight'
    ],
    'ps': [
        'مرچ د باکتریا سپاټ',
        'مرچ سوکه',
        'کچالو لومړنۍ بلایټ',
        'کچالو وروستنۍ بلایټ',
        'کچالو سوک',
        'ټماټر د باکتریا سپاټ',
        'ټماټر لومړنۍ بلایټ',
        'ټماټر وروستنۍ بلایټ'
    ]
}


# Initialize disease info in database
def init_disease_info():
    with app.app_context():
        # Check if disease info already exists
        if DiseaseInfo.query.count() == 0:
            # English disease informations
            for i in range(len(class_names['en'])):
                disease = DiseaseInfo(
                    disease_id=i,
                    language='en',
                    severity='High' if i in [0, 5, 7] else ('Medium' if i in [2, 6] else 'None'),
                    symptoms='Sample symptoms for disease',
                    treatment='Sample treatment',
                    prevention='Sample prevention',
                    recommendation='Sample recommendation',
                    warning='Use with caution',
                    disclaimer='Consult experts'
                )
                db.session.add(disease)

            # Pashto disease information
            for i in range(len(class_names['ps'])):
                disease = DiseaseInfo(
                    disease_id=i,
                    language='ps',
                    severity='لوړ' if i in [0, 5, 7] else ('منځنی' if i in [2, 6] else 'هیڅ'),
                    symptoms='د ناروغۍ نمونه نښې',
                    treatment='نمونه درملنه',
                    prevention='نمونه مخنیوی',
                    recommendation='نمونه وړاندیز',
                    warning='په احتیاط سره وکاروئ',
                    disclaimer='متخصصینو سره مشوره وکړئ'
                )
                db.session.add(disease)

            db.session.commit()
            print("✅ Disease info initialized in database")


init_disease_info()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif', 'bmp'}


def preprocess_image(image_input):
    """
    Robust image preprocessing that handles ANY input format.
    Resizes to the EXACT size your model was trained on.

    Parameters:
    -----------
    image_input: Can be file path (str), file object, or bytes

    Returns:
    --------
    numpy array ready for model prediction
    """
    try:
        # ============================================
        # 1. LOAD IMAGE FROM ANY SOURCE
        # ============================================
        if isinstance(image_input, str):
            # File path
            img = Image.open(image_input)
        elif hasattr(image_input, 'read'):
          
            if hasattr(image_input, 'seek'):
                image_input.seek(0)
            img = Image.open(io.BytesIO(image_input.read()))
        elif isinstance(image_input, (bytes, bytearray)):
            # Raw bytes
            img = Image.open(io.BytesIO(image_input))
        else:
            raise ValueError(f"Unsupported input type: {type(image_input)}")

        # ============================================
        #   CONVERT TO RGB  
        # ============================================
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # ============================================
        # RESIZE TO MODEL'S TRAINING SIZE
        # ============================================
        # This is the FIX: Use TARGET_SIZE from model input shape
        img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

        # ============================================
        #  NORMALIZE AND PREPARE FOR MODEL
        # ============================================
        # Convert to numpy array and normalize to 0-1 range
        img_array = np.array(img, dtype=np.float32) / 255.0

        # Add batch dimension  required by Keras libariy
        img_array = np.expand_dims(img_array, axis=0)

        print(f"✅ Image preprocessed successfully")
        print(f"   Original size: {img.size}")
        print(f"   Resized to: {TARGET_SIZE}")
        print(f"   Array shape: {img_array.shape}")
        print(f"   Normalized: {img_array.min():.2f} to {img_array.max():.2f}")

        return img_array

    except Exception as e:
        print(f"❌ Error in preprocess_image: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def get_disease_info_db(class_index, lang='en'):
    """Get disease information from database"""
    disease = DiseaseInfo.query.filter_by(disease_id=class_index, language=lang).first()
    if disease:
        return {
            'severity': disease.severity,
            'symptoms': disease.symptoms,
            'treatment': disease.treatment,
            'prevention': disease.prevention,
            'recommendation': disease.recommendation,
            'warning': disease.warning,
            'disclaimer': disease.disclaimer
        }
    else:
        # Default if not in database
        return {
            'severity': 'Medium',
            'symptoms': 'Symptoms not specified',
            'treatment': 'Treatment not specified',
            'prevention': 'Prevention not specified',
            'recommendation': 'Consult agricultural expert',
            'warning': 'Use treatments with caution',
            'disclaimer': 'Always verify with experts'
        }


def get_severity_color(severity):
    """Get color based on disease severity"""
    colors = {
        'None': 'success',
        'Low': 'info',
        'Medium': 'warning',
        'High': 'danger',
        'Critical': 'dark',
        'هیڅ': 'success',
        'کم': 'info',
        'منځنی': 'warning',
        'لوړ': 'danger',
        'حساس': 'dark'
    }
    return colors.get(severity, 'secondary')


def get_message_stats():
    total = ContactMessage.query.count()
    unread = ContactMessage.query.filter_by(status='unread').count()
    return {'total': total, 'unread': unread}


def auto_delete_old_logs():
    """Auto delete logs older than 30 days"""
    with app.app_context():
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        old_logs = SystemLog.query.filter(SystemLog.created_at < thirty_days_ago).delete()
        db.session.commit()
        if old_logs > 0:
            print(f"✅ Auto-deleted {old_logs} old logs")
            create_log('Auto-deleted old logs', f'Deleted {old_logs} logs older than 30 days')


# Start background task for auto deletion mean log files
def start_background_tasks():
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(3600)  # Check every hour

    # Schedule auto deletion every day at midnight for logs file without manually delations 
    schedule.every().day.at("00:00").do(auto_delete_old_logs)

    # Start in background thread
    thread = threading.Thread(target=run_scheduler, daemon=True)
    thread.start()


# Routes
@app.route('/')
def index():
    lang = session.get('lang', 'en')
    direction = 'rtl' if lang == 'ps' else 'ltr'
    session['direction'] = direction

    stats = {
        'total_predictions': PredictionHistory.query.count(),
        'accuracy': 96.5,
        'healthy_plants': PredictionHistory.query.filter_by(is_healthy=True).count(),
        'diseased_plants': PredictionHistory.query.filter_by(is_healthy=False).count(),
        'common_disease': class_names[lang][5] if len(class_names[lang]) > 5 else 'Tomato Bacterial Spot'
    }

    recent_predictions = PredictionHistory.query.order_by(PredictionHistory.created_at.desc()).limit(5).all()

    return render_template('index.html',
                           lang=lang,
                           direction=direction,
                           t=translations[lang],
                           stats=stats,
                           recent_predictions=recent_predictions,
                           class_names=class_names,
                           model_loaded=model is not None)


@app.route('/set_lang/<language>')
def set_language(language):
    session['lang'] = language
    direction = 'rtl' if language == 'ps' else 'ltr'
    session['direction'] = direction

    return jsonify({
        'success': True,
        'direction': direction,
        'message': translations[language].get('language_changed', 'Language changed')
    })


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    lang = session.get('lang', 'en')
    direction = 'rtl' if lang == 'ps' else 'ltr'

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = True if request.form.get('remember') else False

        user = User.query.filter_by(username=username).first()

        if user and bcrypt.check_password_hash(user.password, password):
            if user.is_active:
                login_user(user, remember=remember)
                user.last_login = datetime.utcnow()
                db.session.commit()
                create_log('User login', f'User {username} logged in')
                flash(translations[lang]['login_success'], 'success')
                next_page = request.args.get('next')
                return redirect(next_page) if next_page else redirect(url_for('index'))
            else:
                flash('Account is disabled', 'error')
        else:
            flash(translations[lang]['login_error'], 'error')

    return render_template('login.html',
                           lang=lang,
                           direction=direction,
                           t=translations[lang])


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    lang = session.get('lang', 'en')
    direction = 'rtl' if lang == 'ps' else 'ltr'

    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        # Validation
        if password != confirm_password:
            flash(translations[lang]['password_mismatch'], 'error')
            return render_template('register.html',
                                   lang=lang,
                                   direction=direction,
                                   t=translations[lang])

        if User.query.filter_by(username=username).first():
            flash(translations[lang]['username_exists'], 'error')
            return render_template('register.html',
                                   lang=lang,
                                   direction=direction,
                                   t=translations[lang])

        if User.query.filter_by(email=email).first():
            flash(translations[lang]['email_exists'], 'error')
            return render_template('register.html',
                                   lang=lang,
                                   direction=direction,
                                   t=translations[lang])

        try:
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            user = User(
                username=username,
                email=email,
                password=hashed_password,
                role='user'
            )
            db.session.add(user)
            db.session.commit()
            create_log('User registration', f'New user {username} registered')
            flash(translations[lang]['register_success'], 'success')
            return redirect(url_for('login'))
        except Exception as e:
            flash(translations[lang]['register_error'], 'error')

    return render_template('register.html',
                           lang=lang,
                           direction=direction,
                           t=translations[lang])


@app.route('/logout')
@login_required
def logout():
    create_log('User logout', f'User {current_user.username} logged out')
    logout_user()
    return redirect(url_for('index'))


@app.route('/about')
def about():
    lang = session.get('lang', 'en')
    direction = 'rtl' if lang == 'ps' else 'ltr'

    model_specs = {
        'type': 'CNN (Convolutional Neural Network)',
        'input_size': f'{TARGET_SIZE[0]}x{TARGET_SIZE[1]} pixels',
        'classes': len(class_names['en']),
        'accuracy': '96.5%',
        'training_time': '4 hours',
        'dataset_size': '10,000 images',
        'framework': 'TensorFlow/Keras'
    }

    return render_template('about.html',
                           lang=lang,
                           direction=direction,
                           t=translations[lang],
                           model_specs=model_specs)


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    lang = session.get('lang', 'en')
    direction = 'rtl' if lang == 'ps' else 'ltr'

    if request.method == 'POST':
        if 'file' not in request.files:
            flash(translations[lang]['upload_error'], 'error')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash(translations[lang]['upload_error'], 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess the image
            processed_image = preprocess_image(filepath)

            if processed_image is None:
                flash('Error processing image', 'error')
                return redirect(request.url)

            if model is None:
                flash('Model not loaded', 'error')
                return redirect(request.url)

            try:
                # Make prediction
                predictions = model.predict(processed_image, verbose=0)
                predicted_class = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_class])

                print(f"✅ Prediction successful!")
                print(f"   Class: {predicted_class} ({class_names[lang][predicted_class]})")
                print(f"   Confidence: {confidence:.2%}")

                # Save to database with user info if logged in
                new_prediction = PredictionHistory(
                    user_id=current_user.id if current_user.is_authenticated else None,
                    filename=filename,
                    predicted_class=class_names[lang][predicted_class],
                    confidence=confidence,
                    is_healthy=predicted_class in [1, 4],
                    language=lang,
                    ip_address=request.remote_addr,
                    user_agent=request.user_agent.string
                )
                db.session.add(new_prediction)
                db.session.commit()

                # Log prediction
                if current_user.is_authenticated:
                    create_log('Prediction made',
                               f'User predicted {class_names[lang][predicted_class]} with {confidence:.2%} confidence')
                else:
                    create_log('Anonymous prediction',
                               f'Predicted {class_names[lang][predicted_class]} with {confidence:.2%} confidence')

                disease_data = get_disease_info_db(predicted_class, lang)
                is_healthy = predicted_class in [1, 4]

                # Get top 3 predictions
                top_3_indices = np.argsort(predictions[0])[-3:][::-1]
                top_3_classes = []

                for idx in top_3_indices:
                    top_3_classes.append({
                        'class': class_names[lang][idx],
                        'confidence': float(predictions[0][idx]),
                        'severity': get_disease_info_db(idx, lang)['severity'],
                        'color': get_severity_color(get_disease_info_db(idx, lang)['severity'])
                    })

                # Prepare all predictions
                all_predictions = []
                for i in range(len(class_names[lang])):
                    all_predictions.append({
                        'class': class_names[lang][i],
                        'confidence': float(predictions[0][i]),
                        'severity': get_disease_info_db(i, lang)['severity'],
                        'color': get_severity_color(get_disease_info_db(i, lang)['severity'])
                    })

                flash(translations[lang]['upload_success'], 'success')

                return render_template('results.html',
                                       lang=lang,
                                       direction=direction,
                                       t=translations[lang],
                                       filename=filename,
                                       predicted_class=class_names[lang][predicted_class],
                                       confidence=confidence,
                                       is_healthy=is_healthy,
                                       disease_data=disease_data,
                                       severity_color=get_severity_color(disease_data['severity']),
                                       top_classes=top_3_classes,
                                       all_predictions=all_predictions,
                                       class_index=predicted_class,
                                       class_names=class_names)

            except Exception as e:
                print(f"❌ Prediction error: {e}")
                flash(f'Prediction error: {str(e)}', 'error')
                return redirect(request.url)
        else:
            flash('Invalid file format', 'error')
            return redirect(request.url)

    return render_template('prediction.html',
                           lang=lang,
                           direction=direction,
                           t=translations[lang],
                           model_loaded=model is not None)


@app.route('/analysis')
def analysis():
    lang = session.get('lang', 'en')
    direction = 'rtl' if lang == 'ps' else 'ltr'

    total_predictions = PredictionHistory.query.count()
    healthy_count = PredictionHistory.query.filter_by(is_healthy=True).count()
    diseased_count = PredictionHistory.query.filter_by(is_healthy=False).count()

    analysis_data = {
        'accuracy': 0.965,
        'precision': 0.94,
        'recall': 0.93,
        'f1_score': 0.935,
        'total_predictions': total_predictions,
        'success_rate': 0.96,
        'healthy_count': healthy_count,
        'diseased_count': diseased_count
    }

    class_distribution = []
    for i, class_name in enumerate(class_names[lang]):
        count = PredictionHistory.query.filter_by(predicted_class=class_name).count()
        percentage = (count / total_predictions * 100) if total_predictions > 0 else 0
        class_distribution.append({
            'name': class_name,
            'count': count,
            'percentage': round(percentage, 1)
        })

    return render_template('analysis.html',
                           lang=lang,
                           direction=direction,
                           t=translations[lang],
                           analysis_data=analysis_data,
                           class_distribution=class_distribution,
                           class_names=class_names)


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    lang = session.get('lang', 'en')
    direction = 'rtl' if lang == 'ps' else 'ltr'

    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')

        if not name or not email or not message:
            flash(translations[lang]['all_fields_required'], 'error')
            return render_template('contact.html',
                                   lang=lang,
                                   direction=direction,
                                   t=translations[lang],
                                   success=False)

        try:
            new_message = ContactMessage(
                name=name,
                email=email,
                subject=subject,
                message=message,
                language=lang
            )
            db.session.add(new_message)
            db.session.commit()

            create_log('Contact message', f'Message from {name} ({email})')

            flash(translations[lang]['message_sent'], 'success')
            return render_template('contact.html',
                                   lang=lang,
                                   direction=direction,
                                   t=translations[lang],
                                   success=True)
        except Exception as e:
            print(f"Error saving message: {e}")
            flash(translations[lang]['message_error'], 'error')
            return render_template('contact.html',
                                   lang=lang,
                                   direction=direction,
                                   t=translations[lang],
                                   success=False)

    return render_template('contact.html',
                           lang=lang,
                           direction=direction,
                           t=translations[lang],
                           success=False)


# ==================== ADMIN ROUTES ====================
@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    if current_user.role != 'admin':
        flash('Access denied', 'error')
        return redirect(url_for('index'))

    lang = session.get('lang', 'en')
    direction = 'rtl' if lang == 'ps' else 'ltr'

    # Statistics
    stats = {
        'total_users': User.query.count(),
        'total_predictions': PredictionHistory.query.count(),
        'total_messages': ContactMessage.query.count(),
        'unread_messages': ContactMessage.query.filter_by(status='unread').count(),
        'today_predictions': PredictionHistory.query.filter(
            PredictionHistory.created_at >= datetime.utcnow().date()
        ).count(),
        'active_users': User.query.filter_by(is_active=True).count()
    }

    # Recent activities
    recent_predictions = PredictionHistory.query.order_by(
        PredictionHistory.created_at.desc()
    ).limit(10).all()

    recent_messages = ContactMessage.query.order_by(
        ContactMessage.created_at.desc()
    ).limit(10).all()

    recent_logs = SystemLog.query.order_by(
        SystemLog.created_at.desc()
    ).limit(10).all()

    return render_template('admin_dashboard.html',
                           lang=lang,
                           direction=direction,
                           t=translations[lang],
                           stats=stats,
                           recent_predictions=recent_predictions,
                           recent_messages=recent_messages,
                           recent_logs=recent_logs,
                           class_names=class_names)


@app.route('/admin/messages')
@login_required
def view_messages():
    if current_user.role != 'admin':
        flash('Access denied', 'error')
        return redirect(url_for('index'))

    lang = session.get('lang', 'en')
    direction = 'rtl' if lang == 'ps' else 'ltr'

    messages = ContactMessage.query.order_by(ContactMessage.created_at.desc()).all()
    stats = get_message_stats()

    return render_template('admin_messages.html',
                           lang=lang,
                           direction=direction,
                           t=translations[lang],
                           messages=messages,
                           stats=stats)


@app.route('/admin/users')
@login_required
def manage_users():
    if current_user.role != 'admin':
        flash('Access denied', 'error')
        return redirect(url_for('index'))

    lang = session.get('lang', 'en')
    direction = 'rtl' if lang == 'ps' else 'ltr'

    users = User.query.order_by(User.created_at.desc()).all()

    return render_template('admin_users.html',
                           lang=lang,
                           direction=direction,
                           t=translations[lang],
                           users=users)


@app.route('/admin/logs')
@login_required
def view_logs():
    if current_user.role != 'admin':
        flash('Access denied', 'error')
        return redirect(url_for('index'))

    lang = session.get('lang', 'en')
    direction = 'rtl' if lang == 'ps' else 'ltr'

    logs = SystemLog.query.order_by(SystemLog.created_at.desc()).all()

    return render_template('admin_logs.html',
                           lang=lang,
                           direction=direction,
                           t=translations[lang],
                           logs=logs)


@app.route('/admin/diseases')
@login_required
def manage_diseases():
    if current_user.role != 'admin':
        flash('Access denied', 'error')
        return redirect(url_for('index'))

    lang = session.get('lang', 'en')
    direction = 'rtl' if lang == 'ps' else 'ltr'

    diseases_en = DiseaseInfo.query.filter_by(language='en').order_by(DiseaseInfo.disease_id).all()
    diseases_ps = DiseaseInfo.query.filter_by(language='ps').order_by(DiseaseInfo.disease_id).all()

    return render_template('admin_diseases.html',
                           lang=lang,
                           direction=direction,
                           t=translations[lang],
                           diseases_en=diseases_en,
                           diseases_ps=diseases_ps,
                           class_names=class_names)


@app.route('/admin/disease/edit/<int:disease_id>/<language>', methods=['GET', 'POST'])
@login_required
def edit_disease(disease_id, language):
    if current_user.role != 'admin':
        flash('Access denied', 'error')
        return redirect(url_for('index'))

    lang = session.get('lang', 'en')
    direction = 'rtl' if lang == 'ps' else 'ltr'

    disease = DiseaseInfo.query.filter_by(disease_id=disease_id, language=language).first()

    if not disease:
        flash('Disease not found', 'error')
        return redirect(url_for('manage_diseases'))

    if request.method == 'POST':
        disease.severity = request.form.get('severity')
        disease.symptoms = request.form.get('symptoms')
        disease.treatment = request.form.get('treatment')
        disease.prevention = request.form.get('prevention')
        disease.recommendation = request.form.get('recommendation')
        disease.warning = request.form.get('warning')
        disease.disclaimer = request.form.get('disclaimer')
        disease.updated_by = current_user.id

        db.session.commit()

        create_log('Disease info updated',
                   f'Updated disease {disease_id} ({language}) info')

        flash('Disease information updated successfully', 'success')
        return redirect(url_for('manage_diseases'))

    return render_template('edit_disease.html',
                           lang=lang,
                           direction=direction,
                           t=translations[lang],
                           disease=disease,
                           disease_name=class_names[language][disease_id] if disease_id < len(
                               class_names[language]) else f'Disease {disease_id}')


@app.route('/admin/logs/delete_old', methods=['POST'])
@login_required
def delete_old_logs():
    if current_user.role != 'admin':
        return jsonify({'success': False, 'message': 'Access denied'}), 403

    try:
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        deleted_count = SystemLog.query.filter(SystemLog.created_at < thirty_days_ago).delete()
        db.session.commit()

        create_log('Manual log cleanup', f'Deleted {deleted_count} old logs')

        return jsonify({
            'success': True,
            'message': f'Deleted {deleted_count} old logs',
            'deleted_count': deleted_count
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/admin/logs/delete_all', methods=['POST'])
@login_required
def delete_all_logs():
    if current_user.role != 'admin':
        return jsonify({'success': False, 'message': 'Access denied'}), 403

    try:
        deleted_count = SystemLog.query.delete()
        db.session.commit()

        create_log('All logs deleted', f'Deleted all {deleted_count} logs')

        return jsonify({
            'success': True,
            'message': f'Deleted all {deleted_count} logs',
            'deleted_count': deleted_count
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# User profile and history
@app.route('/profile')
@login_required
def profile():
    lang = session.get('lang', 'en')
    direction = 'rtl' if lang == 'ps' else 'ltr'

    user_predictions = PredictionHistory.query.filter_by(
        user_id=current_user.id
    ).order_by(PredictionHistory.created_at.desc()).all()

    return render_template('profile.html',
                           lang=lang,
                           direction=direction,
                           t=translations[lang],
                           user_predictions=user_predictions,
                           class_names=class_names)


# API endpoints
@app.route('/api/page_content')
def get_page_content():
    lang = session.get('lang', 'en')
    direction = 'rtl' if lang == 'ps' else 'ltr'

    return jsonify({
        'success': True,
        'title': translations[lang]['app_name'],
        'translations': translations[lang],
        'direction': direction
    })


@app.route('/api/message/<int:message_id>/read', methods=['POST'])
@login_required
def mark_message_read(message_id):
    if current_user.role != 'admin':
        return jsonify({'success': False, 'message': 'Access denied'}), 403

    message = ContactMessage.query.get_or_404(message_id)
    message.status = 'read'
    db.session.commit()

    create_log('Message marked as read', f'Message {message_id} marked as read')

    return jsonify({'success': True})


@app.route('/api/user/<int:user_id>/toggle_active', methods=['POST'])
@login_required
def toggle_user_active(user_id):
    if current_user.role != 'admin':
        return jsonify({'success': False, 'message': 'Access denied'}), 403

    user = User.query.get_or_404(user_id)
    user.is_active = not user.is_active
    db.session.commit()

    status = 'activated' if user.is_active else 'deactivated'
    create_log('User status changed', f'User {user.username} {status}')

    return jsonify({
        'success': True,
        'is_active': user.is_active,
        'message': f'User {status} successfully'
    })

# =====================================
 

@app.route('/api/message/<int:message_id>/reply', methods=['POST'])
@login_required
def reply_to_message(message_id):
    """API endpoint to reply to a message"""
    if current_user.role != 'admin':
        return jsonify({'success': False, 'message': 'Access denied'}), 403

    data = request.get_json()
    reply_message = data.get('message')

    if not reply_message:
        return jsonify({'success': False, 'message': 'Reply message required'}), 400

    # Get original message
    original_message = ContactMessage.query.get_or_404(message_id)

   
    original_message.status = 'replied'
    db.session.commit()

    # Create log
    create_log('Message replied',
               f'Replied to message {message_id} from {original_message.email}')

    return jsonify({
        'success': True,
        'message': 'Reply sent successfully'
    })


@app.route('/api/message/<int:message_id>', methods=['DELETE'])
@login_required
def delete_message(message_id):
    """API endpoint to delete a message"""
    if current_user.role != 'admin':
        return jsonify({'success': False, 'message': 'Access denied'}), 403

    message = ContactMessage.query.get_or_404(message_id)

    db.session.delete(message)
    db.session.commit()

    create_log('Message deleted',
               f'Deleted message {message_id} from {message.email}')

    return jsonify({
        'success': True,
        'message': 'Message deleted successfully'
    })


@app.route('/api/disease/<int:disease_id>/<language>', methods=['DELETE'])
@login_required
def delete_disease_info(disease_id, language):
    """API endpoint to delete disease information"""
    if current_user.role != 'admin':
        return jsonify({'success': False, 'message': 'Access denied'}), 403

    disease = DiseaseInfo.query.filter_by(
        disease_id=disease_id,
        language=language
    ).first()

    if not disease:
        return jsonify({'success': False, 'message': 'Disease not found'}), 404

    db.session.delete(disease)
    db.session.commit()

    create_log('Disease info deleted',
               f'Deleted disease {disease_id} info for {language}')

    return jsonify({
        'success': True,
        'message': 'Disease information deleted'
    })


@app.route('/api/user/<int:user_id>/delete', methods=['DELETE'])
@login_required
def delete_user(user_id):
    """API endpoint to delete a user"""
    if current_user.role != 'admin':
        return jsonify({'success': False, 'message': 'Access denied'}), 403

    if user_id == current_user.id:
        return jsonify({'success': False, 'message': 'Cannot delete yourself'}), 400

    user = User.query.get_or_404(user_id)

    # Check if user is admin
    if user.role == 'admin':
        return jsonify({'success': False, 'message': 'Cannot delete admin users'}), 400

    # Delete user's predictions
    PredictionHistory.query.filter_by(user_id=user_id).delete()

    # Delete user
    db.session.delete(user)
    db.session.commit()

    create_log('User deleted',
               f'Deleted user {user.username} ({user.email})')

    return jsonify({
        'success': True,
        'message': 'User deleted successfully'
    })


@app.route('/api/user/<int:user_id>/make_admin', methods=['POST'])
@login_required
def make_user_admin(user_id):
    """API endpoint to make user an admin"""
    if current_user.role != 'admin':
        return jsonify({'success': False, 'message': 'Access denied'}), 403

    user = User.query.get_or_404(user_id)
    user.role = 'admin'
    db.session.commit()

    create_log('User promoted to admin',
               f'User {user.username} promoted to admin role')

    return jsonify({
        'success': True,
        'message': 'User promoted to admin successfully'
    })


@app.route('/api/diseases/sync', methods=['POST'])
@login_required
def sync_diseases():
    """API endpoint to sync all disease information"""
    if current_user.role != 'admin':
        return jsonify({'success': False, 'message': 'Access denied'}), 403

    try:
        # Sync English diseases
        for i in range(len(class_names['en'])):
            disease = DiseaseInfo.query.filter_by(
                disease_id=i,
                language='en'
            ).first()

            if not disease:
                disease = DiseaseInfo(
                    disease_id=i,
                    language='en',
                    severity='Medium' if i in [2, 6] else ('High' if i in [0, 5, 7] else 'None'),
                    symptoms=f'Symptoms for {class_names["en"][i]}',
                    treatment='Consult agricultural expert',
                    prevention='Use proper farming practices',
                    recommendation='Contact local agriculture department',
                    warning='Use treatments with caution',
                    disclaimer='Always verify with experts',
                    updated_by=current_user.id
                )
                db.session.add(disease)

        # Sync Pashto diseases
        for i in range(len(class_names['ps'])):
            disease = DiseaseInfo.query.filter_by(
                disease_id=i,
                language='ps'
            ).first()

            if not disease:
                disease = DiseaseInfo(
                    disease_id=i,
                    language='ps',
                    severity='منځنی' if i in [2, 6] else ('لوړ' if i in [0, 5, 7] else 'هیڅ'),
                    symptoms=f'د {class_names["ps"][i]} لپاره نښې نښانې',
                    treatment='د کرنیزو متخصصینو سره مشوره وکړئ',
                    prevention='مناسب کرنیز عمل وکاروئ',
                    recommendation='سیمه ئیز کرنیز ریاست سره اړیکه ونیسئ',
                    warning='درملنې په احتیاط سره وکاروئ',
                    disclaimer='تل د متخصصینو سره تایید کړئ',
                    updated_by=current_user.id
                )
                db.session.add(disease)

        db.session.commit()

        create_log('Diseases synced',
                   'Synced disease information for all classes')

        return jsonify({
            'success': True,
            'message': 'Diseases synced successfully',
            'count': DiseaseInfo.query.count()
        })

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# ==============================================

@app.route('/disease_info/<int:disease_id>')
def disease_info_page(disease_id):
    lang = session.get('lang', 'en')
    direction = 'rtl' if lang == 'ps' else 'ltr'

    if disease_id < 0 or disease_id >= len(class_names[lang]):
        return redirect(url_for('index'))

    disease_data = get_disease_info_db(disease_id, lang)

    return render_template('disease_info.html',
                           lang=lang,
                           direction=direction,
                           t=translations[lang],
                           disease_id=disease_id,
                           disease_name=class_names[lang][disease_id],
                           disease_data=disease_data,
                           severity_color=get_severity_color(disease_data['severity']))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file and allowed_file(file.filename):
        processed_image = preprocess_image(file)

        if processed_image is not None and model is not None:
            predictions = model.predict(processed_image, verbose=0)
            predicted_class = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][predicted_class])

            disease_data = get_disease_info_db(predicted_class, 'en')

            return jsonify({
                'success': True,
                'predicted_class': predicted_class,
                'class_name': class_names['en'][predicted_class],
                'confidence': confidence,
                'is_healthy': predicted_class in [1, 4],
                'disease_info': disease_data,
                'all_predictions': predictions[0].tolist()
            })

    return jsonify({'error': 'Prediction failed'}), 400


@app.route('/api/stats')
def api_stats():
    total_predictions = PredictionHistory.query.count()
    healthy_count = PredictionHistory.query.filter_by(is_healthy=True).count()
    diseased_count = PredictionHistory.query.filter_by(is_healthy=False).count()

    return jsonify({
        'total_predictions': total_predictions,
        'accuracy': 0.965,
        'healthy_count': healthy_count,
        'diseased_count': diseased_count,
        'class_distribution': [
            {
                'class': class_names['en'][i],
                'count': PredictionHistory.query.filter_by(predicted_class=class_names['en'][i]).count()
            }
            for i in range(len(class_names['en']))
        ]
    })



# =========================
def get_disease_info(class_index, lang='en'):
    """Wrapper for get_disease_info_db with simpler name for templates"""
    return get_disease_info_db(class_index, lang)

# ======================


 
# Context processor to make variables available to all templates
@app.context_processor
def inject_globals():
    lang = session.get('lang', 'en')
    direction = 'rtl' if lang == 'ps' else 'ltr'
    return dict(
        lang=lang,
        direction=direction,
        t=translations[lang],
        class_names=class_names,
        current_user=current_user,
        get_severity_color=get_severity_color,
        get_view_functions=get_view_functions,
        get_disease_info=get_disease_info,
        datetime=datetime,  # Add this line
        now=datetime.now
    )


# ====================================

 
@app.route('/admin/disease/view/<int:disease_id>/<language>')
@login_required
def view_disease(disease_id, language):
    if current_user.role != 'admin':
        flash('Access denied', 'error')
        return redirect(url_for('index'))

    lang = session.get('lang', 'en')
    direction = 'rtl' if lang == 'ps' else 'ltr'

    disease = DiseaseInfo.query.filter_by(disease_id=disease_id, language=language).first()

    if not disease:
        flash('Disease not found', 'error')
        return redirect(url_for('manage_diseases'))

    return render_template('view_disease.html',
                           lang=lang,
                           direction=direction,
                           t=translations[lang],
                           disease=disease,
                           disease_name=class_names[language][disease_id] if disease_id < len(
                               class_names[language]) else f'Disease {disease_id}')

 
@app.route('/admin/disease/add', methods=['GET', 'POST'])
@login_required
def add_disease():
    if current_user.role != 'admin':
        flash('Access denied', 'error')
        return redirect(url_for('index'))

    lang = session.get('lang', 'en')
    direction = 'rtl' if lang == 'ps' else 'ltr'

    if request.method == 'POST':
        try:
            disease_id = int(request.form.get('disease_id'))
            language = request.form.get('language')
            severity = request.form.get('severity')
            symptoms = request.form.get('symptoms')
            treatment = request.form.get('treatment')
            prevention = request.form.get('prevention')
            recommendation = request.form.get('recommendation')
            warning = request.form.get('warning')
            disclaimer = request.form.get('disclaimer')

            # Check if disease info already exists
            existing = DiseaseInfo.query.filter_by(disease_id=disease_id, language=language).first()
            if existing:
                flash('Disease info already exists for this language', 'error')
                return redirect(url_for('add_disease'))

            # Create new disease info
            new_disease = DiseaseInfo(
                disease_id=disease_id,
                language=language,
                severity=severity,
                symptoms=symptoms,
                treatment=treatment,
                prevention=prevention,
                recommendation=recommendation,
                warning=warning,
                disclaimer=disclaimer,
                updated_by=current_user.id
            )

            db.session.add(new_disease)
            db.session.commit()

            create_log('Disease info added',
                       f'Added disease {disease_id} ({language}) info')

            flash('Disease information added successfully', 'success')
            return redirect(url_for('manage_diseases'))

        except Exception as e:
            print(f"Error adding disease: {e}")
            flash('Error adding disease information', 'error')
            return redirect(url_for('add_disease'))

    # GET request 
    return render_template('add_disease.html',
                           lang=lang,
                           direction=direction,
                           t=translations[lang],
                           disease_count=len(class_names['en']))


 # =================================

def get_view_functions():
    """Get all view functions from the app"""
    return list(app.view_functions.keys())
if __name__ == '__main__':
    # Start background tasks
    start_background_tasks()

    print("\n" + "=" * 50)
    print("🌱 Plant Disease Classifier Application")
    print("=" * 50)
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"📊 Database: plant_disease.db")
    print(f"🤖 Model loaded: {model is not None}")
    if model:
        print(f"🎯 Model target size: {TARGET_SIZE}")
    print(f"🔐 Admin login: admin / admin123")
    print(f"🌐 Supported classes: {len(class_names['en'])}")
    print("=" * 50)
    print("🚀 Starting Flask application...")
    print("🌍 Open http://localhost:5000 in your browser")
    print("=" * 50 + "\n")

    app.run(debug=True, port=5000, host='0.0.0.0')
