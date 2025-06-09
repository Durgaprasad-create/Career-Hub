import os
import sys
import joblib
import numpy as np

MODEL_ACCURACIES = {
    'logreg': 0.88,
    'rf': 0.85,
    'mlp': 0.91,
    'svm': 0.83,
    'knn': 0.82
}

MODEL_NAME_MAP = {
    'logreg': 'Logistic Regression',
    'rf': 'Random Forest',
    'mlp': 'MLP Classifier',
    'svm': 'SVM',
    'knn': 'K-Nearest Neighbors'
}

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

try:
    from .ml_models.text_utils import clean_text, lemmatize_text
except ImportError:
    def clean_text(text): return str(text).lower().strip()
    def lemmatize_text(text): return clean_text(text)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'ml_models')

try:
    vectorizer = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
    model = joblib.load(os.path.join(MODEL_DIR, 'job_category_model.pkl'))
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    vectorizer, model = None, None

def get_context_aware_titles(category, text, override_type=None):
    """Generate context-aware titles for ALL categories based on text content"""
    text_lower = text.lower()
    
    if category == 'Software Development':
        if 'python' in text_lower:
            return ['Python Developer', 'Backend Developer', 'Software Engineer']
        elif 'react' in text_lower or 'frontend' in text_lower or 'ui' in text_lower:
            return ['Frontend Developer', 'React Developer', 'UI Developer']
        elif 'full stack' in text_lower or 'fullstack' in text_lower:
            return ['Full Stack Developer', 'Software Engineer', 'Web Developer']
        elif 'mobile' in text_lower or 'ios' in text_lower or 'android' in text_lower:
            return ['Mobile Developer', 'App Developer', 'iOS/Android Developer']
        elif 'architect' in text_lower:
            return ['Software Architect', 'Technical Architect', 'Solutions Architect']
        elif 'lead' in text_lower or 'senior' in text_lower:
            return ['Senior Software Engineer', 'Lead Developer', 'Technical Lead']
        else:
            return ['Software Developer', 'Software Engineer', 'Programmer']
    
    elif category == 'Healthcare':
        if 'therapy' in text_lower or 'therapist' in text_lower:
            return ['Physical Therapist', 'Occupational Therapist', 'Rehabilitation Specialist']
        elif 'nurse' in text_lower:
            return ['Registered Nurse', 'Clinical Nurse', 'Staff Nurse']
        elif 'doctor' in text_lower or 'physician' in text_lower:
            return ['Physician', 'Medical Doctor', 'Healthcare Provider']
        elif 'pharmacy' in text_lower or 'pharmacist' in text_lower:
            return ['Pharmacist', 'Clinical Pharmacist', 'Pharmacy Specialist']
        elif 'dental' in text_lower:
            return ['Dentist', 'Dental Hygienist', 'Dental Assistant']
        elif 'medical assistant' in text_lower:
            return ['Medical Assistant', 'Clinical Assistant', 'Healthcare Assistant']
        else:
            return ['Healthcare Professional', 'Medical Specialist', 'Clinical Professional']
    
    elif category == 'Finance':
        if 'analyst' in text_lower and 'financial' in text_lower:
            return ['Financial Analyst', 'Investment Analyst', 'Business Analyst']
        elif 'accountant' in text_lower or 'accounting' in text_lower:
            return ['Accountant', 'Senior Accountant', 'Staff Accountant']
        elif 'bank' in text_lower or 'banking' in text_lower:
            return ['Banking Associate', 'Loan Officer', 'Credit Analyst']
        elif 'investment' in text_lower:
            return ['Investment Advisor', 'Portfolio Manager', 'Investment Analyst']
        elif 'insurance' in text_lower:
            return ['Insurance Agent', 'Underwriter', 'Insurance Specialist']
        elif 'audit' in text_lower:
            return ['Auditor', 'Internal Auditor', 'Financial Auditor']
        elif 'controller' in text_lower or 'manager' in text_lower:
            return ['Financial Controller', 'Finance Manager', 'Accounting Manager']
        else:
            return ['Financial Professional', 'Finance Specialist', 'Financial Analyst']
    
    elif category == 'Data & Analytics':
        if 'data scientist' in text_lower or 'machine learning' in text_lower:
            return ['Data Scientist', 'ML Engineer', 'AI Specialist']
        elif 'data analyst' in text_lower or 'business intelligence' in text_lower:
            return ['Data Analyst', 'Business Intelligence Analyst', 'BI Developer']
        elif 'database' in text_lower or 'dba' in text_lower:
            return ['Database Administrator', 'Database Engineer', 'Data Engineer']
        elif 'research' in text_lower:
            return ['Research Analyst', 'Market Research Analyst', 'Data Researcher']
        elif 'engineer' in text_lower:
            return ['Data Engineer', 'Analytics Engineer', 'ML Engineer']
        else:
            return ['Data Analyst', 'Data Professional', 'Analytics Specialist']
    
    elif category == 'Sales & Marketing':
        if 'digital marketing' in text_lower or 'seo' in text_lower:
            return ['Digital Marketing Manager', 'SEO Specialist', 'Digital Marketing Specialist']
        elif 'account manager' in text_lower or 'account' in text_lower:
            return ['Account Manager', 'Key Account Manager', 'Sales Account Executive']
        elif 'social media' in text_lower:
            return ['Social Media Manager', 'Social Media Specialist', 'Community Manager']
        elif 'content' in text_lower:
            return ['Content Marketing Manager', 'Content Specialist', 'Marketing Content Creator']
        elif 'business development' in text_lower:
            return ['Business Development Manager', 'BD Representative', 'Growth Manager']
        elif 'sales rep' in text_lower or 'representative' in text_lower:
            return ['Sales Representative', 'Sales Associate', 'Account Executive']
        else:
            return ['Marketing Specialist', 'Sales Professional', 'Marketing Manager']
    
    elif category == 'Business Management':
        if 'project manager' in text_lower:
            return ['Project Manager', 'Program Manager', 'Project Coordinator']
        elif 'operations' in text_lower:
            return ['Operations Manager', 'Operations Director', 'Business Operations Manager']
        elif 'consultant' in text_lower:
            return ['Management Consultant', 'Business Consultant', 'Strategy Consultant']
        elif 'director' in text_lower:
            return ['Business Director', 'Operations Director', 'General Manager']
        elif 'analyst' in text_lower:
            return ['Business Analyst', 'Management Analyst', 'Process Analyst']
        else:
            return ['Business Manager', 'Manager', 'Team Lead']
    
    elif category == 'HR & Recruitment':
        if 'recruiter' in text_lower or 'recruitment' in text_lower:
            return ['Recruiter', 'Talent Acquisition Specialist', 'Hiring Manager']
        elif 'hr generalist' in text_lower:
            return ['HR Generalist', 'Human Resources Generalist', 'HR Specialist']
        elif 'compensation' in text_lower:
            return ['Compensation Analyst', 'Benefits Administrator', 'Total Rewards Specialist']
        elif 'training' in text_lower:
            return ['Training Coordinator', 'Learning & Development Specialist', 'HR Trainer']
        else:
            return ['HR Professional', 'Human Resources Specialist', 'HR Manager']
    
    elif category == 'IT Operations':
        # CYBERSECURITY TITLES
        if override_type == 'security' or any(term in text_lower for term in ['security', 'cybersecurity', 'penetration', 'vulnerability']):
            if 'penetration' in text_lower or 'pentest' in text_lower:
                return ['Penetration Tester', 'Ethical Hacker', 'Security Consultant']
            elif 'incident response' in text_lower:
                return ['Incident Response Analyst', 'SOC Analyst', 'Security Operations Analyst']
            elif 'manager' in text_lower or 'lead' in text_lower:
                return ['Cybersecurity Manager', 'Security Team Lead', 'Information Security Manager']
            elif 'architect' in text_lower:
                return ['Security Architect', 'Cybersecurity Architect', 'Information Security Architect']
            elif 'audit' in text_lower:
                return ['Cybersecurity Auditor', 'Security Compliance Auditor', 'IT Security Auditor']
            elif 'cloud' in text_lower:
                return ['Cloud Security Engineer', 'Cloud Security Specialist', 'Cloud Security Architect']
            elif 'network' in text_lower:
                return ['Network Security Engineer', 'Network Security Analyst', 'Security Network Specialist']
            else:
                return ['Cybersecurity Analyst', 'Security Analyst', 'Information Security Analyst']
        
        # CLOUD/DEVOPS TITLES
        elif override_type == 'cloud' or any(term in text_lower for term in ['cloud', 'devops', 'kubernetes', 'docker']):
            if 'architect' in text_lower:
                return ['Cloud Architect', 'Solutions Architect', 'Infrastructure Architect']
            elif 'devops' in text_lower:
                return ['DevOps Engineer', 'DevOps Specialist', 'Platform Engineer']
            elif 'kubernetes' in text_lower or 'docker' in text_lower:
                return ['Container Engineer', 'Kubernetes Engineer', 'Platform Engineer']
            elif 'aws' in text_lower:
                return ['AWS Engineer', 'AWS Solutions Architect', 'Cloud Engineer']
            elif 'azure' in text_lower:
                return ['Azure Engineer', 'Azure Solutions Architect', 'Cloud Engineer']
            else:
                return ['Cloud Engineer', 'Infrastructure Engineer', 'Cloud Operations Engineer']
        
        # TRADITIONAL IT TITLES
        elif 'system administrator' in text_lower or 'sysadmin' in text_lower:
            return ['System Administrator', 'Systems Engineer', 'IT Administrator']
        elif 'network' in text_lower:
            return ['Network Engineer', 'Network Administrator', 'Network Specialist']
        elif 'support' in text_lower:
            return ['IT Support Specialist', 'Technical Support Engineer', 'Help Desk Specialist']
        else:
            return ['IT Professional', 'Technology Specialist', 'Systems Professional']
    
    elif category == 'Education':
        if 'teacher' in text_lower:
            return ['Teacher', 'Educator', 'Instructor']
        elif 'professor' in text_lower:
            return ['Professor', 'Associate Professor', 'Academic Faculty']
        elif 'trainer' in text_lower:
            return ['Corporate Trainer', 'Training Specialist', 'Learning Facilitator']
        elif 'curriculum' in text_lower:
            return ['Curriculum Developer', 'Instructional Designer', 'Education Specialist']
        else:
            return ['Education Professional', 'Academic Professional', 'Instructor']
    
    elif category == 'Legal':
        if 'attorney' in text_lower or 'lawyer' in text_lower:
            return ['Attorney', 'Lawyer', 'Legal Counsel']
        elif 'paralegal' in text_lower:
            return ['Paralegal', 'Legal Assistant', 'Paralegal Specialist']
        elif 'compliance' in text_lower:
            return ['Compliance Officer', 'Legal Compliance Specialist', 'Regulatory Compliance Manager']
        elif 'contract' in text_lower:
            return ['Contract Specialist', 'Contract Attorney', 'Legal Contract Manager']
        else:
            return ['Legal Professional', 'Legal Specialist', 'Legal Advisor']
    
    elif category == 'Creative & Design':
        if 'graphic designer' in text_lower:
            return ['Graphic Designer', 'Visual Designer', 'Brand Designer']
        elif 'ux' in text_lower or 'ui' in text_lower:
            return ['UX/UI Designer', 'Product Designer', 'User Experience Designer']
        elif 'web designer' in text_lower:
            return ['Web Designer', 'Digital Designer', 'Frontend Designer']
        elif 'creative director' in text_lower:
            return ['Creative Director', 'Art Director', 'Design Director']
        elif 'photographer' in text_lower:
            return ['Photographer', 'Visual Content Creator', 'Digital Photographer']
        else:
            return ['Creative Professional', 'Designer', 'Creative Specialist']
    
    elif category == 'Manufacturing & Production':
        if 'quality' in text_lower:
            return ['Quality Control Inspector', 'QA Specialist', 'Quality Assurance Manager']
        elif 'engineer' in text_lower:
            return ['Manufacturing Engineer', 'Process Engineer', 'Industrial Engineer']
        elif 'supervisor' in text_lower or 'manager' in text_lower:
            return ['Production Supervisor', 'Manufacturing Manager', 'Operations Supervisor']
        else:
            return ['Production Specialist', 'Manufacturing Professional', 'Operations Professional']
    
    elif category == 'Customer Service':
        if 'call center' in text_lower:
            return ['Call Center Agent', 'Customer Service Representative', 'Phone Support Agent']
        elif 'technical support' in text_lower:
            return ['Technical Support Representative', 'Technical Support Specialist', 'IT Support Agent']
        elif 'customer success' in text_lower:
            return ['Customer Success Manager', 'Client Success Specialist', 'Account Success Manager']
        else:
            return ['Customer Service Representative', 'Support Specialist', 'Customer Care Agent']
    
    elif category == 'Transportation & Logistics':
        if 'logistics' in text_lower:
            return ['Logistics Coordinator', 'Logistics Manager', 'Supply Chain Specialist']
        elif 'warehouse' in text_lower:
            return ['Warehouse Manager', 'Warehouse Supervisor', 'Warehouse Operations Manager']
        elif 'driver' in text_lower:
            return ['Delivery Driver', 'Truck Driver', 'Transportation Driver']
        else:
            return ['Logistics Professional', 'Transportation Specialist', 'Supply Chain Professional']
    
    elif category == 'Real Estate':
        if 'agent' in text_lower:
            return ['Real Estate Agent', 'Sales Agent', 'Realtor']
        elif 'property manager' in text_lower:
            return ['Property Manager', 'Real Estate Manager', 'Property Management Specialist']
        elif 'broker' in text_lower:
            return ['Real Estate Broker', 'Property Broker', 'Commercial Real Estate Broker']
        else:
            return ['Real Estate Professional', 'Property Specialist', 'Real Estate Consultant']
    
    elif category == 'Media & Communications':
        if 'journalist' in text_lower or 'reporter' in text_lower:
            return ['Journalist', 'Reporter', 'News Writer']
        elif 'public relations' in text_lower or 'pr' in text_lower:
            return ['Public Relations Specialist', 'PR Manager', 'Communications Manager']
        elif 'content creator' in text_lower:
            return ['Content Creator', 'Digital Content Specialist', 'Social Media Content Creator']
        else:
            return ['Communications Specialist', 'Media Professional', 'Content Professional']
    
    # DEFAULT FALLBACK
    else:
        return ['Professional', 'Specialist', 'Expert']

def smart_override_check(text):
    """ENHANCED smart override for obvious misclassifications"""
    text_lower = text.lower()
    
    # Rule 1: Strong cybersecurity indicators
    security_terms = ['cybersecurity', 'penetration testing', 'vulnerability assessment', 'incident response', 'siem', 'ids/ips', 'wireshark', 'kali linux']
    security_count = sum(1 for term in security_terms if term in text_lower)
    
    # Rule 2: Strong cloud/devops indicators  
    cloud_terms = ['devops', 'kubernetes', 'docker', 'terraform', 'jenkins', 'ansible', 'cloud infrastructure', 'aws', 'azure']
    cloud_count = sum(1 for term in cloud_terms if term in text_lower)
    
    # Rule 3: ENHANCED healthcare indicators
    health_terms = [
        'physical therapy', 'physical therapist', 'therapist', 'therapy',
        'registered nurse', 'nurse', 'nursing', 'registered nurse',
        'medical', 'clinical', 'patient care', 'healthcare', 'health care',
        'doctor', 'physician', 'md', 'hospital', 'clinic',
        'pharmacy', 'pharmacist', 'dental', 'dentist',
        'occupational therapy', 'respiratory therapy', 'speech therapy',
        'medical assistant', 'paramedic', 'emt', 'radiology', 'laboratory'
    ]
    health_count = sum(1 for term in health_terms if term in text_lower)
    
    # Rule 4: ENHANCED software development indicators
    software_terms = [
        'python developer', 'java developer', 'javascript developer',
        'react developer', 'angular developer', 'vue developer',
        'frontend developer', 'backend developer', 'full stack developer',
        'software engineer', 'web developer', 'mobile developer',
        'ios developer', 'android developer', 'flutter developer'
    ]
    software_count = sum(1 for term in software_terms if term in text_lower)
    
    # Rule 5: ENHANCED finance indicators
    finance_terms = [
        'financial analyst', 'investment analyst', 'business analyst',
        'accountant', 'accounting', 'cpa', 'auditor', 'audit',
        'banking', 'bank', 'finance', 'financial', 'investment',
        'portfolio manager', 'wealth advisor', 'credit analyst'
    ]
    finance_count = sum(1 for term in finance_terms if term in text_lower)
    
    # Override decisions with LOWER thresholds
    if health_count >= 1:  # Just 1 health term needed!
        return 'Healthcare', 0.95, f"Healthcare Override: {health_count} indicators ({[term for term in health_terms if term in text_lower][:2]})", 'healthcare'
    elif software_count >= 1:  # Just 1 software term needed!
        return 'Software Development', 0.90, f"Software Override: {software_count} indicators", 'software'
    elif finance_count >= 1 and 'analyst' in text_lower:  # Finance + analyst
        return 'Finance', 0.95, f"Finance Override: {finance_count} indicators", 'finance'
    elif security_count >= 2:  # Reduced from 3 to 2
        return 'IT Operations', 0.90, f"Security Override: {security_count} indicators", 'security'
    elif cloud_count >= 2:  # Reduced from 3 to 2
        return 'IT Operations', 0.85, f"Cloud Override: {cloud_count} indicators", 'cloud'
    
    return None, None, None, None

def predict_job_category_and_titles(text):
    """HYBRID prediction with AGGRESSIVE THRESHOLDS to prevent Data & Analytics domination"""
    if not vectorizer or not model:
        return {
            'predicted_category': 'Other Roles', 
            'confidence': 0.0, 
            'suggested_titles': ['Professional'], 
            'top_categories': []
        }
    
    try:
        # Step 1: Check for obvious overrides
        override_category, override_confidence, override_reason, override_type = smart_override_check(text)
        
        if override_category:
            print(f"🎯 {override_reason}")
            return {
                'predicted_category': override_category,
                'confidence': override_confidence,
                'suggested_titles': get_context_aware_titles(override_category, text, override_type),
                'top_categories': [(override_category, override_confidence)]
            }
        
        # Step 2: Use ML model with AGGRESSIVE threshold system
        processed = lemmatize_text(clean_text(text))
        vec = vectorizer.transform([processed])
        
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(vec)[0]
            classes = model.classes_[np.argsort(probas)[::-1]]
            scores = probas[np.argsort(probas)[::-1]]
            
            # AGGRESSIVE THRESHOLDS - Prevent Data & Analytics domination
            thresholds = {
                # ULTRA-LOW for minority classes
                'Healthcare': 0.05,               # Ultra-low (was 0.08)
                'Legal': 0.05,                   # Ultra-low (was 0.08)
                'Education': 0.05,               # Ultra-low (was 0.08)
                'HR & Recruitment': 0.08,        # Very low (was 0.15)
                'Transportation & Logistics': 0.08, # Very low
                'Real Estate': 0.08,             # Very low
                
                # LOW for small classes
                'Finance': 0.10,                 # Low (was 0.12)
                'Creative & Design': 0.10,       # Low (was 0.15)
                'Manufacturing & Production': 0.12, # Low
                'Customer Service': 0.12,        # Low
                'Media & Communications': 0.12,  # Low
                
                # MEDIUM for technical classes
                'IT Operations': 0.15,           # Medium-low (was 0.20)
                'Software Development': 0.18,    # Medium (was 0.25)
                'Sales & Marketing': 0.22,       # Medium (was 0.30)
                'Business Management': 0.25,     # Medium-high (was 0.35)
                
                # HIGH for majority class - Force very high confidence
                'Data & Analytics': 0.50,        # VERY HIGH (was 0.35) - Must be very confident!
            }
            
            category, confidence = classes[0], scores[0]  # Default to highest probability
            
            # Find first category that meets its threshold
            for cls, score in zip(classes, scores):
                threshold = thresholds.get(cls, 0.30)
                if score >= threshold:
                    category, confidence = cls, score
                    print(f"✅ Threshold Match: {cls} ({score:.3f} >= {threshold})")
                    break
            
            top_categories = [(classes[i], float(scores[i])) for i in range(min(3, len(scores)))]
        else:
            category, confidence = model.predict(vec)[0], 1.0
            top_categories = [(category, confidence)]
        
        return {
            'predicted_category': category,
            'confidence': float(confidence),
            'suggested_titles': get_context_aware_titles(category, text),
            'top_categories': top_categories
        }
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return {
            'predicted_category': 'Other Roles', 
            'confidence': 0.0, 
            'suggested_titles': ['Professional'], 
            'top_categories': []
        }

def category_to_titles(category):
    """Backward compatibility function"""
    return get_context_aware_titles(category, "")