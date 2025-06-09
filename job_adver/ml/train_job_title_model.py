import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from collections import Counter
from imblearn.over_sampling import SMOTE
from ml_models.text_utils import clean_text, lemmatize_text

def load_dataset(path):
    df = pd.read_csv(path)
    for col in ['jobtitle', 'experience', 'skills', 'description', 'industry']:
        df[col] = df[col].fillna('') if col in df.columns else ''
    print(f"📋 Dataset: {df.shape}, Columns: {list(df.columns)}")
    return df

def categorize_job_title(title):
    title = str(title).lower()
    categories = {
        'Software Development': [
            'developer', 'engineer', 'software', 'programming', 'python', 'java', 'react', 'javascript', 
            'node', 'angular', 'vue', 'frontend', 'backend', 'fullstack', 'web', 'app', 'mobile', 
            'ios', 'android', 'flutter', 'kotlin', 'swift', 'c++', 'c#', '.net', 'php', 'ruby', 
            'golang', 'rust', 'scala', 'typescript', 'api', 'microservices', 'blockchain', 'coding'
        ],
        'Healthcare': [
            'nurse', 'nursing', 'registered nurse','lpn', 'doctor', 'physician', 'md', 'medical', 'healthcare', 
            'clinical', 'therapy', 'therapist', 'pharmacy', 'pharmacist', 'dental', 'dentist', 
            'surgeon', 'surgery', 'radiologist', 'cardiologist', 'neurologist', 'psychiatrist', 
            'psychologist', 'counselor', 'physical therapy', 'occupational therapy', 
            'respiratory therapy', 'lab technician', 'medical assistant', 'paramedic', 'emt', 
            'veterinarian', 'optometrist', 'chiropractor', 'patient care', 'hospital', 'clinic'
        ],
        'Finance': [
            'finance', 'financial', 'accountant', 'accounting', 'bank', 'banking', 'investment', 
            'analyst', 'cpa', 'cfa', 'auditor', 'audit', 'tax', 'treasury', 'credit', 'loan', 
            'mortgage', 'insurance', 'actuary', 'underwriter', 'portfolio', 'wealth', 'advisor', 
            'broker', 'trader', 'equity', 'bond', 'derivatives', 'risk', 'compliance', 'controller', 
            'bookkeeper', 'payroll', 'budget', 'forecasting', 'financial planning', 'fintech'
        ],
        'Data & Analytics': [
            'data', 'analytics', 'scientist', 'machine learning', 'ml', 'ai', 'artificial intelligence', 
            'database', 'sql', 'nosql', 'bi', 'business intelligence', 'tableau', 'powerbi', 'excel', 
            'statistics', 'statistician', 'research', 'big data', 'hadoop', 'spark', 'kafka', 
            'etl', 'data warehouse', 'data mining', 'predictive', 'reporting', 'visualization', 
            'python', 'r', 'sas', 'spss', 'tensorflow', 'pytorch', 'deep learning', 'nlp'
        ],
        'Sales & Marketing': [
            'sales', 'marketing', 'account', 'client', 'customer', 'brand', 'digital', 'seo', 
            'sem', 'social media', 'content', 'copywriter', 'campaign', 'advertising', 'promotion', 
            'business development', 'lead generation', 'crm', 'salesforce', 'hubspot', 'email marketing', 
            'growth', 'acquisition', 'retention', 'conversion', 'funnel', 'roi', 'kpi', 'market research', 
            'competitive analysis', 'product marketing', 'field sales', 'inside sales', 'territory'
        ],
        'Business Management': [
            'manager', 'management', 'executive', 'director', 'ceo', 'coo', 'cfo', 'cto', 'vp', 
            'president', 'operations', 'strategy', 'consulting', 'consultant', 'project manager', 
            'program manager', 'product manager', 'business analyst', 'process improvement', 
            'lean', 'six sigma', 'agile', 'scrum', 'kanban', 'change management', 'transformation', 
            'leadership', 'team lead', 'supervisor', 'coordinator', 'administrator', 'planning'
        ],
        'HR & Recruitment': [
            'hr', 'human resources', 'recruit', 'recruiter', 'recruitment', 'talent', 'hiring', 
            'staffing', 'personnel', 'onboarding', 'training', 'development', 'compensation', 
            'benefits', 'payroll', 'employee relations', 'performance', 'engagement', 'culture', 
            'diversity', 'inclusion', 'wellness', 'hris', 'workday', 'adp', 'succession planning', 
            'talent acquisition', 'sourcing', 'screening', 'interviewing', 'background check'
        ],
        'IT Operations': [
            'devops', 'sysadmin', 'system administrator', 'network', 'infrastructure', 'support', 
            'security', 'cybersecurity', 'cloud', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 
            'jenkins', 'ansible', 'terraform', 'monitoring', 'logging', 'backup', 'disaster recovery', 
            'helpdesk', 'technical support', 'it support', 'server', 'database administrator', 
            'dba', 'linux', 'windows', 'unix', 'vmware', 'virtualization', 'storage', 'networking'
        ],
        'Education': [
            'teacher', 'instructor', 'professor', 'education', 'educator', 'school', 'university', 
            'college', 'academic', 'curriculum', 'principal', 'dean', 'superintendent', 'tutor', 
            'trainer', 'facilitator', 'learning', 'development', 'elearning', 'online learning', 
            'instructional design', 'course', 'classroom', 'student', 'faculty', 'research', 
            'scholarship', 'thesis', 'dissertation', 'pedagogy', 'assessment', 'evaluation'
        ],
        'Legal': [
            'lawyer', 'attorney', 'legal', 'law', 'counsel', 'paralegal', 'litigation', 'contract', 
            'corporate law', 'criminal law', 'family law', 'immigration law', 'intellectual property', 
            'patent', 'trademark', 'copyright', 'compliance', 'regulatory', 'court', 'judge', 
            'magistrate', 'clerk', 'legal assistant', 'legal secretary', 'notary', 'mediator', 
            'arbitrator', 'legal research', 'case management', 'discovery', 'deposition'
        ],
        'Creative & Design': [
            'designer', 'graphic designer', 'web designer', 'ui designer', 'ux designer', 'creative', 
            'ux', 'ui', 'user experience', 'user interface', 'wireframe', 'prototype',
            'artist', 'illustrator', 'animator', 'video editor', 'photographer', 'videographer', 
            'art director', 'creative director', 'brand designer', 'logo designer', 'print designer', 
            'digital designer', 'motion graphics', 'visual designer', 'product designer', 'fashion designer', 
            'interior designer', 'architect', 'layout', 'typography', 'color theory', 'adobe', 
            'photoshop', 'illustrator', 'indesign', 'figma', 'sketch', 'canva', 'creative writing'
        ],
        'Manufacturing & Production': [
            'manufacturing', 'production', 'assembly', 'quality control', 'quality assurance', 
            'process engineer', 'industrial engineer', 'mechanical engineer', 'electrical engineer', 
            'factory', 'plant', 'operations', 'maintenance', 'technician', 'machinist', 'welder', 
            'fabricator', 'forklift', 'warehouse', 'logistics', 'supply chain', 'procurement', 
            'inventory', 'lean manufacturing', 'six sigma', 'kaizen', 'safety', 'osha'
        ],
        'Customer Service': [
            'customer service', 'customer support', 'call center', 'help desk', 'support specialist', 
            'client services', 'customer success', 'account management', 'relationship management', 
            'service representative', 'support agent', 'technical support', 'chat support', 
            'phone support', 'email support', 'escalation', 'resolution', 'satisfaction', 
            'retention', 'loyalty', 'feedback', 'survey', 'crm', 'ticketing system'
        ],
        'Transportation & Logistics': [
            'driver', 'truck driver', 'delivery', 'shipping', 'logistics', 'transportation', 
            'supply chain', 'warehouse', 'distribution', 'freight', 'cargo', 'fleet', 'dispatch', 
            'route planning', 'inventory management', 'materials handling', 'forklift operator', 
            'loading', 'unloading', 'packaging', 'fulfillment', 'courier', 'pilot', 'aviation', 
            'maritime', 'port', 'customs', 'import', 'export', 'international trade'
        ],
        'Real Estate': [
            'real estate', 'realtor', 'agent', 'broker', 'property', 'leasing', 'rental', 
            'property management', 'commercial real estate', 'residential', 'appraisal', 'appraiser', 
            'mortgage', 'loan officer', 'title', 'escrow', 'closing', 'inspection', 'valuation', 
            'investment property', 'development', 'construction', 'zoning', 'permits', 'listings'
        ],
        'Media & Communications': [
            'journalist', 'reporter', 'editor', 'writer', 'content creator', 'blogger', 'copywriter', 
            'communications', 'public relations', 'pr', 'media', 'broadcasting', 'radio', 'television', 
            'podcast', 'social media', 'digital media', 'content marketing', 'storytelling', 
            'news', 'publishing', 'editorial', 'proofreading', 'translation', 'interpreter'
        ]
    }
    
    # ULTRA-AGGRESSIVE weights for minority and high-skill categories
    weights = {
        'Healthcare': 10, 'Finance': 8, 'Legal': 8, 'Education': 6, 
        'Data & Analytics': 4, 'Software Development': 3, 'Creative & Design': 10,
        'Manufacturing & Production': 4, 'Transportation & Logistics': 3,
        'Real Estate': 4, 'Media & Communications': 4
    }
    
    scores = {}
    for cat, keywords in categories.items():
        weight = weights.get(cat, 1)
        scores[cat] = sum(weight * (4 if title.startswith(kw) else 2) 
                         for kw in keywords if kw in title)
    
    return max(scores.items(), key=lambda x: x[1])[0] if max(scores.values()) > 0 else 'Other Roles'

def preprocess_profile(row):
    job_title = str(row['jobtitle']).lower()
    boost_terms = {
        'healthcare': 15, 'medical': 15, 'nurse': 15, 'doctor': 15,
        'finance': 10, 'bank': 10, 'financial': 10, 'accounting': 10,
        'education': 8, 'teacher': 8, 'legal': 8, 'law': 8
    }
    multiplier = next((boost for term, boost in boost_terms.items() if term in job_title), 4)
    
    profile = f"{' '.join([job_title] * multiplier)} {row.get('skills', '')} {row.get('industry', '')} {row.get('experience', '')}"
    return lemmatize_text(clean_text(profile))

def train_all_models(X_train_tfidf, y_train, X_test_tfidf, y_test):
    # SMOTE balance
    smote = SMOTE(sampling_strategy='auto', k_neighbors=1, random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_tfidf, y_train)
    X_train_dense, X_test_dense = X_train_balanced.toarray(), X_test_tfidf.toarray()
    
    print(f"Training: {X_train_balanced.shape[0]} samples, {X_train_balanced.shape[1]} features")
    
    # ALL 5 MODELS - Ultra-compact configs
    models = {
        'lr': (LogisticRegression(max_iter=15000, C=8.0, solver='liblinear', class_weight='balanced', random_state=42), 'sparse'),
        'rf': (RandomForestClassifier(n_estimators=200, max_depth=25, class_weight='balanced_subsample', random_state=42, n_jobs=-1), 'sparse'),
        'svm': (SVC(C=15.0, kernel='linear', class_weight='balanced', probability=True, random_state=42), 'sparse'),
        'knn': (KNeighborsClassifier(n_neighbors=7, weights='distance', metric='cosine', n_jobs=-1), 'dense'),
        'mlp': (MLPClassifier(hidden_layer_sizes=(80,), max_iter=500, alpha=0.01, solver='adam', random_state=42, early_stopping=True), 'dense')
    }

    trained_models, model_scores, best_score, best_model = {}, {}, 0, None
    
    for name, (model, data_type) in models.items():
        try:
            X_train_data = X_train_dense if data_type == 'dense' else X_train_balanced
            X_test_data = X_test_dense if data_type == 'dense' else X_test_tfidf
            
            model.fit(X_train_data, y_train_balanced)
            score = f1_score(y_test, model.predict(X_test_data), average='weighted')
            
            model_scores[name] = score
            trained_models[name] = model
            print(f"✅ {name.upper()}: {score:.4f}")
            
            if score > best_score:
                best_score, best_model = score, model
        except Exception as e:
            print(f"❌ {name.upper()}: {str(e)[:30]}...")
            model_scores[name] = 0.0

    # ENSEMBLE from top sparse-compatible models
    top_models = sorted([(n, s) for n, s in model_scores.items() if n in ['lr', 'rf', 'svm'] and s > 0], 
                       key=lambda x: x[1], reverse=True)[:3]
    
    if len(top_models) >= 2:
        try:
            ensemble = VotingClassifier([(name, trained_models[name]) for name, _ in top_models], voting='soft')
            ensemble.fit(X_train_balanced, y_train_balanced)
            ensemble_score = f1_score(y_test, ensemble.predict(X_test_tfidf), average='weighted')
            print(f"🎯 ENSEMBLE: {ensemble_score:.4f}")
            
            if ensemble_score > best_score:
                best_model, best_score = ensemble, ensemble_score
        except Exception as e:
            print(f"❌ Ensemble failed: {e}")
    
    return trained_models, best_model, model_scores

def get_smart_titles(category, text):
    patterns = {
        'Healthcare': {
            ('therapy', 'rehab', 'physical therapy'): ['Physical Therapist', 'Rehabilitation Specialist'], 
            ('nurse', 'rn', 'nursing'): ['Registered Nurse', 'Clinical Nurse'], 
            ('doctor', 'md', 'physician'): ['Physician', 'Medical Doctor'],
            ('pharmacy', 'pharmacist'): ['Pharmacist', 'Clinical Pharmacist'],
            ('dental', 'dentist'): ['Dentist', 'Dental Hygienist']
        },
        'Finance': {
            ('analyst', 'investment', 'financial analyst'): ['Financial Analyst', 'Investment Analyst'], 
            ('accountant', 'cpa', 'accounting'): ['Accountant', 'Senior Accountant'], 
            ('bank', 'banking', 'credit'): ['Banking Associate', 'Loan Officer'],
            ('insurance', 'underwriter'): ['Insurance Agent', 'Underwriter']
        },
        'Software Development': {
            ('python', 'backend', 'django'): ['Python Developer', 'Backend Developer'], 
            ('react', 'frontend', 'javascript'): ['Frontend Developer', 'UI Developer'], 
            ('full stack', 'fullstack'): ['Full Stack Developer', 'Software Engineer'],
            ('mobile', 'ios', 'android'): ['Mobile Developer', 'App Developer']
        },
        'Data & Analytics': {
            ('data scientist', 'machine learning', 'ml'): ['Data Scientist', 'ML Engineer'],
            ('data analyst', 'analytics', 'bi'): ['Data Analyst', 'Business Intelligence Analyst'],
            ('database', 'sql', 'dba'): ['Database Administrator', 'Data Engineer']
        }
    }
    
    defaults = {
        'Healthcare': ['Healthcare Professional', 'Medical Specialist'], 
        'Finance': ['Finance Professional', 'Financial Specialist'],
        'Software Development': ['Software Developer', 'Software Engineer'], 
        'Data & Analytics': ['Data Analyst', 'Analytics Specialist'],
        'Sales & Marketing': ['Marketing Specialist', 'Sales Professional'], 
        'Business Management': ['Business Manager', 'Management Professional'],
        'HR & Recruitment': ['HR Professional', 'Human Resources'], 
        'IT Operations': ['IT Professional', 'Technical Specialist'],
        'Education': ['Education Professional', 'Academic Specialist'], 
        'Legal': ['Legal Professional', 'Legal Specialist'],
        'Creative & Design': ['Creative Professional', 'Design Specialist','UI/UX Developer'],
        'Manufacturing & Production': ['Manufacturing Professional', 'Production Specialist'],
        'Customer Service': ['Customer Service Representative', 'Support Specialist'],
        'Transportation & Logistics': ['Logistics Professional', 'Transportation Specialist'],
        'Real Estate': ['Real Estate Professional', 'Property Specialist'],
        'Media & Communications': ['Media Professional', 'Communications Specialist']
    }
    
    text = text.lower()
    if category in patterns:
        for keywords, titles in patterns[category].items():
            if any(kw in text for kw in keywords):
                return titles
    
    return defaults.get(category, ['Professional'])

def predict_job_category_and_titles(text, vectorizer=None, model=None):
    if not vectorizer:
        vectorizer = joblib.load('ml_models/tfidf_vectorizer.pkl')
        model = joblib.load('ml_models/job_category_model.pkl')
    
    processed = lemmatize_text(clean_text(text))
    vec = vectorizer.transform([processed])
    
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(vec)[0]
        classes = model.classes_[np.argsort(probas)[::-1]]
        scores = probas[np.argsort(probas)[::-1]]
        
        thresholds = {
            'Healthcare': 0.08, 'Finance': 0.12, 'Education': 0.08, 'Legal': 0.08,
            'Creative & Design': 0.15, 'HR & Recruitment': 0.2, 'Sales & Marketing': 0.25,
            'IT Operations': 0.35, 'Data & Analytics': 0.4, 'Business Management': 0.4,
            'Software Development': 0.5, 'Manufacturing & Production': 0.3,
            'Customer Service': 0.25, 'Transportation & Logistics': 0.3,
            'Real Estate': 0.3, 'Media & Communications': 0.3
        }
        category, confidence = classes[0], scores[0]
        
        for cls, score in zip(classes, scores):
            if score >= thresholds.get(cls, 0.3):
                category, confidence = cls, score
                break
    else:
        category, confidence = model.predict(vec)[0], 1.0
    
    return {
        'predicted_category': category,
        'confidence': float(confidence),
        'suggested_titles': get_smart_titles(category, text)
    }

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    print("🚀 Training ALL 5 Algorithms (16 Categories)")
    
    # Load and process
    df = load_dataset("Datasets.csv")
    df['job_category'] = df['jobtitle'].apply(categorize_job_title)
    df['job_profile'] = df.apply(preprocess_profile, axis=1)
    
    print(f"Categories: {df['job_category'].value_counts().to_dict()}")
    
    # Train/test split
    X, y = df['job_profile'], df['job_category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Vectorize
    vectorizer = TfidfVectorizer(
        max_features=8000, 
        min_df=2, 
        max_df=0.85, 
        ngram_range=(1, 2), 
        stop_words='english', 
        analyzer='word'
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"Training: {X_train_tfidf.shape[0]} samples, {X_train_tfidf.shape[1]} features")
    
    # Train models
    trained_models, final_model, scores = train_all_models(X_train_tfidf, y_train, X_test_tfidf, y_test)
    
    # Save models
    os.makedirs('ml_models', exist_ok=True)
    joblib.dump(vectorizer, 'ml_models/tfidf_vectorizer.pkl')
    joblib.dump(final_model, 'ml_models/job_category_model.pkl')
    
    # ===== FIXED TEST SECTION - Use ml_utils.py =====
    print("\n🧪 Testing with AGGRESSIVE THRESHOLDS:")
    print("=" * 60)
    
    try:
        # Import the updated prediction function
        from ml_utils import predict_job_category_and_titles
        
        test_cases = [
            "Physical Therapist with 5 years experience in rehabilitation and patient care",
            "Financial Analyst specializing in investment portfolio management and risk assessment", 
            "Python Developer with expertise in Django, React, and full-stack web development",
            "Data Scientist with machine learning, deep learning, and statistical modeling experience"
        ]
        
        for test_text in test_cases:
            result = predict_job_category_and_titles(test_text)
            job_title = test_text.split()[0] + " " + test_text.split()[1]
            print(f"📝 {job_title} → {result['predicted_category']} ({result['confidence']:.3f})")
            print(f"   💼 Suggested Titles: {', '.join(result['suggested_titles'][:2])}")
            print()
        
    except ImportError as e:
        print(f"⚠️  Cannot import ml_utils.py: {e}")
        print("⚠️  Please run this after the Django app is set up.")
        print("⚠️  The models are trained and saved successfully!")
    
    print(f"🎉 Training Complete! Final Scores: {scores}")
    print("✅ Models saved to ml_models/ directory")
    print("🔧 Test the web application to see aggressive thresholds in action!")