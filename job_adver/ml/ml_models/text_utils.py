# text_utils.py
import re
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    return re.sub(r'\W+', ' ', text.lower()).strip()

def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def preprocess_profile(row):
    profile = (
        f"{row['skills']} {row['skills']} "
        f"{row['experience']} {row['description']} "
        f"{row['industry']} {row['industry']}"
    )
    return lemmatize_text(clean_text(profile))

def custom_analyzer(text):
    words = text.lower().split()
    result = list(words)

    domain_keywords = {
        'tech': ['python', 'java', 'javascript', 'html', 'css', 'react', 'angular', 'node', 'php', 'sql', 'cloud',
                 'aws', 'azure', 'developer', 'software', 'engineer', 'programming', 'django', 'flask'],
        'data': ['analyst', 'analytics', 'data', 'statistics', 'research', 'scientist', 'machine learning', 'ai',
                 'excel', 'bi', 'reporting', 'tableau'],
        'business': ['management', 'strategy', 'executive', 'project', 'operations', 'director', 'manager'],
        'sales': ['sales', 'marketing', 'customer', 'client', 'revenue', 'leads', 'seo', 'campaign']
    }

    full_text = ' '.join(words)
    for terms in domain_keywords.values():
        if any(term in full_text for term in terms):
            result += terms * 3  # Boost weight

    return result
