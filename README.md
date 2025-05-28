 CareerHub is a Django-based job portal integrated with Machine Learning. It predicts job titles, classifies job categories, and recommends roles based on user input like skills and experience. Includes data visualizations, admin tools, and ML-powered features using Logistic Regression, Random Forest, SVM, KNN, and MLP.

Welcome to **CareerHub**, a full-stack web platform built with **Django** and enhanced by **Machine Learning** to bridge the gap between **employers** and **job seekers** with intelligent predictions and automation.

---

## 🚀 Key Highlights

- 🔐 Unified **login system** for employers & employees  
- 📄 Post & browse **real-time job openings**  
- 🧠 Predict **job titles** based on user profile  
- 📊 Classify jobs into smart **job categories**  
- 🧪 Trained with 5 robust **ML algorithms**  
- 📈 View **model accuracy** via charts (Bar / Pie / Line)  
- 🧾 Organized dashboard for job tracking  

---

## 🧠 Machine Learning at Work

CareerHub is not just a job board – it's a predictive engine.  
The system learns from a dataset of job titles, descriptions, skills, and experience to make intelligent suggestions.

| Model                  | Description                          |
|------------------------|--------------------------------------|
| 🤖 Logistic Regression | Fast, accurate text classifier       |
| 🌲 Random Forest       | Ensemble-based decision model        |
| 🧠 MLP Classifier       | Neural network for feature-rich data |
| 💡 Support Vector Machine | High-margin classifier            |
| 👥 KNN Classifier       | Similarity-based job matching        |

> All models are trained using `TfidfVectorizer` on job profile text and saved as `.pkl` files in `/ml_models/`.

---

## 🖥️ Tech Stack

| Technology     | Purpose                         |
|----------------|----------------------------------|
| Django         | Backend framework (Python)       |
| HTML, CSS      | Frontend structure               |
| Bootstrap      | Responsive design                |
| Scikit-learn   | Machine learning models          |
| Matplotlib     | Chart rendering (accuracy graphs)|
| SQLite         | Lightweight database             |
| Joblib         | Model saving/loading             |

---

## 📊 Visual Insights

CareerHub includes an interactive **Model Evaluation Dashboard** where you can:

- 🔎 See which ML model performs best
- 📈 Compare accuracy visually (Bar / Line / Pie)
- 🎯 Understand prediction confidence & top suggestions
