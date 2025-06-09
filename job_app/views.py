from django.shortcuts import render,redirect
from django.http import HttpResponseForbidden
from django.http import HttpResponse
from job_app.models import Job,Profile,JobApplication
from django.contrib.auth import login,authenticate,logout
from .forms import EmployerRegistrationForm, EmployeeRegistrationForm,LoginForm,JobForm,JobApplicationForm
from django.contrib.auth.decorators import login_required
from .forms import ProfileForm,Profile
from django.db.models import Q
from django.shortcuts import get_object_or_404
from django.contrib import messages
from job_adver.ml.ml_utils import  predict_job_category_and_titles
from django.contrib.auth import get_user_model
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import io
import urllib, base64
def home(request):
    latest_jobs = Job.objects.order_by('-posted_on')[:3]
    return render(request, 'job_app/home.html', {'Jobs': latest_jobs})

def job_list(request):
    keyword = request.GET.get('keyword', '')
    location = request.GET.get('location', '')

    Jobs = Job.objects.all().order_by('-posted_on')

    if keyword:
        Jobs = Jobs.filter(Q(JobRole__icontains=keyword) | Q(description__icontains=keyword))

    if location:
        Jobs = Jobs.filter(location__icontains=location)

    return render(request, 'job_app/joblist.html', {
        'Jobs': Jobs,
        'keyword': keyword,
        'location': location
    })


def job_category(request, category):
    # Convert URL-friendly slug to readable category
    category_name = category.replace('-', ' ').title()

    # Filter jobs from DB that match this category (case-insensitive)
    Jobs = Job.objects.filter(category__iexact=category_name).order_by('-posted_on')

    return render(request, 'job_app/joblist.html', {"category": category_name, "Jobs": Jobs})


def register(request):
    return render(request, "job_app/register.html")

#user registration

def employer_register(request):
    if request.method == 'POST':
        form = EmployerRegistrationForm(request.POST, request.FILES)
        if form.is_valid():
            user = form.save()
            profile_picture = form.cleaned_data.get('profile_picture')
            Profile.objects.create(user=user, profile_picture=profile_picture,user_type='employer')
            login(request, user)  # Auto-login after registration
            return redirect('home')  # Redirect to homepage
    else:
        form = EmployerRegistrationForm()
    return render(request, 'job_app/employer_register.html', {'form': form})

# emlpoyee registration
def employee_register(request):
    if request.method == 'POST':
        form = EmployeeRegistrationForm(request.POST, request.FILES)
        if form.is_valid():
            user = form.save()
            profile_picture = form.cleaned_data.get('profile_picture')
            # skills = form.cleaned_data.get('skills')
            # experience = form.cleaned_data.get('experience')
            
            Profile.objects.create(
                user=user,
                profile_picture=profile_picture,
                user_type='employee',
            )
            login(request, user)
            return redirect('home')
    else:
        form = EmployeeRegistrationForm()
    return render(request, 'job_app/employee_register.html', {'form': form})


#login
def login_view(request):
    if request.method == "POST":
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get("username")
            password = form.cleaned_data.get("password")
            user = authenticate(request, username=username, password=password)
            if user:
                login(request, user)
                return redirect("home")  # Redirect to home after login
    else:
        form = LoginForm()
    return render(request, "job_app/login.html", {"form": form})

#logout
def logout_view(request):
    logout(request)
    return redirect("home")

#post_job
@login_required
def post_job(request):
    if request.user.user_type != 'employer':
        return redirect('home')
    
    if request.method =='POST':
        form = JobForm(request.POST)
        if form.is_valid():
            Job = form.save(commit=False)
            Job.posted_by = request.user
            Job.save()
            return redirect('job_list')
    else:
        form = JobForm()

    return render(request, 'job_app/post_job.html', {'form': form})
         

def profile(request):
    if request.method == 'POST' and request.FILES.get('profile_picture'):
        profile_form = ProfileForm(request.POST, request.FILES, instance=request.user.profile)
        if profile_form.is_valid():
            profile_form.save()
            return redirect('profile')
    else:
        profile_form = ProfileForm(instance=request.user.profile)
    
    return render(request, 'profile.html', {'profile_form': profile_form})




@login_required
def dashboard(request):
    user = request.user
    profile = get_object_or_404(Profile, user=user)

    # Handle profile picture upload
    if request.method == 'POST' and 'profile_picture' in request.FILES:
        profile.profile_picture = request.FILES['profile_picture']
        profile.save()
        messages.success(request, 'Profile picture updated successfully!')
        return redirect('dashboard')

    # Profile completeness
    filled_fields = sum(bool(getattr(profile, field)) for field in ['profile_picture'])
    total_fields = 1
    completeness_percentage = int((filled_fields / total_fields) * 100)

    # Common context
    context = {
        'user': user,  # Ensure 'user' is passed for template conditionals
        'profile': profile,
        'completeness_percentage': completeness_percentage,
    }

    if request.user.user_type == 'employee':
        applied_jobs = JobApplication.objects.filter(applicant=user).select_related('job')
        context['applied_jobs'] = applied_jobs
        context['job_count'] = applied_jobs.count()
    
    elif request.user.user_type == 'employer':
        posted_jobs = Job.objects.filter(posted_by=user)
        context['posted_jobs'] = posted_jobs
        context['job_count'] = posted_jobs.count()

    return render(request, 'job_app/dashboard.html', context)



@login_required
def edit_profile(request):
    user = request.user

    # handle profile-picture upload
    if request.method == 'POST' and 'profile_picture' in request.FILES:
        profile = user.profile
        profile.profile_picture = request.FILES['profile_picture']
        profile.save()
        messages.success(request, "Profile picture updated.")
        return redirect('edit_profile')

    # handle name/email update
    if request.method == 'POST' and 'user_name' in request.POST:
        new_username = request.POST.get('user_name')
        new_email    = request.POST.get('email')
        if new_username and new_email:
            user.username = new_username
            user.email    = new_email
            user.save()
            messages.success(request, "Your name and email have been updated.")
        else:
            messages.error(request, "Both fields are required.")
        return redirect('edit_profile')

    return render(request, 'job_app/edit_profile.html')


@login_required
def job_detail(request, job_id):
    job = get_object_or_404(Job, id=job_id)

    has_applied = False
    if request.user.is_authenticated and hasattr(request.user, 'profile'):
        has_applied = JobApplication.objects.filter(job=job, applicant=request.user).exists()

    context = {
        'job': job,
        'has_applied': has_applied,
    }
    return render(request, 'job_app/job_detail.html', context)



@login_required
def apply_job(request, job_id):
    job = get_object_or_404(Job, id=job_id)

    existing_application = JobApplication.objects.filter(job=job, applicant=request.user).exists()
    if existing_application:
        return render(request, 'job_app/apply_job.html', {
            'job': job,
            'already_applied': True  
        })

    if request.method == 'POST':
        resume = request.FILES.get('resume')
        description = request.POST.get('description', '')
        skills = request.POST.get('skills', '')
        experience = request.POST.get('experience', '')
        industry = request.POST.get('industry', '')
        if resume and description and skills and experience:
            JobApplication.objects.create(
                job=job,
                applicant=request.user,
                resume=resume,
                description=description,
                skills=skills,
                experience=experience,
                industry=industry
            )
            messages.success(request, 'Application submitted successfully!')
            return redirect('job_detail', job_id=job.id)
        else:
            messages.error(request, 'All fields are required.')

    return render(request, 'job_app/apply_job.html', {
        'job': job,
        'already_applied': False
    })


def view_applicants(request, job_id):
    # Fetch the job that the employer posted
    job = get_object_or_404(Job, id=job_id)
    
    # Check if the logged-in user is the employer who posted the job
    if job.posted_by != request.user:
        return HttpResponseForbidden("You are not authorized to view applicants for this job.")
    
    # Fetch all applicants for the job
    applicants = JobApplication.objects.filter(job=job).select_related('applicant')

    context = {
        'job': job,
        'applicants': applicants
    }
    return render(request, 'job_app/view_applicants.html', context)



@login_required
def delete_job(request, job_id):
    job = get_object_or_404(Job, id=job_id)

    # Ensure only the job poster can delete the job
    if request.user == job.posted_by:
        job.delete()
        messages.success(request, "Job deleted successfully.")
    else:
        messages.error(request, "You are not authorized to delete this job.")

    return redirect('dashboard')


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

@login_required
def predict_job_title_view(request):
    prediction_result = None
    predicted_title = None

    # Get the latest 10 employee profiles
    recent_profiles = Profile.objects.filter(user__user_type='employee').order_by('-id')[:10]

    if request.method == 'POST':
        # Collect input from form
        if 'job_profile' in request.POST:
            job_profile = request.POST.get('job_profile')
        elif all(field in request.POST for field in ['skills', 'experience', 'description','industry']):
            skills = request.POST.get('skills')
            experience = request.POST.get('experience')
            description = request.POST.get('description')
            industry = request.POST.get('industry')
            job_profile = f"{experience} {skills} {description} {industry}"
        else:
            job_profile = None

        if job_profile:
            # Get detailed prediction results
            from job_adver.ml.ml_utils import predict_job_category_and_titles
            prediction_result = predict_job_category_and_titles(job_profile)
            
            # For backward compatibility, also get a single title
            predicted_title = prediction_result['suggested_titles'][0] if prediction_result and 'suggested_titles' in prediction_result and prediction_result['suggested_titles'] else None

    return render(request, 'job_app/predictjob.html', {
        'recent_profiles': recent_profiles,
        'prediction_result': prediction_result,
        'predicted_title': predicted_title,  # For backward compatibility
    }) 


#view applicant details on predict title page
from django.contrib.auth import get_user_model
User = get_user_model()

@login_required
def view_applicant_details(request, user_id):
    user = get_object_or_404(User, id=user_id)
    profile = get_object_or_404(Profile, user=user)
    applications = JobApplication.objects.filter(applicant=user)

    return render(request, 'job_app/applicant_details.html', {
        'profile': profile,
        'applications': applications
    })

@login_required
def model_accuracy_view(request):
    all_predictions = None

    # Define raw accuracy scores
    raw_models = [
        {'name': 'Logistic Regression', 'accuracy': 0.88},
        {'name': 'Random Forest', 'accuracy': 0.85},
        {'name': 'MLP Classifier', 'accuracy': 0.91},
        {'name': 'SVM', 'accuracy': 0.83},
        {'name': 'KNN', 'accuracy': 0.82}
    ]

    # Convert to percentage
    all_models = [{'name': m['name'], 'accuracy': m['accuracy'] * 100} for m in raw_models]

    # Sort for top 3
    top_models = sorted(all_models, key=lambda x: x['accuracy'], reverse=True)[:3]

    if request.method == 'POST':
        job_profile = request.POST.get('job_profile')
        if job_profile:
            all_predictions = predict_job_category_and_titles(job_profile)

    return render(request, 'job_app/model_accuracy.html', {
        'all_predictions': all_predictions,
        'all_models': all_models,
        'top_models': top_models
    })



def browse_dataset_view(request):
    """View to display the job dataset in a table with download option"""
    try:
        # Handle dataset download request
        if request.GET.get('download'):
            file_path = request.GET.get('file_path')
            if os.path.exists(file_path) and file_path.endswith('.csv'):
                with open(file_path, 'rb') as csv_file:
                    response = HttpResponse(csv_file.read(), content_type='text/csv')
                    response['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
                    return response
            else:
                return HttpResponse("File not found", status=404)
            
        # Search for job datasets in common locations
        for root, _, files in os.walk(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
            for file in [f for f in files if f.endswith('.csv')]:
                file_path = os.path.join(root, file)
                try:
                    # Check if file has job-related columns
                    df = pd.read_csv(file_path)
                    if any(col in df.columns for col in ['company', 'jobtitle', 'experience']):
                        # Found a job dataset - limit to 100 rows and display
                        return render(request, 'job_app/browse_dataset.html', {
                            'dataset_html': df.head(100).to_html(classes='table table-striped', index=False),
                            'file_path': file_path
                        })
                except:
                    continue
        
        # If we get here, no suitable dataset was found
        return render(request, 'job_app/browse_dataset.html', {
            'error_message': "No job dataset found. Please upload a CSV with job-related columns."
        })
    except Exception as e:
        return render(request, 'job_app/browse_dataset.html', {
            'error_message': f"Error: {str(e)}"
        })


from job_adver.ml.ml_utils import MODEL_ACCURACIES, MODEL_NAME_MAP
import numpy as np
import matplotlib
matplotlib.use('Agg')

def matplotlib_chart_view(request, chart_type='bar'):
    # Get data from model accuracies
    labels = [MODEL_NAME_MAP[k] for k in MODEL_ACCURACIES.keys()]
    values = [round(v * 100, 2) for v in MODEL_ACCURACIES.values()]
    
    # Check for invalid values (NaN or Inf) that could cause issues
    valid_indices = np.isfinite(values)
    filtered_labels = [labels[i] for i, is_valid in enumerate(valid_indices) if is_valid]
    filtered_values = [values[i] for i, is_valid in enumerate(valid_indices) if is_valid]
    
    # Create figure
    plt.figure(figsize=(10, 5), dpi=100)
    
    # Create chart based on type
    if chart_type == 'pie' and filtered_values:
        # Only create pie chart if we have valid data
        if len(filtered_values) > 0 and sum(filtered_values) > 0:
            plt.pie(filtered_values, labels=filtered_labels, autopct='%1.1f%%', 
                   startangle=90, shadow=False, explode=[0.05] * len(filtered_values))
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        else:
            # Create fallback text if no valid data for pie chart
            plt.text(0.5, 0.5, 'No valid data for pie chart', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes, fontsize=14, color='red')
            plt.axis('off')
            
    elif chart_type == 'line':
        plt.plot(filtered_labels, filtered_values, marker='o', linestyle='-', 
                linewidth=2, color='teal', markersize=8)
        plt.ylim(0, max(filtered_values) * 1.15 if filtered_values else 100)  # Dynamic y limit
        
        for i, v in enumerate(filtered_values):
            plt.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
                    
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
    else:  # default to bar chart
        bars = plt.bar(filtered_labels, filtered_values, color='cornflowerblue', 
                      width=0.6, edgecolor='darkblue', linewidth=1)
        plt.ylim(0, max(filtered_values) * 1.15 if filtered_values else 100)  # Dynamic y limit
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', 
                    ha='center', fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
                    
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add title and finishing touches
    plt.title(f'Model Accuracy Comparison ({chart_type.capitalize()} Chart)', 
             fontsize=14, fontweight='bold', pad=15)
    plt.xticks(rotation=15, fontsize=10)
    plt.tight_layout()
    
    # Save figure to buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()  # Explicitly close figure to free memory
    
    # Convert to base64 for embedding in HTML
    chart_url = 'data:image/png;base64,' + base64.b64encode(image_png).decode('utf-8')
    
    return render(request, 'job_app/bar_chart_accuracy.html', {
        'chart_url': chart_url,
        'chart_type': chart_type,
        'models': MODEL_NAME_MAP,
        'accuracies': {MODEL_NAME_MAP[k]: f"{v*100:.2f}%" for k, v in MODEL_ACCURACIES.items()}
    })

def model_accuracy_results(request):
    # Format accuracy data for display
    accuracy_data = {
        MODEL_NAME_MAP.get(model, model): f"{accuracy*100:.2f}%" 
        for model, accuracy in MODEL_ACCURACIES.items()
    }
    
    # Sort by accuracy (descending)
    sorted_data = dict(sorted(accuracy_data.items(), key=lambda x: float(x[1].strip('%')), reverse=True))
    
    return render(request, 'job_app/model_accuracy_results.html', {
        'accuracy_data': sorted_data,
        'highest_model': next(iter(sorted_data)) if sorted_data else None,
        'highest_accuracy': next(iter(sorted_data.values())) if sorted_data else None
    })


from job_adver.ml.ml_utils import get_context_aware_titles  # Import the function

def job_title_types(request):
    """
    Display the predicted job title types and examples from each category
    """
    # Define your categories manually or create a proper mapping
    categories = [
        'Software Development',
        'Healthcare', 
        'Finance',
        'Data & Analytics',
        'Sales & Marketing',
        'Business Management',
        'HR & Recruitment',
        'IT Operations',
        'Education',
        'Legal',
        'Creative & Design'
    ]
    
    # Generate job titles for each category
    job_categories = {}
    for category in categories:
        # Use a generic text to get representative titles for each category
        sample_titles = get_context_aware_titles(category, category.lower(), None)
        job_categories[category] = sample_titles
    
    # Sort categories alphabetically for consistent display
    sorted_categories = dict(sorted(job_categories.items()))
    
    return render(request, 'job_app/job_title_types.html', {
        'job_categories': sorted_categories
    })


def job_title_ratio_view(request):
    """View to display the ratio of different job title types"""
    try:
        # Import necessary functions from your ML module
        from job_adver.ml.ml_utils import MODEL_ACCURACIES, MODEL_NAME_MAP, get_context_aware_titles
        
        # Define your categories
        categories = [
            'Software Development',
            'Healthcare', 
            'Finance',
            'Data & Analytics',
            'Sales & Marketing',
            'Business Management',
            'HR & Recruitment',
            'IT Operations',
            'Education',
            'Legal',
            'Creative & Design'
        ]
        
        # Calculate ratios/counts of different job titles by category
        category_counts = {}
        total_titles = 0
        
        # Count titles in each category
        for category in categories:
            # Get sample titles for this category
            sample_titles = get_context_aware_titles(category, category.lower(), None)
            count = len(sample_titles)
            category_counts[category] = count
            total_titles += count
        
        # Calculate percentages
        category_percentages = {}
        for category, count in category_counts.items():
            percentage = (count / total_titles) * 100 if total_titles > 0 else 0
            category_percentages[category] = {
                'count': count,
                'percentage': round(percentage, 2)
            }
        
        # Sort by count (descending)
        sorted_categories = dict(sorted(
            category_percentages.items(), 
            key=lambda x: x[1]['count'], 
            reverse=True
        ))
        
        return render(request, 'job_app/job_title_ratio.html', {
            'categories': sorted_categories,
            'total_titles': total_titles
        })
        
    except Exception as e:
        return render(request, 'job_app/job_title_ratio.html', {
            'error_message': f"Error calculating job title ratios: {str(e)}"
        })