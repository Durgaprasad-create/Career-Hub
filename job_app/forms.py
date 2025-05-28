from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import CustomUser,Job,Profile
from django.contrib.auth.forms import AuthenticationForm
from .models import JobApplication

class EmployerRegistrationForm(UserCreationForm):
    profile_picture = forms.ImageField(required=False)
    class Meta:
        model = CustomUser
        fields = ['username', 'email', 'password1', 'password2','profile_picture']

    def __init__(self, *args, **kwargs):
        super(EmployerRegistrationForm, self).__init__(*args, **kwargs)
        # Add 'form-control' class to every form field
        for field in self.fields.values():
            field.widget.attrs.update({'class': 'form-control'})

        # If you want to style the file input differently
        self.fields['profile_picture'].widget.attrs.update({'class': 'form-control-file'})
    
    def save(self, commit=True):
        user = super().save(commit=False)
        user.user_type = 'employer'
        if commit:
            user.save()
        return user
        

class EmployeeRegistrationForm(UserCreationForm):
    profile_picture = forms.ImageField(required=False)
    # skills = forms.CharField(widget=forms.Textarea(attrs={'class': 'form-control'}), required=True)
    # experience = forms.CharField(widget=forms.Textarea(attrs={'class': 'form-control'}), required=True)
    class Meta:
        model = CustomUser
        fields = ['username', 'email', 'password1', 'password2','profile_picture']

    def __init__(self, *args, **kwargs):
        super(EmployeeRegistrationForm, self).__init__(*args, **kwargs)
        # Add 'form-control' class to every form field
        for field in self.fields.values():
            field.widget.attrs.update({'class': 'form-control'})

        # If you want to style the file input differently
        self.fields['profile_picture'].widget.attrs.update({'class': 'form-control-file'})
    
    def save(self, commit=True):
        user = super().save(commit=False)
        user.user_type = 'employee'
        if commit:
            user.save()
        return user

class LoginForm(AuthenticationForm):
    username = forms.CharField(widget=forms.TextInput(attrs={'class': 'form-control'}))
    password = forms.CharField(widget=forms.PasswordInput(attrs={'class': 'form-control'}))


# post_job

class JobForm(forms.ModelForm):
    class Meta:
        model = Job
        fields = ['JobRole', 'company', 'location', 'category', 'description']

        widgets = {
            'JobRole': forms.TextInput(attrs={'class': 'form-control'}),
            'company': forms.TextInput(attrs={'class': 'form-control'}),  
            'location': forms.TextInput(attrs={'class': 'form-control'}),
            'category': forms.TextInput(attrs={'class': 'form-control'}), 
            'description': forms.Textarea(attrs={'class': 'form-control'}),
        }


class ProfileForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ['profile_picture']


# forms.py
class JobApplicationForm(forms.ModelForm):
    skills = forms.CharField(
        max_length=500,
        required=True,
        widget=forms.Textarea(attrs={'placeholder': 'List your skills', 'rows': 3})
    )
    experience = forms.CharField(
        max_length=500,
        required=True,
        widget=forms.Textarea(attrs={'placeholder': 'Describe your experience', 'rows': 3})
    )
    description = forms.CharField(
        max_length=500,
        required=True,
        widget=forms.Textarea(attrs={'placeholder': 'Give a Descrption of your skills', 'rows': 3})
    )
    industry = forms.CharField(
        max_length=500,
        required=True,
        widget=forms.Textarea(attrs={'placeholder': 'Specify your industry or Feild', 'rows': 3})
    )
    class Meta:
        model = JobApplication
        fields = ['resume', 'description', 'skills', 'experience','industry']
