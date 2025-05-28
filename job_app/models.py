from django.db import models
from django.contrib.auth.models import AbstractUser
from django.conf import settings
from django.core.exceptions import ValidationError

def validate_not_empty(value):
    if not value.strip():
        raise ValidationError("This field cannot be empty or just whitespace.")

class Job(models.Model):
    JobRole = models.CharField(max_length=200)
    company = models.CharField(max_length=200)
    location = models.CharField(max_length=200)
    category = models.CharField(max_length=100)
    description = models.TextField()
    posted_on = models.DateTimeField(auto_now_add=True)
    posted_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)

    def __str__(self):
        return self.JobRole

class CustomUser(AbstractUser):
    USER_TYPE_CHOICES = (
        ('employer', 'Employer'),
        ('employee', 'Employee'),
    )
    user_type = models.CharField(max_length=10, choices=USER_TYPE_CHOICES, default='employee')

    def __str__(self):
        return self.username

class Profile(models.Model):
    USER_TYPE_CHOICES = (
        ('employee', 'Employee'),
        ('employer', 'Employer'),
    )

    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    user_type = models.CharField(max_length=20, choices=USER_TYPE_CHOICES)
    profile_picture = models.ImageField(upload_to='profile_pics/', null=True, blank=True)

    skills = models.TextField(blank=True, validators=[validate_not_empty])
    experience = models.TextField(blank=True, validators=[validate_not_empty])
    description = models.TextField(blank=True, validators=[validate_not_empty])
    industry = models.CharField(max_length=100, blank=True, validators=[validate_not_empty])


    def __str__(self):
        return self.user.username

class JobApplication(models.Model):
    job = models.ForeignKey('Job', on_delete=models.CASCADE, related_name='applications')
    applicant = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    resume = models.FileField(upload_to='resumes/')
    description = models.TextField(blank=True, validators=[validate_not_empty])
    skills = models.TextField(blank=True, validators=[validate_not_empty])
    experience = models.CharField(max_length=100, blank=True, validators=[validate_not_empty])
    job_profile = models.TextField(blank=True)
    industry = models.CharField(max_length=100, blank=True, validators=[validate_not_empty])
    
    def save(self, *args, **kwargs):
        self.job_profile = f"{self.experience} {self.description} {self.skills} {self.industry}".strip()
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.applicant.username} - {self.job.JobRole}"
