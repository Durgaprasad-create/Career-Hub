from django.contrib import admin
from .models import Job, Profile, JobApplication

@admin.register(Job)
class JobAdmin(admin.ModelAdmin):
    list_display = ('JobRole', 'company', 'location', 'category', 'posted_by', 'posted_on')
    search_fields = ('JobRole', 'company', 'location', 'category')
    list_filter = ('category', 'posted_on')

@admin.register(Profile)
class ProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'user_type')
    search_fields = ('user__username',)

@admin.register(JobApplication)
class JobApplicationAdmin(admin.ModelAdmin):
    list_display = ('job', 'applicant')
    search_fields = ('job__JobRole', 'applicant__username')
