# Generated by Django 5.2 on 2025-05-02 14:49

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('job_app', '0002_profile_experience_profile_job_profile_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='profile',
            name='experience',
        ),
        migrations.RemoveField(
            model_name='profile',
            name='job_profile',
        ),
        migrations.RemoveField(
            model_name='profile',
            name='skills',
        ),
        migrations.AddField(
            model_name='jobapplication',
            name='experience',
            field=models.CharField(blank=True, max_length=100),
        ),
        migrations.AddField(
            model_name='jobapplication',
            name='job_profile',
            field=models.TextField(blank=True),
        ),
        migrations.AddField(
            model_name='jobapplication',
            name='skills',
            field=models.TextField(blank=True),
        ),
    ]
