from django.urls import path
from . import views
from .views import employer_register, employee_register
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('', views.home, name='home'),
    path('jobs/', views.job_list, name='job_list'),
    path('category/<str:category>/', views.job_category, name='job_category'),
    path('register/employer/', employer_register, name='employer_register'),
    path('register/employee/', employee_register, name='employee_register'),
    path('post-job/', views.post_job, name='post_job'),
    path('job/<int:job_id>/', views.job_detail, name='job_detail'),
    path('jobs/<int:job_id>/apply/', views.apply_job, name='apply_job'),
    path('predict-job-title/', views.predict_job_title_view, name='predict_job_title'),

]  + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
