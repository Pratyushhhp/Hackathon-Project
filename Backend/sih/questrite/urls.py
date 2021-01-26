from django.urls import path
from django.conf.urls import url

from . import views

urlpatterns = [
    path('index', views.index, name = 'index'),
    path('', views.index, name = 'index'),
    path('signup', views.signup, name = 'signup'),
    path('about', views.about, name = 'about'),
    path('analyst', views.analyst, name = 'analyst'),
    path('contactus', views.contactus, name = 'contactus'),
    path('faqs', views.faqs, name = 'faqs'),
    path('mcq1', views.mcq1, name = 'mcq1'),
    path('mcq2', views.mcq2, name = 'mcq2'),
    path('mcq3', views.mcq3, name = 'mcq3'),
    path('mcq4', views.mcq4, name = 'mcq4'),
    path('ourservices', views.ourservices, name = 'ourservices'),
    path('personality', views.personality, name = 'personality'),
    path('result', views.result, name = 'result'),
    path('resultp', views.resultp, name = 'resultp'),
    path('reviews', views.reviews, name = 'reviews'),
    path('test1', views.test1, name = 'test1'),
    path('test2', views.test2, name = 'test2'),
    path('test3', views.test3, name = 'test3'),  
    path('logout', views.logout, name = 'logout'),
    path('Realistic', views.realistic, name = "Realistic"),
    path('Investigative', views.investigative, name = "Investigative"),
    path('Artistic', views.artistic, name = "Artistic"),
    path('Social', views.social, name = "Social"),
    path('Enterprising', views.enterprising, name = "Enterprising"),
    path('Conventional', views.conventional, name = "Conventional"),
    path('counsellor', views.counsellor, name = "counsellor"),
    path('scholarship', views.scholarship, name = "scholarship")

]