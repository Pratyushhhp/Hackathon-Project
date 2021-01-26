from django.db import models

# Create your models here.

class Personality(models.Model):
    questions = models.TextField(blank = True, null = True)
    answers = models.CharField(max_length=100, blank= True, null = True)

class School(models.Model):
    questions = models.TextField(blank = True, null = True)
    answers = models.CharField(max_length=100, blank = True, null = True)

class Graduates(models.Model):
    questions = models.TextField(blank = True, null = True)
    answers = models.CharField(max_length = 100, blank = True, null = True)

class Dropouts(models.Model):
    questions = models.TextField(blank = True, null = True)
    answers = models.CharField(max_length=100, blank = True, null = True)

