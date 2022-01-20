from django.db import models

# Create your models here.
class Summarizer(models.Model):
    Actual_Summarizer = models.TextField()
    Extractive_Summarizer = models.TextField()
    Abstract_Summarizer = models.TextField()