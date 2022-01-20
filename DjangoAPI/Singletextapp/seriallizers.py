from rest_framework import serializers
from Singletextapp.models import Summarizer

class SummarizerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Summarizer
        fields = ('Actual_Summarizer',
                   'Extractive_Summarizer',
                   'Abstract_Summarizer')