from django.urls import re_path
from Singletextapp import views

urlpatterns = [
    re_path(r'^single/$',views.summarizerApi)
]