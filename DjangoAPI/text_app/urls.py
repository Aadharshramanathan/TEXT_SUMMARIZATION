from django.urls import re_path
from text_app import views

urlpatterns=[
    re_path(r'^textapi$',views.textapi)
]