from django.conf.urls import url
from text_app import views

from django.conf.urls.static import static
from django.conf import settings

urlpatterns=[
    url(r'^textapi$',views.textapi)
] + + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)