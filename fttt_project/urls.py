# fttt_project/urls.py

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.i18n import set_language

urlpatterns = [
    path('admin/', admin.site.urls),
    # Direct all traffic to the 'core' app's URLs
    path('', include('core.urls')), 
    path('api/', include('predictor.urls')),
    path('i18n/', include('django.conf.urls.i18n')),
]
if settings.DEBUG:

    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)


