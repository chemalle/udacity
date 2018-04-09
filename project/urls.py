from django.conf.urls import url
from . import views


urlpatterns = [
    url(r'^$', views.home, name='home'),
    url(r'^stock/$', views.stock, name='udacity'),
    url(r'^codigo/$', views.codigo, name='code'),
    url(r'^stock2/$', views.stock2, name='udacity2'),
    url(r'^proposta/$', views.proposta, name='proposta'),
    url(r'^code/$', views.code, name='code2'),

]
