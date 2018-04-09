from django.forms import ModelForm
from .models import Input, Input2


class InputForm(ModelForm):
    class Meta:
        model = Input
        fields = '__all__'


class InputForm2(ModelForm):
    class Meta:
        model = Input2
        fields = '__all__'
