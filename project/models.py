from django.db import models
from django import forms


class Input(models.Model):
    ticker = models.CharField(max_length=10, help_text='Insira o ticker do ativo')




class Input2(models.Model):
    tickerA = models.CharField(max_length=10, help_text='Insira o ticker do ativo')
    tickerB = models.CharField(max_length=10, help_text='Insira o ticker do ativo')
    tickerC = models.CharField(max_length=10, help_text='Insira o ticker do ativo')
    tickerD = models.CharField(max_length=10, help_text='Insira o ticker do ativo')
    tickerE = models.CharField(max_length=10, help_text='Insira o ticker do ativo')
