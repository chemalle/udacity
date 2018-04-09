from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.template.loader import render_to_string
from django.contrib.auth.models import User
from django.core.mail import EmailMessage
from django import forms
from django.shortcuts import render_to_response
import pandas as pd
import numpy as np
from sklearn import covariance, cluster
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
from sklearn import ensemble
from .forms import InputForm, InputForm2
from pandas_datareader import data as web



def home(request):
    return render(request, 'project/home.html')


def stock(request):
    if request.method == 'POST':
        form = InputForm(request.POST or None, request.FILES or None)
        if form.is_valid():
            form = form.save(commit=False)
            return UDACITY(form)

    else:
        form = InputForm()


    return render(request, 'stocks.html', context = {'form': form})




def UDACITY(form):
    try:
        ticker = form.ticker+'.sa'
        stocks = ['QQQ','EWZ','VXX','GLD','^BVSP',ticker]


        optimization = web.DataReader(stocks, data_source='yahoo')
        optimization.fillna(method='bfill', inplace=True)

        optimization = optimization.to_frame(filter_observations=True)['Close'].reset_index()
        optimization.rename(columns={'minor': 'Ticker',
                          'Adj Close': 'Close'}, inplace=True)
        optimization = optimization.pivot(index='Date', columns='Ticker')
        optimization.columns = optimization.columns.droplevel(0)
        # pivot each ticker to a column

        X, y = [], []
        for index,row in optimization.iterrows():
            X.append(row[0:-1])
            y.append(row[-1])

        X, y = shuffle(X, y, random_state=23)

        num_training = int(0.9 * len(X))
        X_train, y_train = X[:num_training], y[:num_training]
        X_test, y_test = X[num_training:], y[num_training:]
        rf_regressor = RandomForestRegressor(n_estimators=1000, max_depth=10,random_state=23)

        rf_regressor.fit(X_train, y_train)
        y_pred = rf_regressor.predict(X_test)

        adaboost_pred = rf_regressor.predict(optimization.iloc[:,0:-1])
        adaboost_pred = adaboost_pred.reshape(-1,1)

        optimization['Prediction'] = adaboost_pred

        mse = mean_squared_error(optimization['Prediction'], optimization.iloc[:,-2])
        evs = explained_variance_score(optimization['Prediction'], optimization.iloc[:,-2])

        optimization['MSE']= mse
        optimization['EVS'] = evs

        SPX = optimization

        SPX['SMA3'] = pd.Series.rolling(SPX['Prediction'], 3).mean()
        SPX['SMA8'] = pd.Series.rolling(SPX['Prediction'], 8).mean()
        SPX['SMA21'] = pd.Series.rolling(SPX['Prediction'], 21).mean()
        SPX['SMA50'] = pd.Series.rolling(SPX['Prediction'], 50).mean()
        SPX['SMA200'] = pd.Series.rolling(SPX['Prediction'], 200).mean()

        SPX.dropna(inplace=True)

    #     return SPX

        DayTrading = SPX['SMA3'] - SPX['SMA21']
        SPX['DayTrading'] = DayTrading
        SPX['SwingTrade'] = np.where(SPX['DayTrading']>0, 1,-1)
        SPX['GOLDEN_SMA3'] = SPX['SwingTrade'].shift(+1)
        SPX['position3'] = np.where(SPX['SMA3'] > SPX['SMA21'], 1, -1)
        SPX['position8'] = np.where(SPX['SMA8'] > SPX['SMA21'], 1, -1)
        Recomenda = SPX['SMA50'] - SPX['SMA200']
        SPX['Recomenda'] = Recomenda
        SPX['GOLDEN'] = np.where(SPX['Recomenda']>0, 1,-1)
        SPX['GOLDEN_CROSS'] = SPX['GOLDEN'].shift(+1)
        SPX['position'] = np.where(SPX['SMA3'] > SPX['SMA21'], 1, -1)
        SPX['market'] = np.log(SPX[ticker] / SPX[ticker].shift(1))
        # vectorized calculation of strategy returns
        SPX['strategy'] = SPX['position'].shift(1) * SPX['market']
        Golden_Rule = SPX['strategy'] - SPX['market']
        SPX['GOLDEN_RULE'] = Golden_Rule

        if SPX['GOLDEN_SMA3'][-1] == -1 and SPX['SwingTrade'][-1] == 1:
            average = [[ticker,SPX['SMA21'][-1], 'SMA3 just REACHED!', SPX['GOLDEN_RULE'][-1],optimization['MSE'][-1],optimization['EVS'][-1], 'Random Forest']]
        elif SPX['position3'][-1] == 1:
            average = [[ticker, SPX['SMA21'][-1], 'SMA3',SPX['GOLDEN_RULE'][-1],optimization['MSE'][-1],optimization['EVS'][-1], 'Random Forest']]
        elif SPX['position8'][-1] == 1:
            average = [[ticker,SPX['SMA21'][-1], 'SMA8',SPX['GOLDEN_RULE'][-1],optimization['MSE'][-1],optimization['EVS'][-1], 'Random Forest']]
        elif SPX['GOLDEN_CROSS'][-1] == -1 and SPX['GOLDEN'][-1] == 1:
            average = [[ticker,SPX['SMA200'][-1], 'Golden_Cross REACHED!',SPX['GOLDEN_RULE'][-1],optimization['MSE'][-1],optimization['EVS'][-1], 'Random Forest']]
        elif SPX['GOLDEN'][-1] == 1:
            average = [[ticker,SPX['SMA200'][-1], 'Long Term',SPX['GOLDEN_RULE'][-1],optimization['MSE'][-1],optimization['EVS'][-1], 'Random Forest']]
        else:
            average = [[ticker,SPX['SMA200'][-1], 'downward trend',SPX['GOLDEN_RULE'][-1],optimization['MSE'][-1],optimization['EVS'][-1], 'Random Forest']]




        barchart = pd.DataFrame(list(average))
        barchart = barchart
        barchart.columns = ['Ticker', 'SMA','ACTION','Strategy Vs Market','MSE','EVS','Method']
        barchart2 = barchart.to_html(index=False,columns=['Ticker', 'SMA','ACTION','Strategy Vs Market','MSE','EVS','Method'])
        EVS =  "{0:.2f}%".format(float(barchart['EVS'][0]*100))
        MSE =  '{:,.2f}'.format(float(barchart['MSE'][0]))

    except Exception:
        return render_to_response('project/apologies.html')
    return render_to_response('name.html', context= {'barchart':barchart2,'EVS':EVS, 'MSE': MSE})





def codigo(request):
    return render(request, 'project/codigo.html')





#data_barchart = pd.read_excel('stocks_prediction_11052017.xlsx','VAR')


def stock2(request):
    if request.method == 'POST':
        form = InputForm2(request.POST or None, request.FILES or None)
        if form.is_valid():
            form = form.save(commit=False)
            return kluster(form)

    else:
        form = InputForm2()


    return render(request, 'carteira.html', context = {'form': form})






from sklearn import covariance
from pandas_datareader import data as web
import quandl
import numpy as np
quandl.ApiConfig.api_key = "oAe9Zos9MifP13eC9yRM"
def kluster(form):

    try:

        tickerA = web.DataReader(form.tickerA+'.sa',data_source='yahoo')[-252:]
        tickerB = web.DataReader(form.tickerB+'.sa',data_source='yahoo')[-252:]
        tickerC = web.DataReader(form.tickerC+'.sa',data_source='yahoo')[-252:]
        tickerD = web.DataReader(form.tickerD+'.sa',data_source='yahoo')[-252:]
        tickerE = web.DataReader(form.tickerE+'.sa',data_source='yahoo')[-252:]

        barchart = [tickerA,tickerB,tickerC,tickerD,tickerE]

        names = [form.tickerA,form.tickerB,form.tickerC,form.tickerD,form.tickerE]

        quotes = []


        for item in barchart:
            portfolio = pd.DataFrame(item)
            quotes.append(portfolio)

        names = pd.DataFrame(names).T
        opening_quotes = np.array([quote.Open for quote in quotes]).astype(np.float)
        closing_quotes = np.array([quote.Close for quote in quotes]).astype(np.float)


        delta_quotes = closing_quotes - opening_quotes


        edge_model = covariance.GraphLassoCV()

        X = delta_quotes.copy().T
        X /= X.std(axis=0)

        with np.errstate(invalid='ignore'):
            edge_model.fit(X)

        from sklearn import cluster

        _, labels = cluster.affinity_propagation(edge_model.covariance_)
        num_labels = labels.max()

        k = []

        for i in range(num_labels + 1):
            try:
                cluster= ( i+1, ', '.join(names.T[0][labels == i]))
                k.append(cluster)
            except Exception:
                    pass  # or you could use 'continue'

        kluster = pd.DataFrame(list(k))
        kluster.columns = ['Cluster','Ticker']
        kluster = kluster.to_html(index=False,columns=['Cluster', 'Ticker'])

    except Exception:
        return render_to_response('project/apologies.html')

    return render_to_response('cluster.html', context= {'kluster':kluster})




def proposta(request):
    return render_to_response('project/Proposta_EduardoChemalle_ProjetoFinal_V4.html')


def code(request):
    return render_to_response('project/code.html')
