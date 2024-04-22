import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
#import pandas_datareader.data as web
import plotly.graph_objects as go
import statistics as sta
from operator import itemgetter
import math
from datetime import datetime
from datetime import timedelta
from datetime import date
import io
import yfinance as yf
import requests
from requests_html import HTMLSession

st.session_state.update(st.session_state)
for k, v in st.session_state.items():
    st.session_state[k] = v

st.set_page_config(
    page_title='Análise Crypto',
    layout="wide"

                   )

hide_menu = '''
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        '''
st.markdown(hide_menu, unsafe_allow_html=True)

col1, col2 = st.columns([3, 3])


############################
#   Setup
############################

col1.title("Análise de Cryptos")

lista_crypto_select = ['BTC', 'ETH', 'SOL', 'LINK', 'MATIC', 'XRP', 'PYTH', 'MKR', 'RNDR', 'AVAX', 'UNI7083', 'PENDLE', 'AGIX', 'STRK22691', 'DOT', 'STETH', 'ADA', 'FIL', 'AR']

if 'token_nome' not in st.session_state:
    st.session_state.token_nome = 'BTC'

token_nome = col1.selectbox(
   label="Token",
   options=lista_crypto_select,
   key='token_nome'
)

token_nome = token_nome+'-USD'
#st.session_state['token_nome'] = token_nome

if 'intervalo' not in st.session_state:
    st.session_state.intervalo = 'DIÁRIO'

intervalo = col2.radio('Periodicidade', ['DIÁRIO','SEMANAL','MENSAL'], key='intervalo')

#st.session_state['intervalo'] = intervalo

############################
#   Tratamento de Dados
############################

if 'crypto_1d' not in st.session_state and 'low_rsi_1d' not in st.session_state:
    with st.spinner('Carregando dados...'):
        session = HTMLSession()
        num_currencies=200    # Número de tokens a serem listados
        resp = session.get(f"https://finance.yahoo.com/crypto?offset=0&count={num_currencies}")
        tables = pd.read_html(resp.html.raw_html)
        df_cryptonames = tables[0].copy()
        lista_crypto = df_cryptonames.Symbol.tolist()

        lista_crypto_select = ['BTC', 'ETH', 'SOL', 'LINK', 'MATIC', 'XRP', 'PYTH', 'MKR', 'RNDR', 'AVAX', 'UNI7083', 'PENDLE', 'AGIX', 'STRK22691', 'DOT', 'STETH', 'ADA', 'FIL', 'AR']

        try:
          lista_crypto = [token for token in lista_crypto if token[:-4] in lista_crypto_select]
        except:
          pass

        crypto_1d = {}
        crypto_1w = {}
        crypto_1m = {}


        # PERÍODO MÁXIMO EM MESES:
        periodo = 36
        periodo = str(periodo)+'mo'
        #periodo = 'MAX'    # PARA TER O PERÍODO MÁXIMO DE CADA AÇÃO


        # CRIANDO DICIONÁRIO COM TODAS AS CRYPTOS A SEREM ANALISADAS:
        for i in range(len(lista_crypto)):
          b = lista_crypto[i]
          a_1d = yf.Ticker(b).history(period=periodo).iloc[:,:5]
          a_1w = yf.Ticker(b).history(period=periodo, interval = '1wk').iloc[:,:5]
          a_1m = yf.Ticker(b).history(period=periodo, interval = '1mo').iloc[:,:5]

          crypto_1d[lista_crypto[i]] = a_1d
          crypto_1w[lista_crypto[i]] = a_1w
          crypto_1m[lista_crypto[i]] = a_1m


        def calculate_rsi(prices):
            deltas = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
            gains = [delta if delta > 0 else 0 for delta in deltas]
            losses = [-delta if delta < 0 else 0 for delta in deltas]

            avg_gain = sum(gains[-14:]) / 14
            avg_loss = sum(losses[-14:]) / 14

            # Handling the case where avg_loss is zero
            if avg_loss == 0:
                return 100  # RSI is 100 when avg_loss is zero

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            return rsi


        def ema(data, window=200):
            return data.ewm(span=window, adjust=False).mean()


        for token in lista_crypto:

            crypto_1d[token] = crypto_1d[token].reset_index()
            crypto_1w[token] = crypto_1w[token].reset_index()
            crypto_1m[token] = crypto_1m[token].reset_index()
            #crypto_1d[token]['Date'] = pd.to_datetime(crypto_1d[token]['Date'])

            crypto_1d[token]['RSI'] = np.nan
            crypto_1w[token]['RSI'] = np.nan
            crypto_1m[token]['RSI'] = np.nan

            crypto_1d[token]['EMA200'] = ema(crypto_1d[token]['Close'])

            crypto_1d[token]['EMA50'] = ema(crypto_1d[token]['Close'], 50)

            # Pi Cycle Top Indicator (Indicador de topo de mercado)
            crypto_1d[token]['EMA350'] = ema(crypto_1d[token]['Close'], 350)
            crypto_1d[token]['EMA111'] = ema(crypto_1d[token]['Close'], 111)


            rsi_values = []
            for i in range(len(crypto_1d[token]['Close'])):
                if i <= 14:
                    crypto_1d[token]['RSI'][i] = np.nan
                else:
                    rsi = calculate_rsi(crypto_1d[token]['Close'][:i+1])
                    crypto_1d[token]['RSI'][i] = rsi

            rsi_values = []
            for i in range(len(crypto_1w[token]['Close'])):
                if i <= 14:
                    crypto_1w[token]['RSI'][i] = np.nan
                else:
                    rsi = calculate_rsi(crypto_1w[token]['Close'][:i+1])
                    crypto_1w[token]['RSI'][i] = rsi

            rsi_values = []
            for i in range(len(crypto_1m[token]['Close'])):
                if i <= 14:
                    crypto_1m[token]['RSI'][i] = np.nan
                else:
                    rsi = calculate_rsi(crypto_1m[token]['Close'][:i+1])
                    crypto_1m[token]['RSI'][i] = rsi

        bear_list = []
        bull_list = []

        low_rsi_1d = {'Token': [], 'RSI': []}
        low_rsi_1w = {'Token': [], 'RSI': []}
        low_rsi_1m = {'Token': [], 'RSI': []}

        for token in lista_crypto:

          low_rsi_1d['Token'].append(token)
          low_rsi_1d['RSI'].append(crypto_1d[token]['RSI'].iloc[-1])
          if crypto_1d[token]['RSI'].iloc[-1] <= 30:
            print(token,': RSI 1D em sobrevenda')
          elif crypto_1d[token]['RSI'].iloc[-1] < 50:
            print(token,': RSI 1D menor que 50')

          low_rsi_1w['Token'].append(token)
          low_rsi_1w['RSI'].append(crypto_1w[token]['RSI'].iloc[-1])
          if crypto_1w[token]['RSI'].iloc[-1] <= 30:
            print(token,': RSI 1W em sobrevenda')
          elif crypto_1w[token]['RSI'].iloc[-1] < 50:
            print(token,': RSI 1W menor que 50')

          low_rsi_1m['Token'].append(token)
          low_rsi_1m['RSI'].append(crypto_1m[token]['RSI'].iloc[-1])
          if crypto_1m[token]['RSI'].iloc[-1] <= 30:
            print(token,': RSI 1M em sobrevenda')
          elif crypto_1m[token]['RSI'].iloc[-1] < 50:
            print(token,': RSI 1M menor que 50')

          if crypto_1d[token]['EMA200'].iloc[-1] < crypto_1d[token]['EMA50'].iloc[-1]:
            print(token,": EMA200 menor que o EMA50 (bull)")
            bull_list.append(token)
          elif crypto_1d[token]['EMA200'].iloc[-1] > crypto_1d[token]['EMA50'].iloc[-1]:
            print(token,": EMA50 menor que o EMA200 (bear)")
            bear_list.append(token)

          print('\n')

        low_rsi_1d = pd.DataFrame(low_rsi_1d).sort_values(by='RSI').head(20).reset_index(drop=True)
        low_rsi_1w = pd.DataFrame(low_rsi_1w).sort_values(by='RSI').head(20).reset_index(drop=True)
        low_rsi_1m = pd.DataFrame(low_rsi_1m).sort_values(by='RSI').head(20).reset_index(drop=True)

        st.session_state.low_rsi_1d = low_rsi_1d
        st.session_state.low_rsi_1w = low_rsi_1w
        st.session_state.low_rsi_1m = low_rsi_1m
        st.session_state.crypto_1d = crypto_1d
        st.session_state.crypto_1w = crypto_1w
        st.session_state.crypto_1m = crypto_1m


low_rsi_1d = st.session_state['low_rsi_1d']
low_rsi_1w = st.session_state['low_rsi_1w']
low_rsi_1m = st.session_state['low_rsi_1m']
crypto_1d = st.session_state['crypto_1d']
crypto_1w = st.session_state['crypto_1w']
crypto_1m = st.session_state['crypto_1m']


# Diário
if intervalo == 'DIÁRIO':
  # Diário

  # Gráfico RSI token

  fig, ax = plt.subplots(figsize=(16, 6))
  plt.title(token_nome+': RSI '+intervalo)
  plt.grid()

  ax.legend(loc='lower right')
  plt.plot(crypto_1d[token_nome]['Date'], crypto_1d[token_nome]['RSI'] , color='blue', label = 'RSI 1D')
  ax.legend(loc='upper left')

  plt.axhline(y=30, color='green', linestyle='--', linewidth=1.5,label='SOBREVENDA')
  ax.legend(loc='upper left')
  plt.axhline(y=70, color='red', linestyle='--', linewidth=1.5,label='SOBRECOMPRA')
  ax.legend(loc='upper left')
  col2.pyplot(fig)


  candle = go.Figure(data=[go.Candlestick(x=crypto_1d[token_nome]['Date'],
                  open=crypto_1d[token_nome]['Open'],
                  high=crypto_1d[token_nome]['High'],
                  low=crypto_1d[token_nome]['Low'],
                  close=crypto_1d[token_nome]['Close'])])

  candle.update_layout(title=token_nome+' '+intervalo, titlefont=dict(color='gray', size=28), height=500 )
  col1.write(candle)

  # RSI Geral

  col2.table(low_rsi_1d)

  if crypto_1d[token_nome]['RSI'].iloc[-1] <= 30:
    st.write(token_nome,': RSI 1D em sobrevenda')
  elif crypto_1d[token_nome]['RSI'].iloc[-1] < 50:
    st.write(token_nome,': RSI 1D menor que 50')

  if crypto_1d[token_nome]['EMA200'].iloc[-1] < crypto_1d[token_nome]['EMA50'].iloc[-1]:
    st.write(token_nome,": EMA200 menor que o EMA50 (bull)")
  elif crypto_1d[token_nome]['EMA200'].iloc[-1] > crypto_1d[token_nome]['EMA50'].iloc[-1]:
    st.write(token_nome,": EMA50 menor que o EMA200 (bear)")

  # Gráfico Médias

  fig, ax = plt.subplots(figsize=(16, 6))
  plt.title(token_nome+': MÉDIAS '+intervalo)

  plt.plot(crypto_1d[token_nome]['Date'], crypto_1d[token_nome]['Close'] , color='green', label = 'Preço')
  plt.grid()

  ax.legend(loc='lower right')
  plt.plot(crypto_1d[token_nome]['Date'], crypto_1d[token_nome]['EMA111'] , color='red', label = 'Pi Cycle Top Indicator')
  ax.legend(loc='upper left')
  plt.plot(crypto_1d[token_nome]['Date'], crypto_1d[token_nome]['EMA350'] , color='red')
  ax.legend(loc='upper left')
  plt.plot(crypto_1d[token_nome]['Date'], crypto_1d[token_nome]['EMA50'] , color='purple', label = 'EMA50')
  ax.legend(loc='upper left')
  plt.plot(crypto_1d[token_nome]['Date'], crypto_1d[token_nome]['EMA200'] , color='blue', label = 'EMA200')
  ax.legend(loc='upper left')
  col1.pyplot(fig)


# Semanal
elif intervalo == 'SEMANAL':

  # Gráfico RSI

  fig, ax = plt.subplots(figsize=(16, 6))
  plt.title(token_nome+': RSI '+intervalo)
  plt.grid()

  ax.legend(loc='lower right')
  plt.plot(crypto_1w[token_nome]['Date'], crypto_1w[token_nome]['RSI'] , color='blue', label = 'RSI 1D')
  ax.legend(loc='upper left')

  plt.axhline(y=30, color='green', linestyle='--', linewidth=1.5,label='SOBREVENDA')
  ax.legend(loc='upper left')
  plt.axhline(y=70, color='red', linestyle='--', linewidth=1.5,label='SOBRECOMPRA')
  ax.legend(loc='upper left')
  col2.pyplot(fig)


  candle = go.Figure(data=[go.Candlestick(x=crypto_1w[token_nome]['Date'],
                  open=crypto_1w[token_nome]['Open'],
                  high=crypto_1w[token_nome]['High'],
                  low=crypto_1w[token_nome]['Low'],
                  close=crypto_1w[token_nome]['Close'])])

  candle.update_layout(title=token_nome+' '+intervalo, titlefont=dict(color='gray', size=28), height=500 )
  col1.write(candle)

  # RSI Geral

  col2.table(low_rsi_1w)

  if crypto_1w[token_nome]['RSI'].iloc[-1] <= 30:
    st.write(token_nome,': RSI 1W em sobrevenda')
  elif crypto_1w[token_nome]['RSI'].iloc[-1] < 50:
    st.write(token_nome,': RSI 1W menor que 50')


# Mensal
elif intervalo == 'MENSAL':

  # Gráfico RSI

  fig, ax = plt.subplots(figsize=(16, 6))
  plt.title(token_nome+': RSI '+intervalo)
  plt.grid()

  ax.legend(loc='lower right')
  plt.plot(crypto_1m[token_nome]['Date'], crypto_1m[token_nome]['RSI'] , color='blue', label = 'RSI 1D')
  ax.legend(loc='upper left')

  plt.axhline(y=30, color='green', linestyle='--', linewidth=1.5,label='SOBREVENDA')
  ax.legend(loc='upper left')
  plt.axhline(y=70, color='red', linestyle='--', linewidth=1.5,label='SOBRECOMPRA')
  ax.legend(loc='upper left')
  col2.pyplot(fig)


  candle = go.Figure(data=[go.Candlestick(x=crypto_1m[token_nome]['Date'],
                  open=crypto_1m[token_nome]['Open'],
                  high=crypto_1m[token_nome]['High'],
                  low=crypto_1m[token_nome]['Low'],
                  close=crypto_1m[token_nome]['Close'])])

  candle.update_layout(title=token_nome+' '+intervalo, titlefont=dict(color='gray', size=28), height=500 )
  col1.write(candle)

  # RSI Geral

  col2.table(low_rsi_1m)

  if crypto_1m[token_nome]['RSI'].iloc[-1] <= 30:
    st.write(token_nome,': RSI 1M em sobrevenda')
  elif crypto_1m[token_nome]['RSI'].iloc[-1] < 50:
    st.write(token_nome,': RSI 1M menor que 50')