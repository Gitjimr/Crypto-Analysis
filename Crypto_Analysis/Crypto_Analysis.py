import streamlit as st
from streamlit_drawable_canvas import st_canvas

st.session_state.update(st.session_state)
for k, v in st.session_state.items():
    st.session_state[k] = v

import os

path = os.path.dirname(__file__)

st.set_page_config(
    page_title='CRYPTINHO',
    layout="wide"
)

# hide_menu = '''
#         <style>
#         #MainMenu {visibility: hidden; }
#         footer {visibility: hidden;}
#         </style>
#         '''
# st.markdown(hide_menu, unsafe_allow_html=True)

import math
import statistics as stat
import numpy as np
import pandas as pd
import re

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from os import listdir
from os.path import isfile, join

#import cv2  # OpenCV
from PIL import Image

import io
from io import BytesIO
from io import StringIO

import time
from datetime import date, datetime, timedelta
from workadays import workdays as wd
import requests

from google import genai

from google.genai import types

import yfinance as yf


################# LEITURA #################

try:
    with open("add_symbols.txt", "r", encoding="utf-8") as f:
        raw_lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

    symbols = []
    for ln in raw_lines:
        parts = [p.strip().upper() for p in re.split(r"[,;\s]+", ln) if p.strip()]
        symbols.extend(parts)

    # Remove duplicatas mantendo ordem
    seen = set()
    lista_crypto = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            lista_crypto.append(s)

    print("Símbolos lidos:", lista_crypto)

except Exception as e:
    print("Erro ao ler add_symbols.txt:", e)
    lista_crypto = ["BTC", "ETH"]  # ajuste o fallback como preferir


################# FUNCTIONS #################

# Função para calcular RSI vetorizado
def calculate_rsi(close, window=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


########################################  OUTROS SINAIS:

def add_reversal_indicators(df):
    df["Williams_%R"] = calculate_williams_r(df["High"], df["Low"], df["Close"])
    df["StochRSI"] = calculate_stoch_rsi(df["Close"])
    df["VolumeSpike"] = calculate_volume_spike(df["Volume"])

    adx, di_plus, di_minus = calculate_adx(df["High"], df["Low"], df["Close"])
    df["ADX"] = adx
    df["DI+"] = di_plus
    df["DI-"] = di_minus

    df["BullishDivergence"] = detect_bullish_divergence(df["Close"], df["RSI"])
    df["BearishDivergence"] = detect_bearish_divergence(df["Close"], df["RSI"])

    return df

def calculate_williams_r(high, low, close, window=14):
    highest_high = high.rolling(window=window).max()
    lowest_low = low.rolling(window=window).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)

def calculate_stoch_rsi(close, window=14):
    rsi = calculate_rsi(close, window)
    min_rsi = rsi.rolling(window=window).min()
    max_rsi = rsi.rolling(window=window).max()
    return (rsi - min_rsi) / (max_rsi - min_rsi + 1e-10)

def calculate_volume_spike(volume, window=20, threshold=2.0):
    avg_volume = volume.rolling(window=window).mean()
    return volume > avg_volume * threshold

def detect_bullish_divergence(close, rsi, lookback=14):
    signal = pd.Series(False, index=close.index)
    for i in range(lookback, len(close)):
        # Encontra mínimos locais no preço e RSI, ignorando NaNs
        price_lows = close[i - lookback:i + 1].dropna()
        rsi_lows = rsi[i - lookback:i + 1].dropna()

        # Garante que as janelas não estão vazias
        if price_lows.empty or rsi_lows.empty:
            continue

        min_price_idx = price_lows.idxmin()
        min_rsi_idx = rsi_lows.idxmin()

        # Verifica se o mínimo do preço é anterior ao mínimo do RSI
        if (min_price_idx < min_rsi_idx and
            close[min_rsi_idx] < close[min_price_idx] and
            rsi[min_rsi_idx] > rsi[min_price_idx]):
            signal[i] = True

    return signal

def detect_bearish_divergence(close, rsi, lookback=14):
    signal = pd.Series(False, index=close.index)
    for i in range(lookback, len(close)):
        # Encontra máximos locais no preço e RSI
        price_highs = close[i-lookback:i+1].dropna()
        rsi_highs = rsi[i-lookback:i+1].dropna()

        # Garante que as janelas não estão vazias
        if price_highs.empty or rsi_highs.empty:
            continue

        max_price_idx = price_highs.idxmax()
        max_rsi_idx = rsi_highs.idxmax()

        # Verifica se o máximo do preço é anterior ao máximo do RSI
        if (max_price_idx < max_rsi_idx and
            close[max_rsi_idx] > close[max_price_idx] and
            rsi[max_rsi_idx] < rsi[max_price_idx]):
            signal[i] = True

    return signal

def calculate_adx(high, low, close, window=14):
    # Cálculo do DM+ e DM- usando Pandas Series
    plus_dm = high.diff()
    minus_dm = -low.diff()  # Movimento negativo

    # Mantemos como Series do Pandas
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    # True Range (TR)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Suavização com EMA (mantendo como Series)
    alpha = 1 / window
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    return adx, plus_di, minus_di

########################################

# Função para baixar e processar dados
def fetch_crypto_data(symbols, periodo, freq="1d", data_maxima=None):

    # Define a data máxima como hoje se não for fornecida
    if data_maxima is None:
        data_maxima = pd.Timestamp.today().normalize()
    else:
        data_maxima = pd.to_datetime(data_maxima)

    data = yf.download([f"{sym}-USD" for sym in symbols if sym != "USD"], period=periodo, group_by="ticker")
    results = {}

    for sym in symbols:

        # Se o símbolo for "USD", invertemos os valores de BTC-USD para representar USD-BTC
        if sym == "USD":
            df = data["BTC-USD"].reset_index()
            cols_to_invert = ["Open", "High", "Low", "Close"]
            # Inverte os preços: 1 / preço
            df[cols_to_invert] = 1 / df[cols_to_invert]
        else:
            df = data[f"{sym}-USD"].reset_index()


        # Filtra os dados até a data máxima
        df = df[df['Date'] <= data_maxima]


        # # Imprimir a primeira e última data disponível
        # if not df.empty:
        #     first_date = df['Date'].iloc[0]
        #     last_date = df['Date'].iloc[-1]
        #     print(f"{sym}: Primeira data = {first_date.date()}, Última data = {last_date.date()}")
        # else:
        #     print(f"{sym}: Dados não disponíveis.")



        # Se for semanal, pega apenas de 7 em 7 dias
        if freq == "1s":
            df = df.iloc[::7].reset_index(drop=True)

        # Se for mensal, pega apenas o último dado de cada mês
        elif freq == "1m":
            df["Month"] = df["Date"].dt.to_period("M")  # Cria coluna de mês
            df = df.groupby("Month").last().reset_index()  # Pega o último registro de cada mês

        df["RSI"] = calculate_rsi(df["Close"])
        df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
        df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
        df["EMA350"] = df["Close"].ewm(span=350, adjust=False).mean()
        df["EMA350x2"] = df["Close"].ewm(span=350, adjust=False).mean() * 2
        df["EMA111"] = df["Close"].ewm(span=111, adjust=False).mean()
        df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
        df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
        df["MACD"] = df["EMA12"] - df["EMA26"]
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

        # Cálculo do OBV (On-Balance Volume)
        df["Price_Change"] = df["Close"].diff()  # Diferença entre fechamentos
        df["OBV"] = 0  # Inicializa OBV
        df.loc[df["Price_Change"] > 0, "OBV"] = df["Volume"]  # Se preço sobe, soma volume
        df.loc[df["Price_Change"] < 0, "OBV"] = -df["Volume"]  # Se preço cai, subtrai volume
        df["OBV"] = df["OBV"].cumsum()  # Soma cumulativa para manter o OBV atualizado

        # Cálculo das Bollinger Bands
        df["SMA20"] = df["Close"].rolling(window=20).mean()  # Média móvel simples de 20 períodos
        df["STD20"] = df["Close"].rolling(window=20).std()  # Desvio padrão de 20 períodos
        df["Upper_BB"] = df["SMA20"] + (df["STD20"] * 2)  # Banda superior
        df["Lower_BB"] = df["SMA20"] - (df["STD20"] * 2)  # Banda inferior
        ## Cálculo da largura das Bollinger Bands (Distância entre as bandas)
        df["BB_Width"] = df["Upper_BB"] - df["Lower_BB"]
        ## Verificar se as bandas estão se estreitando (contraindo) ou expandindo
        df["BB_Contraction"] = df["BB_Width"].diff()  # Diferença entre a largura atual e a anterior
        ## Se a largura está diminuindo, as bandas estão se contraindo (estreitando)
        df["BB_Contraction_Status"] = df["BB_Contraction"].apply(lambda x: "Contraindo" if x < 0 else "Expandindo")

        # Sinais adicionais
        df = add_reversal_indicators(df)

        # Cálculo das retrações de Fibonacci (High e Low do último ano)
        # Pegamos o último ano de dados
        one_year_data = df[df["Date"] > df["Date"].max() - pd.DateOffset(years=1)]

        # Encontrar o High e Low do último ano
        high_1y = one_year_data["High"].max()
        low_1y = one_year_data["Low"].min()

        # Calcular as retrações de Fibonacci
        fibonacci_levels = {
            "23.6%": high_1y - (high_1y - low_1y) * 0.236,
            "38.2%": high_1y - (high_1y - low_1y) * 0.382,
            "50%": high_1y - (high_1y - low_1y) * 0.5,
            "61.8%": high_1y - (high_1y - low_1y) * 0.618,
            "78.6%": high_1y - (high_1y - low_1y) * 0.786,
        }

        # Adicionando as retrações de Fibonacci como novas colunas
        for level, value in fibonacci_levels.items():
            df[f"Fib_{level}"] = value

        # Cálculo das expansões de Fibonacci
        fibonacci_extensions = {
            "61.8%": high_1y + (high_1y - low_1y) * 0.618,
            "100%": high_1y + (high_1y - low_1y) * 1.0,
            "161.8%": high_1y + (high_1y - low_1y) * 1.618,
            "261.8%": high_1y + (high_1y - low_1y) * 2.618,
            "423.6%": high_1y + (high_1y - low_1y) * 4.236,
        }

        # Adicionando as expansões de Fibonacci como novas colunas
        for level, value in fibonacci_extensions.items():
            df[f"Ext_{level}"] = value

        results[sym] = df

    return results

########################################  ANALISAR ATIVOS:

def analisar_sinais_cripto(df_analise, nome_periodo_analise, lista_crypto,
                           crypto_1d, crypto_1w, crypto_1m,
                           lista_crypto_25, lista_crypto_50, lista_crypto_100):
    """
    Analisa sinais de compra e venda para uma lista de criptomoedas
    com base em um DataFrame de um período específico (diário, semanal, etc.),
    e também utiliza dados de RSI de outros períodos.

    Args:
        df_analise (pd.DataFrame): DataFrame principal para análise (ex: dados semanais).
        nome_periodo_analise (str): Nome do período que está sendo analisado (ex: "Semanal (1S)").
        lista_crypto (list): Lista de tickers das criptomoedas.
        crypto_1d (pd.DataFrame): DataFrame com dados diários (usado para RSI 1D e variação de preço 1S).
        crypto_1w (pd.DataFrame): DataFrame com dados semanais (usado para RSI 1S).
        crypto_1m (pd.DataFrame): DataFrame com dados mensais (usado para RSI 1M).
        lista_crypto_25 (list): Lista de tokens do Top 25.
        lista_crypto_50 (list): Lista de tokens do Top 50.
        lista_crypto_100 (list): Lista de tokens do Top 100.
        pd (module): Módulo pandas.

    Returns:
        tuple: Contendo as strings de sinais, DataFrames de resumo e variação de preço.
    """

    variacoes_preco_1S_str = "" # Renomeado para evitar conflito de nome com a string externa

    # Inicializando as mensagens com formatação HTML permitida pelo Telegram
    sinais_compra_geral_ = f"<b>* SINAIS DE COMPRA ({nome_periodo_analise}): *</b>\n\n"
    sinais_venda_geral_ = f"<b>* SINAIS DE VENDA ({nome_periodo_analise}): *</b>\n\n"

    sinais_compra_25_ = f"<b>* SINAIS DE COMPRA TOP 25 ({nome_periodo_analise}): *</b>\n\n"
    sinais_venda_25_ = f"<b>* SINAIS DE VENDA TOP 25 ({nome_periodo_analise}): *</b>\n\n"

    sinais_compra_50_ = f"<b>* SINAIS DE COMPRA TOP 50 ({nome_periodo_analise}): *</b>\n\n"
    sinais_venda_50_ = f"<b>* SINAIS DE VENDA TOP 50 ({nome_periodo_analise}): *</b>\n\n"

    sinais_compra_100_ = f"<b>* SINAIS DE COMPRA TOP 100 ({nome_periodo_analise}): *</b>\n\n" # Corrigido de TOP 50 para TOP 100
    sinais_venda_100_ = f"<b>* SINAIS DE VENDA TOP 100 ({nome_periodo_analise}): *</b>\n\n" # Corrigido de TOP 50 para TOP 100


    df_buy_sell_ratio = pd.DataFrame(columns=["Token", f"Buy/Sell Ratio"])
    df_agregado_compra_venda = pd.DataFrame(columns=["Token", f"Compra x Venda"])


    for token in lista_crypto:

        sinais_compra_token = "" # Renomeado para escopo local
        sinais_venda_token = ""  # Renomeado para escopo local

        # Destaque do token: negrito e sublinhado
        token_destacado = f"<b><u>{token}</u></b>"

        # Inicializando os contadores
        razao_compra = 0
        total_sinais_compra = 9 + 8 # Mantido conforme original, ajuste se os tipos de sinais mudarem

        razao_venda = 0
        total_sinais_venda = 11 + 8 # Mantido conforme original

        # -------------- SINAIS DE COMPRA (🟢) --------------
        # EMA50 > EMA200 (Bullish)
        if df_analise[token]['EMA200'].iloc[-1] < df_analise[token]['EMA50'].iloc[-1]:
            razao_compra += 1
            rc = f"<b>[{razao_compra}/{total_sinais_compra}]</b>"
            sinais_compra_token += (f"🟢 {token_destacado} ({nome_periodo_analise}): <i>EMA200</i> (<code>{df_analise[token]['EMA200'].iloc[-1]}</code>) menor que "
                                  f"<i>EMA50</i> (<code>{df_analise[token]['EMA50'].iloc[-1]}</code>) → <b>Bull</b> {rc}\n")
            # Condição: Preço abaixo das EMAs 50, 111 e 200, mas acima da EMA350
            if (df_analise[token]['Close'].iloc[-1] < df_analise[token]['EMA50'].iloc[-1] and
                df_analise[token]['Close'].iloc[-1] < df_analise[token]['EMA111'].iloc[-1] and
                df_analise[token]['Close'].iloc[-1] <= df_analise[token]['EMA200'].iloc[-1] and
                df_analise[token]['Close'].iloc[-1] >= df_analise[token]['EMA350'].iloc[-1]):
                razao_compra += 1
                rc = f"<b>[{razao_compra}/{total_sinais_compra}]</b>"
                sinais_compra_token += (f"🔽 {token_destacado} ({nome_periodo_analise}): Preço abaixo das <i>EMA's 50, 111 e 200</i>, mas acima do suporte da "
                                      f"<i>EMA350</i> (<code>{df_analise[token]['EMA350'].iloc[-1]}</code>) {rc}\n")

        # RSI em sobrevenda (≤ 30) - Referências específicas mantidas
        if crypto_1d[token]['RSI'].iloc[-1] <= 30:
            razao_compra += 1
            rc = f"<b>[{razao_compra}/{total_sinais_compra}]</b>"
            sinais_compra_token += (f"📉 {token_destacado}: RSI <b>1D</b> em sobrevenda (<code>{crypto_1d[token]['RSI'].iloc[-1]:.2f}</code>) {rc}\n")
        if crypto_1w[token]['RSI'].iloc[-1] <= 30: # Supondo que crypto_1w está disponível
            razao_compra += 1
            rc = f"<b>[{razao_compra}/{total_sinais_compra}]</b>"
            sinais_compra_token += (f"📉 {token_destacado}: RSI <b>1S</b> em sobrevenda (<code>{crypto_1w[token]['RSI'].iloc[-1]:.2f}</code>) {rc}\n")
        if crypto_1m[token]['RSI'].iloc[-1] <= 30: # Supondo que crypto_1m está disponível
            razao_compra += 1
            rc = f"<b>[{razao_compra}/{total_sinais_compra}]</b>"
            sinais_compra_token += (f"📉 {token_destacado}: RSI <b>1M</b> em sobrevenda (<code>{crypto_1m[token]['RSI'].iloc[-1]:.2f}</code>) {rc}\n")

        # MACD positivo (histograma > 0)
        if df_analise[token]['MACD_Hist'].iloc[-1] > 0:
            razao_compra += 1
            rc = f"<b>[{razao_compra}/{total_sinais_compra}]</b>"
            sinais_compra_token += (f"📈 {token_destacado} ({nome_periodo_analise}): MACD acima da linha de sinal → <b>Bull</b> {rc}\n")

        # OBV e Price Change
        if df_analise[token]['OBV'].iloc[-1] > 0:
            razao_compra += 1
            rc = f"<b>[{razao_compra}/{total_sinais_compra}]</b>"
            if df_analise[token]['Price_Change'].iloc[-1] > 0:
                sinais_compra_token += (f"📊 {token_destacado} ({nome_periodo_analise}): OBV e Price Change subindo → Tendência de alta confiável {rc}\n")
            elif df_analise[token]['Price_Change'].iloc[-1] < 0:
                sinais_compra_token += (f"🔽 {token_destacado} ({nome_periodo_analise}): OBV subindo e Price Change caindo → Possível suporte/recuperação {rc}\n")

        # Preço abaixo da Banda de Bollinger Inferior
        if df_analise[token]['Lower_BB'].iloc[-1] > df_analise[token]['Close'].iloc[-1]:
            razao_compra += 1
            rc = f"<b>[{razao_compra}/{total_sinais_compra}]</b>"
            if df_analise[token]["BB_Contraction_Status"].iloc[-1] == "Contraindo":
                sinais_compra_token += (f"📉 {token_destacado} ({nome_periodo_analise}): Preço abaixo da banda de Bollinger inferior (<code>{df_analise[token]['Lower_BB'].iloc[-1]}</code>) → "
                                      f"Pressão de compra, contraindo → Grande movimento ruptivo se aproximando {rc}\n")
            else:
                sinais_compra_token += (f"📉 {token_destacado} ({nome_periodo_analise}): Preço abaixo da banda de Bollinger inferior (<code>{df_analise[token]['Lower_BB'].iloc[-1]}</code>) → "
                                      f"Pressão de compra, expandindo → Alta volatilidade {rc}\n")

        # Preço entre retrações de Fibonacci de 61.8% e 38,2% (Golden Pocket)
        if (df_analise[token]['Fib_38.2%'].iloc[-1] >= df_analise[token]['Close'].iloc[-1] and
            df_analise[token]['Close'].iloc[-1] >= df_analise[token]['Fib_61.8%'].iloc[-1]):
            razao_compra += 1
            rc = f"<b>[{razao_compra}/{total_sinais_compra}]</b>"
            sinais_compra_token += (f"🔺 {token_destacado} ({nome_periodo_analise}): Preço entre retrações de Fibo de 38.2% (<code>{df_analise[token]['Fib_38.2%'].iloc[-1]}</code>) (Compra arrojada)"
                                  f"e 61,8% (<code>{df_analise[token]['Fib_61.8%'].iloc[-1]}</code>) (Golden Pocket - Compra conservadora) {rc}\n")

        # OUTROS SINAIS DE COMPRA:
        if df_analise[token]['Williams_%R'].iloc[-1] <= -80:
            razao_compra += 1
            rc = f"<b>[{razao_compra}/{total_sinais_compra}]</b>"
            sinais_compra_token += (f"🟢 {token_destacado} ({nome_periodo_analise}): Williams %R (<code>{df_analise[token]['Williams_%R'].iloc[-1]:.2f}</code>) <= -80 (Oversold) {rc}\n")

        if df_analise[token]['StochRSI'].iloc[-1] <= 0.2:
            razao_compra += 1
            rc = f"<b>[{razao_compra}/{total_sinais_compra}]</b>"
            sinais_compra_token += (f"🟢 {token_destacado} ({nome_periodo_analise}): StochRSI (<code>{df_analise[token]['StochRSI'].iloc[-1]:.2f}</code>) <= 0.2 (Oversold) {rc}\n")

        if df_analise[token]['VolumeSpike'].iloc[-1] and df_analise[token]['Close'].iloc[-1] > df_analise[token]['Close'].iloc[-2]:
            razao_compra += 1
            rc = f"<b>[{razao_compra}/{total_sinais_compra}]</b>"
            sinais_compra_token += (f"🟢 {token_destacado} ({nome_periodo_analise}): Volume Spike com preço subindo (Acumulação) {rc}\n")

        if df_analise[token]['ADX'].iloc[-1] > 25 and df_analise[token]['DI+'].iloc[-1] > df_analise[token]['DI-'].iloc[-1]:
            razao_compra += 1
            rc = f"<b>[{razao_compra}/{total_sinais_compra}]</b>"
            sinais_compra_token += (f"🟢 {token_destacado} ({nome_periodo_analise}): ADX (<code>{df_analise[token]['ADX'].iloc[-1]:.2f}</code>) > 25 e DI+ > DI- (Tendência de Alta) {rc}\n")

        if df_analise[token]['BullishDivergence'].iloc[-1]:
            razao_compra += 1
            rc = f"<b>[{razao_compra}/{total_sinais_compra}]</b>"
            sinais_compra_token += (f"🟢 {token_destacado} ({nome_periodo_analise}): Divergência de Alta detectada {rc}\n")

        if (df_analise[token]['DI+'].iloc[-1] > df_analise[token]['DI-'].iloc[-1] and
            df_analise[token]['DI+'].iloc[-2] < df_analise[token]['DI-'].iloc[-2]):
            razao_compra += 1
            rc = f"<b>[{razao_compra}/{total_sinais_compra}]</b>"
            sinais_compra_token += (f"🟢 {token_destacado} ({nome_periodo_analise}): DI+ cruzou acima do DI- (Mudança de Tendência) {rc}\n")

        if (df_analise[token]['Williams_%R'].iloc[-1] > -80 and
            df_analise[token]['Williams_%R'].iloc[-2] <= -80):
            razao_compra += 1
            rc = f"<b>[{razao_compra}/{total_sinais_compra}]</b>"
            sinais_compra_token += (f"🟢 {token_destacado} ({nome_periodo_analise}): Williams %R cruzando acima de -80 (Saindo de Oversold) {rc}\n")

        if (df_analise[token]['StochRSI'].iloc[-1] > 0.2 and
            df_analise[token]['StochRSI'].iloc[-2] <= 0.2):
            razao_compra += 1
            rc = f"<b>[{razao_compra}/{total_sinais_compra}]</b>"
            sinais_compra_token += (f"🟢 {token_destacado} ({nome_periodo_analise}): StochRSI cruzando acima de 0.2 (Momentum Positivo) {rc}\n")

        # -------------- SINAIS DE VENDA (🔴) --------------
        # EMA50 < EMA200 (Bearish)
        if df_analise[token]['EMA200'].iloc[-1] > df_analise[token]['EMA50'].iloc[-1]:
            razao_venda += 1
            rv = f"<b>[{razao_venda}/{total_sinais_venda}]</b>"
            sinais_venda_token += (f"🔴 {token_destacado} ({nome_periodo_analise}): <i>EMA200</i> (<code>{df_analise[token]['EMA200'].iloc[-1]}</code>) > "
                                 f"<i>EMA50</i> (<code>{df_analise[token]['EMA50'].iloc[-1]}</code>) → <b>Bear</b> {rv}\n")

        # Preço acima de todas as EMAs
        if (df_analise[token]['Close'].iloc[-1] > df_analise[token]['EMA50'].iloc[-1] and
            df_analise[token]['Close'].iloc[-1] > df_analise[token]['EMA111'].iloc[-1] and
            df_analise[token]['Close'].iloc[-1] > df_analise[token]['EMA200'].iloc[-1] and
            df_analise[token]['Close'].iloc[-1] > df_analise[token]['EMA350'].iloc[-1]):
            razao_venda += 1
            rv = f"<b>[{razao_venda}/{total_sinais_venda}]</b>"
            sinais_venda_token += (f"🔴 {token_destacado} ({nome_periodo_analise}): Preço acima das <i>EMA's 50, 111 e 200</i> e acima da <i>EMA350</i> (suporte) {rv}\n")

        # Preço acima da EMA350x2
        if df_analise[token]['Close'].iloc[-1] > df_analise[token]['EMA350x2'].iloc[-1]:
            razao_venda += 1
            rv = f"<b>[{razao_venda}/{total_sinais_venda}]</b>"
            sinais_venda_token += (f"🔴 {token_destacado} ({nome_periodo_analise}): Preço acima da <i>EMA350 x 2</i> (<code>{df_analise[token]['EMA350x2'].iloc[-1]}</code>) → "
                                 f"Atenção ao cruzamento com <i>EMA111</i> (<code>{df_analise[token]['EMA111'].iloc[-1]}</code>) {rv}\n")

        # Pi Cycle Top (alerta)
        if (df_analise[token]['EMA111'].iloc[-1] / df_analise[token]['EMA350x2'].iloc[-1]) >= 0.9: # Divisão por zero pode ocorrer se EMA350x2 for zero
            razao_venda += 1
            rv = f"<b>[{razao_venda}/{total_sinais_venda}]</b>"
            sinais_venda_token += (f"⚠️ {token_destacado} ({nome_periodo_analise}): Pi Cycle Top em <i>ZONA DE ALERTA</i> → Topo em formação com "
                                 f"<i>EMA350x2</i> (<code>{df_analise[token]['EMA350x2'].iloc[-1]}</code>) e "
                                 f"<i>EMA111</i> (<code>{df_analise[token]['EMA111'].iloc[-1]}</code>) se cruzando {rv}\n")

        # RSI em sobrecompra (≥ 70) - Referências específicas mantidas
        if crypto_1d[token]['RSI'].iloc[-1] >= 70:
            razao_venda += 1
            rv = f"<b>[{razao_venda}/{total_sinais_venda}]</b>"
            sinais_venda_token += (f"📈 {token_destacado}: RSI <b>1D</b> em sobrecompra (<code>{crypto_1d[token]['RSI'].iloc[-1]:.2f}</code>) {rv}\n")
        if crypto_1w[token]['RSI'].iloc[-1] >= 70: # Supondo que crypto_1w está disponível
            razao_venda += 1
            rv = f"<b>[{razao_venda}/{total_sinais_venda}]</b>"
            sinais_venda_token += (f"📈 {token_destacado}: RSI <b>1S</b> em sobrecompra (<code>{crypto_1w[token]['RSI'].iloc[-1]:.2f}</code>) {rv}\n")
        if crypto_1m[token]['RSI'].iloc[-1] >= 70: # Supondo que crypto_1m está disponível
            razao_venda += 1
            rv = f"<b>[{razao_venda}/{total_sinais_venda}]</b>"
            sinais_venda_token += (f"📈 {token_destacado}: RSI <b>1M</b> em sobrecompra (<code>{crypto_1m[token]['RSI'].iloc[-1]:.2f}</code>) {rv}\n")

        # MACD negativo (histograma < 0)
        if df_analise[token]['MACD_Hist'].iloc[-1] < 0:
            razao_venda += 1
            rv = f"<b>[{razao_venda}/{total_sinais_venda}]</b>"
            sinais_venda_token += (f"📉 {token_destacado} ({nome_periodo_analise}): MACD abaixo da linha de sinal → <b>Bear</b> {rv}\n")

        # OBV negativo e Price Change
        if df_analise[token]['OBV'].iloc[-1] < 0:
            razao_venda += 1
            rv = f"<b>[{razao_venda}/{total_sinais_venda}]</b>"
            if df_analise[token]['Price_Change'].iloc[-1] > 0:
                sinais_venda_token += (f"🔻 {token_destacado} ({nome_periodo_analise}): OBV caindo e Price Change subindo → Divergência → Alerta de reversão! {rv}\n")
            elif df_analise[token]['Price_Change'].iloc[-1] < 0:
                sinais_venda_token += (f"🔻 {token_destacado} ({nome_periodo_analise}): OBV e Price Change caindo → Tendência de baixa confirmada {rv}\n")

        # Preço acima da Banda de Bollinger Superior (originalmente estava como "abaixo da inferior" na seção de venda, corrigido para lógica de venda)
        if df_analise[token]['Upper_BB'].iloc[-1] < df_analise[token]['Close'].iloc[-1]: # Corrigido: venda se preço acima da banda superior
            razao_venda += 1
            rv = f"<b>[{razao_venda}/{total_sinais_venda}]</b>"
            # A descrição original para este sinal de VENDA parecia copiada da COMPRA. Ajustei para um contexto de venda.
            if df_analise[token]["BB_Contraction_Status"].iloc[-1] == "Contraindo":
                 sinais_venda_token += (f"📈 {token_destacado} ({nome_periodo_analise}): Preço acima da banda de Bollinger superior (<code>{df_analise[token]['Upper_BB'].iloc[-1]}</code>) → "
                                   f"Pressão de venda, contraindo → Grande movimento ruptivo se aproximando {rv}\n")
            else:
                 sinais_venda_token += (f"📈 {token_destacado} ({nome_periodo_analise}): Preço acima da banda de Bollinger superior (<code>{df_analise[token]['Upper_BB'].iloc[-1]}</code>) → "
                                   f"Pressão de venda, expandindo → Alta volatilidade {rv}\n")


        # Preço abaixo da retração de Fibonacci de 78.6%
        if df_analise[token]['Fib_78.6%'].iloc[-1] > df_analise[token]['Close'].iloc[-1]:
            razao_venda += 1
            rv = f"<b>[{razao_venda}/{total_sinais_venda}]</b>"
            sinais_venda_token += (f"📉 {token_destacado} ({nome_periodo_analise}): Preço abaixo da retração de Fibo de 78.6% (<code>{df_analise[token]['Fib_78.6%'].iloc[-1]}</code>) {rv}\n")

        # Preço entre as extensões de Fibonacci de 100% e 161.8%
        if (df_analise[token]['Ext_161.8%'].iloc[-1] >= df_analise[token]['Close'].iloc[-1] and
            df_analise[token]['Close'].iloc[-1] >= df_analise[token]['Ext_100%'].iloc[-1]):
            razao_venda += 1
            rv = f"<b>[{razao_venda}/{total_sinais_venda}]</b>"
            sinais_venda_token += (f"📉 {token_destacado} ({nome_periodo_analise}): Preço entre as extensões de Fibo de 100% (<code>{df_analise[token]['Ext_100%'].iloc[-1]}</code>) "
                                 f"e 161,8% (<code>{df_analise[token]['Ext_161.8%'].iloc[-1]}</code>) {rv}\n")

        # OUTROS SINAIS DE VENDA:
        if df_analise[token]['Williams_%R'].iloc[-1] >= -20:
            razao_venda += 1
            rv = f"<b>[{razao_venda}/{total_sinais_venda}]</b>"
            sinais_venda_token += (f"🔴 {token_destacado} ({nome_periodo_analise}): Williams %R (<code>{df_analise[token]['Williams_%R'].iloc[-1]:.2f}</code>) >= -20 (Overbought) {rv}\n")

        if df_analise[token]['StochRSI'].iloc[-1] >= 0.8:
            razao_venda += 1
            rv = f"<b>[{razao_venda}/{total_sinais_venda}]</b>"
            sinais_venda_token += (f"🔴 {token_destacado} ({nome_periodo_analise}): StochRSI (<code>{df_analise[token]['StochRSI'].iloc[-1]:.2f}</code>) >= 0.8 (Overbought) {rv}\n")

        if df_analise[token]['VolumeSpike'].iloc[-1] and df_analise[token]['Close'].iloc[-1] < df_analise[token]['Close'].iloc[-2]:
            razao_venda += 1
            rv = f"<b>[{razao_venda}/{total_sinais_venda}]</b>"
            sinais_venda_token += (f"🔴 {token_destacado} ({nome_periodo_analise}): Volume Spike com preço caindo (Distribuição) {rv}\n")

        if df_analise[token]['ADX'].iloc[-1] > 25 and df_analise[token]['DI-'].iloc[-1] > df_analise[token]['DI+'].iloc[-1]:
            razao_venda += 1
            rv = f"<b>[{razao_venda}/{total_sinais_venda}]</b>"
            sinais_venda_token += (f"🔴 {token_destacado} ({nome_periodo_analise}): ADX (<code>{df_analise[token]['ADX'].iloc[-1]:.2f}</code>) > 25 e DI- > DI+ (Tendência de Baixa) {rv}\n")

        if (df_analise[token]['DI-'].iloc[-1] > df_analise[token]['DI+'].iloc[-1] and
            df_analise[token]['DI-'].iloc[-2] < df_analise[token]['DI+'].iloc[-2]):
            razao_venda += 1
            rv = f"<b>[{razao_venda}/{total_sinais_venda}]</b>"
            sinais_venda_token += (f"🔴 {token_destacado} ({nome_periodo_analise}): DI- cruzou acima do DI+ (Mudança de Tendência) {rv}\n")

        if (df_analise[token]['Williams_%R'].iloc[-1] < -20 and
            df_analise[token]['Williams_%R'].iloc[-2] >= -20):
            razao_venda += 1
            rv = f"<b>[{razao_venda}/{total_sinais_venda}]</b>"
            sinais_venda_token += (f"🔴 {token_destacado} ({nome_periodo_analise}): Williams %R cruzando abaixo de -20 (Saindo de Overbought) {rv}\n")

        if (df_analise[token]['StochRSI'].iloc[-1] < 0.8 and
            df_analise[token]['StochRSI'].iloc[-2] >= 0.8):
            razao_venda += 1
            rv = f"<b>[{razao_venda}/{total_sinais_venda}]</b>"
            sinais_venda_token += (f"🔴 {token_destacado} ({nome_periodo_analise}): StochRSI cruzando abaixo de 0.8 (Momentum Negativo) {rv}\n")

        if df_analise[token]['BearishDivergence'].iloc[-1]:
            razao_venda += 1
            rv = f"<b>[{razao_venda}/{total_sinais_venda}]</b>"
            sinais_venda_token += (f"🔴 {token_destacado} ({nome_periodo_analise}): Divergência de Baixa detectada "
                                f"{rv}\n")


        # Ajustar listas com base no volume de mercado

        sinais_compra_geral_ += sinais_compra_token
        sinais_venda_geral_ += sinais_venda_token

        sinais_compra_25_ += sinais_compra_token if token in lista_crypto_25 else ""
        sinais_venda_25_ += sinais_venda_token if token in lista_crypto_25 else ""

        sinais_compra_50_ += sinais_compra_token if token in lista_crypto_50 else ""
        sinais_venda_50_ += sinais_venda_token if token in lista_crypto_50 else ""

        sinais_compra_100_ += sinais_compra_token if token in lista_crypto_100 else ""
        sinais_venda_100_ += sinais_venda_token if token in lista_crypto_100 else ""


        # Razão de Compra/Venda do Token
        # Adicionado tratamento para evitar divisão por zero se razao_venda for 0
        ratio_cv = (razao_compra + 1) / (razao_venda + 1)
        df_buy_sell_ratio.loc[len(df_buy_sell_ratio)] = [token, ratio_cv]
        df_agregado_compra_venda.loc[len(df_agregado_compra_venda)] = [token, f"COMPRA: {razao_compra}/{total_sinais_compra}, VENDA: {razao_venda}/{total_sinais_venda}"]


        # Variação no preço 1S (calculada com base no DataFrame diário 'crypto_1d')
        # Esta parte permanece usando 'crypto_1d' para a variação semanal específica.
        try:
            preco_atual_1d = crypto_1d[token]['Close'].iloc[-1]
            preco_7d_atras_1d = crypto_1d[token]['Close'].iloc[-7]
            variacao_1s = ((preco_atual_1d - preco_7d_atras_1d) / preco_7d_atras_1d) * 100 # Multiplicado por 100 para percentual
            variacoes_preco_1S_str += (f"📊 {token_destacado} : Preço Atual (1D) = <code>{preco_atual_1d:.4f}</code> // "
                                     f"Preço 7 dias atrás (1D) = <code>{preco_7d_atras_1d:.4f}</code> // "
                                     f"Variação (7D) = {variacao_1s:.2f}%\n")
        except IndexError:
            variacoes_preco_1S_str += f"📊 {token_destacado} : Dados insuficientes em crypto_1d para calcular variação de 7 dias.\n"
        except KeyError:
            variacoes_preco_1S_str += f"📊 {token_destacado} : Token não encontrado em crypto_1d para calcular variação de 7 dias.\n"


    return (sinais_compra_geral_, sinais_venda_geral_,
            sinais_compra_25_, sinais_venda_25_,
            sinais_compra_50_, sinais_venda_50_,
            sinais_compra_100_, sinais_venda_100_,
            df_buy_sell_ratio, df_agregado_compra_venda,
            variacoes_preco_1S_str)


def exibir_sinais_streamlit(
    sinais_compra_25, sinais_venda_25,
    sinais_compra_50, sinais_venda_50,
    sinais_compra_100, sinais_venda_100,
    variacoes_preco_1S_str,
    sinais_compra_geral, sinais_venda_geral,
    df_buy_sell_ratio, df_agregado_compra_venda,
    token_sel: str = 'BTC', periodicidade: str = '1W'
):
    import re
    import pandas as pd
    import streamlit as st

    # ========= Helpers =========
    def strip_html(s: str) -> str:
        return re.sub(r"<[^>]+>", "", s or "")

    def parse_signals(texto: str) -> pd.DataFrame:
        if not texto:
            return pd.DataFrame(columns=["icon", "token", "mensagem", "periodo"])

        linhas = [ln for ln in texto.splitlines() if ln.strip()]
        periodo = ""
        if linhas and "* SINAIS" in strip_html(linhas[0]):
            periodo = strip_html(linhas[0]).replace("*", "").strip()

        rows = []
        for ln in linhas:
            if "* SINAIS" in ln:
                continue
            ln_clean = ln.strip()
            if len(ln_clean) < 2:
                continue

            icon = ""
            head = ln_clean[:2]
            if any(e in head for e in ["🟢", "🔴", "📈", "📉", "🔻", "🔺", "📊", "⚠️", "🔽"]):
                icon = head.strip()
            elif ln_clean[:1] in ["🟢", "🔴", "📈", "📉", "🔻", "🔺", "📊", "⚠️", "🔽"]:
                icon = ln_clean[:1]

            m_token = re.search(r"<b><u>([^<]+)</u></b>", ln_clean)
            token = m_token.group(1) if m_token else ""

            msg = strip_html(ln_clean)
            msg = re.sub(r"\s*\[\d+/\d+\]\s*$", "", msg).strip()

            rows.append({"icon": icon, "token": token, "mensagem": msg, "periodo": periodo})

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df[["icon", "token", "mensagem", "periodo"]]
        return df

    def concat_parse(*blocos: str) -> pd.DataFrame:
        dfs = [parse_signals(b) for b in blocos if b]
        if not dfs:
            return pd.DataFrame(columns=["icon", "token", "mensagem", "periodo"])
        df = pd.concat(dfs, ignore_index=True)
        return df.sort_values(by=["token", "icon"], ignore_index=True)

    # ========= UI =========
    st.subheader(f"📈 Indicadores {periodicidade}")

    tab_geral, tab_variacoes, tab_ratios, tab_resumo = st.tabs(
        [f"Sinais {token_sel}", "Variação 7D", "Buy/Sell Ratio", "Compra × Venda"]
    )

    # ===== Geral =====
    with tab_geral:
        st.caption("Sinais agregados (Compra e Venda)")
        df_geral = concat_parse(sinais_compra_geral, sinais_venda_geral)
        filtro_token = token_sel

        df_show = df_geral.copy()
        df_show = df_show[df_show["token"].str.contains(filtro_token, case=False, na=False)]

        ordem_icon = {"🟢": 0, "🔺": 0, "📈": 0, "📊": 1, "⚠️": 2, "🔽": 3, "📉": 4, "🔻": 4, "🔴": 5}
        df_show["ord_icon"] = df_show["icon"].map(ordem_icon).fillna(9)
        df_show = df_show.sort_values(by=["token", "ord_icon"]).drop(columns=["ord_icon"])

        st.dataframe(
            df_show.rename(columns={"icon": " ", "token": "Token", "mensagem": "Sinal", "periodo": "Tipo de Sinal"}),
            use_container_width=True,
            hide_index=True
        )


    # ===== Variações 7D =====
    with tab_variacoes:
        st.caption("Resumo de variação de preço em 7 dias (1D)")
        linhas = [ln for ln in variacoes_preco_1S_str.splitlines() if ln.strip()]
        rows = []
        for ln in linhas:
            token_m = re.search(r"<b><u>([^<]+)</u></b>", ln)
            token = token_m.group(1) if token_m else ""
            cur_m = re.search(r"Atual\s*\(1D\)\s*=\s*<code>([^<]+)</code>", ln)
            old_m = re.search(r"7 dias atrás\s*\(1D\)\s*=\s*<code>([^<]+)</code>", ln)
            var_m = re.search(r"Variação\s*\(7D\)\s*=\s*([-\d.,]+)%", ln)
            atual = float(cur_m.group(1)) if cur_m else None
            antigo = float(old_m.group(1)) if old_m else None
            variacao = float(var_m.group(1).replace(",", ".")) if var_m else None
            rows.append({
                "Token": token,
                "Preço atual (1D)": atual,
                "Preço 7D atrás (1D)": antigo,
                "Variação (7D) %": variacao
            })
        df_var = pd.DataFrame(rows)
        if not df_var.empty:
            df_var = df_var.sort_values(by="Variação (7D) %", ascending=False)

        def highlight_token(row):
            if row["Token"] == token_sel:
                return [f"background-color: #ff4d4d; font-weight: bold;"] * len(row)
            return [""] * len(row)

        styled_df_var = df_var.style.apply(highlight_token, axis=1)

        st.dataframe(
            styled_df_var,
            use_container_width=True, hide_index=True
        )


    # ===== Buy/Sell Ratio =====
    with tab_ratios:
        st.caption("Buy/Sell Ratio (quanto maior, mais 'atrativo' o conjunto de sinais do token)")
        if not df_buy_sell_ratio.empty:
            df_r = df_buy_sell_ratio.copy()
            df_r = df_r.sort_values(by="Buy/Sell Ratio", ascending=False)
            df_r["Buy/Sell Ratio"] = df_r["Buy/Sell Ratio"].astype(float).round(3)

            df_r.rename(columns={"Token": "Token", "Buy/Sell Ratio": "Buy/Sell Ratio"})
            styled_df_r = df_r.style.apply(highlight_token, axis=1)

            st.dataframe(
                styled_df_r,
                use_container_width=True, hide_index=True
            )

        else:
            st.info("Sem dados de Buy/Sell Ratio.")

    # ===== Compra × Venda =====
    with tab_resumo:
        st.caption("Contadores agregados (COMPRA/VENDA) por token")
        if not df_agregado_compra_venda.empty:
            def split_counts(s: str):
                compra_m = re.search(r"COMPRA:\s*(\d+/\d+)", s or "")
                venda_m  = re.search(r"VENDA:\s*(\d+/\d+)", s or "")
                return (compra_m.group(1) if compra_m else "", venda_m.group(1) if venda_m else "")

            df_cv = df_agregado_compra_venda.copy()
            compra_venda = df_cv["Compra x Venda"].apply(split_counts)
            df_cv["Compra"] = compra_venda.apply(lambda x: x[0])
            df_cv["Venda"]  = compra_venda.apply(lambda x: x[1])
            df_cv = df_cv.drop(columns=["Compra x Venda"]).rename(columns={"Token": "Token"})

            styled_df_cv = df_cv.style.apply(highlight_token, axis=1)

            st.dataframe(styled_df_cv, use_container_width=True, hide_index=True)
        else:
            st.info("Sem dados de Compra × Venda.")

##############################

def plot_weekly_with_indicators(
    s_close=None,
    s_date=None,
    df=None,
    token_sel = 'BTC',
    compute_missing=False,
    rsi_period=14
):

    # -------- preparar dados de entrada --------
    if df is None:
        if s_close is None or s_date is None:
            raise ValueError("Forneça df OU (s_close e s_date).")
        df = pd.DataFrame({'Date': pd.to_datetime(s_date), 'Close': pd.to_numeric(s_close)})
    else:
        df = df.copy()
        # normalizar nomes mais comuns (case-insensitive)
        cols_lower = {c.lower(): c for c in df.columns}
        # garantir colunas padrão se existirem com variações
        if 'date' in cols_lower and 'Date' not in df.columns:
            df.rename(columns={cols_lower['date']: 'Date'}, inplace=True)
        if 'close' in cols_lower and 'Close' not in df.columns:
            df.rename(columns={cols_lower['close']: 'Close'}, inplace=True)

    # ordenar por data e garantir datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)
    else:
        # se não tiver Date, tenta índice datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("O DataFrame precisa ter coluna 'Date' ou índice DateTime.")
        df = df.sort_index()
        df['Date'] = df.index

    # -------- calcular indicadores ausentes (opcional) --------
    if compute_missing:
        if 'EMA50' not in df.columns and 'Close' in df.columns:
            df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        if 'EMA200' not in df.columns and 'Close' in df.columns:
            df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
        if 'RSI' not in df.columns and 'Close' in df.columns:
            # RSI (Wilder)
            delta = df['Close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
            rs = avg_gain / (avg_loss.replace(0, np.nan))
            df['RSI'] = 100 - (100 / (1 + rs))

    # -------- criar figura com eixo secundário --------
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Close (primário)
    if 'Close' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['Close'],
                mode='lines',
                name='BTC Close',
                line=dict(width=2)
            ),
            secondary_y=False
        )

    # EMAs (primário, se existirem)
    if 'EMA50' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['EMA50'],
                mode='lines',
                name='EMA50',
                line=dict(width=1.5, dash='dot')
            ),
            secondary_y=False
        )

    if 'EMA200' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['EMA200'],
                mode='lines',
                name='EMA200',
                line=dict(width=1.5, dash='dash')
            ),
            secondary_y=False
        )

    # RSI (secundário)
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['RSI'],
                mode='lines',
                name='RSI',
                line=dict(width=2)
            ),
            secondary_y=True
        )
        # linhas guia 30/50/70 no eixo RSI
        for lvl, dash in [(30, 'dot'), (50, 'dash'), (70, 'dot')]:
            fig.add_hline(
                y=lvl,
                line=dict(width=1, dash=dash),
                secondary_y=True
            )

    # -------- Fibs/Extensões (podem ser séries ou valores constantes) --------
    fib_cols = ['Fib_38.2%', 'Fib_50%', 'Fib_61.8%', 'Ext_61.8%', 'Ext_100%', 'Ext_161.8%']
    for col in fib_cols:
        if col in df.columns:
            s = df[col]
            # se for constante (ou quase), usa hline; senão, plota como linha
            if pd.api.types.is_numeric_dtype(s):
                if s.nunique(dropna=True) <= 1:
                    level = float(s.dropna().iloc[-1]) if s.dropna().shape[0] else None
                    if level is not None:
                        fig.add_hline(
                            y=level,
                            line=dict(width=1),
                            annotation_text=col,
                            annotation_position='top left',
                            secondary_y=False
                        )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=df['Date'],
                            y=s,
                            mode='lines',
                            name=col,
                            line=dict(width=1)
                        ),
                        secondary_y=False
                    )

    # -------- layout --------
    fig.update_layout(
        title=f'{token_sel} Semanal: Close, EMAs, RSI e Fibs/Extensões',
        template='plotly_dark',
        hovermode='x unified',
        margin=dict(l=60, r=60, t=60, b=40),
        legend=dict(x=0.01, y=0.99)
    )

    fig.update_xaxes(title_text='Data')
    fig.update_yaxes(
        title_text='Preço (USD)',
        secondary_y=False
    )
    fig.update_yaxes(
        title_text='RSI',
        range=[0, 100],
        secondary_y=True
    )

    st.plotly_chart(fig)


def plot_with_semicircles(
    s_close, s_date,
    token_sel = 'BTC',
    x0=datetime(2025, 3, 2), #~CamelFinance's
    num_ondas=8,
    periodo_sem=4 * 7,
    y_pico=10
):

    # --- gerar ondas ---
    x_vals = []
    y_vals = []

    # ondas pra trás
    for i in range(num_ondas // 2, 0, -1):
        x_start = x0 - timedelta(weeks=i * periodo_sem)
        x_end = x_start + timedelta(weeks=periodo_sem)
        x_wave = pd.date_range(start=x_start, end=x_end, periods=100)
        t = np.linspace(0, np.pi, 100)
        y_wave = y_pico * np.sin(t)
        x_vals.extend(x_wave)
        y_vals.extend(y_wave)

    # ondas pra frente
    for i in range(num_ondas // 2):
        x_start = x0 + timedelta(weeks=i * periodo_sem)
        x_end = x_start + timedelta(weeks=periodo_sem)
        x_wave = pd.date_range(start=x_start, end=x_end, periods=100)
        t = np.linspace(0, np.pi, 100)
        y_wave = y_pico * np.sin(t)
        x_vals.extend(x_wave)
        y_vals.extend(y_wave)

    # --- plotar tudo ---
    fig = go.Figure()

    # BTC Close (eixo Y da esquerda)
    fig.add_trace(go.Scatter(
        x=s_date.values,
        y=s_close.values,
        mode='lines',
        name='Close',
        line=dict(color='gold', width=2),
        yaxis='y1'
    ))

    # Ondas semicirculares (eixo Y da direita)
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines',
        name='Semiciclos',
        line=dict(color='cyan', width=3, dash='dot'),
        yaxis='y2'
    ))

    # --- layout ---
    fig.update_layout(
        title=f'{token_sel} Close + Ondas Simétricas',
        template='plotly_dark',
        hovermode='x unified',
        xaxis=dict(title='Data'),
        yaxis=dict(
            title='Preço (USD)',
            #titlefont=dict(color='gold'),
            tickfont=dict(color='gold')
        ),
        yaxis2=dict(
            title='Amplitude das Ondas',
            #titlefont=dict(color='cyan'),
            tickfont=dict(color='cyan'),
            overlaying='y',
            side='right'
        ),
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=60, r=60, t=60, b=40)
    )

    #fig.show()
    st.plotly_chart(fig, theme=None, use_container_width=True)

def plot_weekly_with_indicators(
    token_sel = 'BTC',
    s_close=None,
    s_date=None,
    df=None,
    compute_missing=False,
    rsi_period=14
):

    # -------- preparar dados de entrada --------
    if df is None:
        if s_close is None or s_date is None:
            raise ValueError("Forneça df OU (s_close e s_date).")
        df = pd.DataFrame({'Date': pd.to_datetime(s_date), 'Close': pd.to_numeric(s_close)})
    else:
        df = df.copy()
        # normalizar nomes mais comuns (case-insensitive)
        cols_lower = {c.lower(): c for c in df.columns}
        # garantir colunas padrão se existirem com variações
        if 'date' in cols_lower and 'Date' not in df.columns:
            df.rename(columns={cols_lower['date']: 'Date'}, inplace=True)
        if 'close' in cols_lower and 'Close' not in df.columns:
            df.rename(columns={cols_lower['close']: 'Close'}, inplace=True)

    # ordenar por data e garantir datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)
    else:
        # se não tiver Date, tenta índice datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("O DataFrame precisa ter coluna 'Date' ou índice DateTime.")
        df = df.sort_index()
        df['Date'] = df.index

    # -------- calcular indicadores ausentes (opcional) --------
    if compute_missing:
        if 'EMA50' not in df.columns and 'Close' in df.columns:
            df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        if 'EMA200' not in df.columns and 'Close' in df.columns:
            df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
        if 'RSI' not in df.columns and 'Close' in df.columns:
            # RSI (Wilder)
            delta = df['Close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
            rs = avg_gain / (avg_loss.replace(0, np.nan))
            df['RSI'] = 100 - (100 / (1 + rs))

    # -------- criar figura com eixo secundário --------
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Close (primário)
    if 'Close' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['Close'],
                mode='lines',
                name='BTC Close',
                line=dict(width=2)
            ),
            secondary_y=False
        )

    # EMAs (primário, se existirem)
    if 'EMA50' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['EMA50'],
                mode='lines',
                name='EMA50',
                line=dict(width=1.5, dash='dot')
            ),
            secondary_y=False
        )

    if 'EMA200' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['EMA200'],
                mode='lines',
                name='EMA200',
                line=dict(width=1.5, dash='dash')
            ),
            secondary_y=False
        )

    # RSI (secundário)
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['RSI'],
                mode='lines',
                name='RSI',
                line=dict(width=2)
            ),
            secondary_y=True
        )
        # linhas guia 30/50/70 no eixo RSI
        for lvl, dash in [(30, 'dot'), (50, 'dash'), (70, 'dot')]:
            fig.add_hline(
                y=lvl,
                line=dict(width=1, dash=dash),
                secondary_y=True
            )

    # -------- Fibs/Extensões (podem ser séries ou valores constantes) --------
    fib_cols = ['Fib_38.2%', 'Fib_50%', 'Fib_61.8%', 'Ext_61.8%', 'Ext_100%', 'Ext_161.8%']
    for col in fib_cols:
        if col in df.columns:
            s = df[col]
            # se for constante (ou quase), usa hline; senão, plota como linha
            if pd.api.types.is_numeric_dtype(s):
                if s.nunique(dropna=True) <= 1:
                    level = float(s.dropna().iloc[-1]) if s.dropna().shape[0] else None
                    if level is not None:
                        fig.add_hline(
                            y=level,
                            line=dict(width=1),
                            annotation_text=col,
                            annotation_position='top left',
                            secondary_y=False
                        )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=df['Date'],
                            y=s,
                            mode='lines',
                            name=col,
                            line=dict(width=1)
                        ),
                        secondary_y=False
                    )

    # -------- layout --------
    fig.update_layout(
        title=f'{token_sel} Semanal: Close, EMAs, RSI e Fibs/Extensões',
        template='plotly_dark',
        hovermode='x unified',
        margin=dict(l=60, r=60, t=60, b=40),
        legend=dict(x=0.01, y=0.99)
    )

    fig.update_xaxes(title_text='Data')
    fig.update_yaxes(
        title_text='Preço (USD)',
        secondary_y=False
    )
    fig.update_yaxes(
        title_text='RSI',
        range=[0, 100],
        secondary_y=True
    )

    #fig.show()
    st.plotly_chart(fig, theme=None, use_container_width=True)

def _parse_txt_fast(text: str):
    if not text:
        return {}, pd.DataFrame(columns=["Ativo", "Qt", "Data"])

    # Remove comentários e vazios
    txtraw = "\n".join(
        ln for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    )

    # Separa ativos das %
    parts = txtraw.split("@", 1)
    alvo_txt = parts[0].strip()
    trade_txt = parts[1].strip() if len(parts) > 1 else ""

    # --- Alvos (regex multiline, sem loop) ---
    alvo_pairs = re.findall(r"(?mi)^\s*([A-Za-z0-9_\-]+)\s+([0-9]*\.?[0-9]+)\s*$", alvo_txt)
    df_targets = pd.DataFrame(alvo_pairs, columns=["Ativo", "Pct"])
    df_targets["Pct"] = pd.to_numeric(df_targets["Pct"], errors="coerce")
    df_targets["Ativo"] = df_targets["Ativo"].str.upper()
    df_targets = df_targets.dropna(subset=["Pct"]).groupby("Ativo", as_index=False)["Pct"].sum()

    targets = dict(zip(df_targets["Ativo"], df_targets["Pct"]))
    soma = float(df_targets["Pct"].sum())
    if soma < 1 - 1e-9:
        targets["USD"] = round(1 - soma, 12)
    elif soma > 1 + 1e-9:
        st.warning(f"Soma dos alvos = {soma:.4f} (>1). Ajuste seu TXT.")

    # --- Trades: usa read_csv com sep regex (sem for) ---
    if trade_txt:
        df_trades = pd.read_csv(
            StringIO(trade_txt),
            sep=r"[,\s]+",
            engine="python",
            header=None,
            names=["Ativo", "Qt", "Data"],
            usecols=[0, 1, 2]
        )
        df_trades["Ativo"] = df_trades["Ativo"].str.upper()
        df_trades["Qt"] = pd.to_numeric(df_trades["Qt"], errors="coerce")
        df_trades["Data"] = pd.to_datetime(df_trades["Data"], dayfirst=True, errors="coerce")
        df_trades = df_trades.dropna(subset=["Ativo", "Qt"])
    else:
        df_trades = pd.DataFrame(columns=["Ativo", "Qt", "Data"])

    return targets, df_trades


def _prices_from_crypto1d_fast(crypto_1d: dict, symbols: pd.Index) -> pd.Series:
    """
    Retorna Series PriceUSD indexada pelos símbolos (USD=1.0).
    Sem loops explícitos: usa list comprehension + Series.
    """
    syms = pd.Index(symbols).unique()
    prices = [
        1.0 if s == "USD"
        else (
            float(crypto_1d[s]["Close"].iloc[-1])
            if (s in crypto_1d and "Close" in crypto_1d[s].columns and len(crypto_1d[s]["Close"]) > 0)
            else np.nan
        )
        for s in syms
    ]
    return pd.Series(prices, index=syms, name="PriceUSD")


def _rebalance_fast(targets: dict, trades: pd.DataFrame, prices: pd.Series):
    # quantidades atuais por símbolo (groupby vetorizado)
    qty = trades.groupby("Ativo", as_index=True)["Qt"].sum() if not trades.empty else pd.Series(dtype=float)

    # garante presença de todos os alvos
    idx_all = pd.Index(sorted(set(qty.index) | set(targets.keys())))
    qty = qty.reindex(idx_all, fill_value=0.0)

    price = prices.reindex(idx_all)
    val = qty * price
    total = float(val.replace([np.inf, -np.inf], np.nan).sum(skipna=True))

    cur_pct = (val / total).fillna(0.0) if total > 0 else pd.Series(0.0, index=idx_all)
    tgt_pct = pd.Series(targets, dtype=float).reindex(idx_all, fill_value=0.0)
    tgt_val = tgt_pct * total

    diff_usd = tgt_val - val
    delta_units = diff_usd / price.replace(0, np.nan)

    out = pd.DataFrame({
        "Ativo": idx_all,
        "Qt": qty.values,
        "PriceUSD": price.values,
        "ValueUSD": val.values,
        "% Atual": cur_pct.values,
        "% Alvo": tgt_pct.values,
        "DiffUSD": diff_usd.values,
        "DeltaUnits": delta_units.values
    })
    return out, total

def exibir_rebalanceamento(crypto_1d, up, txt_manual):

    content = up.read().decode("utf-8", errors="ignore") if up else txt_manual.strip()
    if not content:
        st.info("Envie um arquivo ou cole o conteúdo.")

    targets, trades = _parse_txt_fast(content)

    st.subheader("🧾 Aquisições")
    st.dataframe(trades.sort_values("Data"), use_container_width=True, hide_index=True)

    syms = pd.Index(
        sorted(set(list(targets.keys()) + trades["Ativo"].unique().tolist() if not trades.empty else [])))
    prices = _prices_from_crypto1d_fast(crypto_1d, syms)

    out, total = _rebalance_fast(targets, trades, prices)

    st.subheader(f"💰 Valor total: **${total:,.2f}**")

    show = out.copy()
    show["% Atual"] = (show["% Atual"] * 100).round(2)
    show["% Alvo"] = (show["% Alvo"] * 100).round(2)
    show["PriceUSD"] = pd.to_numeric(show["PriceUSD"], errors="coerce").round(8)
    show["ValueUSD"] = pd.to_numeric(show["ValueUSD"], errors="coerce").round(2)
    show["DiffUSD"] = pd.to_numeric(show["DiffUSD"], errors="coerce").round(2)
    show["DeltaUnits"] = pd.to_numeric(show["DeltaUnits"], errors="coerce").round(10)

    st.subheader("Plano de Rebalanceamento")
    st.dataframe(
        show.rename(columns={
            "Qt": "Qtd atual", "PriceUSD": "Preço (USD)",
            "ValueUSD": "Valor (USD)",
            "DiffUSD": "Ajuste USD", "DeltaUnits": "Δ Unidades"
        }),
        use_container_width=True, hide_index=True
    )

    st.subheader("Ações sugeridas (exclui USD)")
    if not out.empty:
        mask_crypto = (out["Ativo"] != "USD") & out["DiffUSD"].notna()
        buys = out.loc[mask_crypto & (out["DiffUSD"] > 0), ["Ativo", "DiffUSD", "DeltaUnits"]]
        sells = out.loc[mask_crypto & (out["DiffUSD"] < 0), ["Ativo", "DiffUSD", "DeltaUnits"]]
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Comprar**")
            st.dataframe(
                buys.assign(DiffUSD=lambda d: d["DiffUSD"].round(2),
                            DeltaUnits=lambda d: d["DeltaUnits"].round(10)),
                use_container_width=True, hide_index=True
            )
        with col2:
            st.write("**Vender**")
            st.dataframe(
                sells.assign(DiffUSD=lambda d: d["DiffUSD"].round(2),
                             DeltaUnits=lambda d: d["DeltaUnits"].round(10)),
                use_container_width=True, hide_index=True
            )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            fig = px.pie(
                show,
                values="% Atual",
                names="Ativo",
                title="Distribuição Atual",
                color_discrete_sequence=px.colors.sequential.Tealgrn
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            df_targets = pd.DataFrame([targets]).T.reset_index()
            df_targets.columns = ["Ativo", "% Alvo"]
            fig = px.pie(
                df_targets,
                values="% Alvo",
                names="Ativo",
                title="Distribuição da Carteira Planejada",
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

def mostrar_fear_and_greed():
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=1")
        data = resp.json()
        valor = data['data'][0]['value']
        classificacao = data['data'][0]['value_classification']
        return valor, classificacao
    except Exception as e:
        st.error(f"Erro ao obter o indicador Fear & Greed: {e}")
        return 0, 0

def print_links_uteis():
    st.markdown("### 🔗 Links Úteis — Análise de Mercado")

    st.markdown("""
        **📊 Indicadores Gerais**
        - [**TVL (Total Value Locked)** — DefiLlama](https://defillama.com/)
        - [**Market Cap & Dados** — CoinGecko](https://www.coingecko.com/pt)
    
        **🌐 Situação Macro da Criptoeconomia**
        - [**Dashboard Macro Cripto** — SoSoValue](https://sosovalue.com/pt/dashboard/charts)
    
        **⚠️ Indicadores de Alerta**
        - [**AHR999 Indicator** — linha verde = zona de oportunidade](https://sosovalue.com/pt/dashboard/ahr-999-indicator)
        - [**ETF Fund Flow (1W)** — atenção a saídas 2 semanas seguidas](https://sosovalue.com/pt/dashboard/total-crypto-spot-etf-fund-flow-1w)
        """)


def print_volume_review(token_sel):
    if token_sel in lista_crypto_100:
        st.warning(
            f"⚠️ {token_sel}: Altcoin de **volume muito baixo**, geralmente pode estar relacionado "
            "ao movimento de ativos com **alto volume**. Requer atenção redobrada."
        )
    elif token_sel in lista_crypto_50:
        st.info(
            f"ℹ️ {token_sel}: Altcoin de **volume relativamente baixo**, pode estar correlacionado "
            "com movimentos de ativos de **alto volume**."
        )
    elif token_sel not in ['BTC', 'USD', 'USDT', 'XAUT']:
        st.info(
            f"ℹ️ {token_sel}: Altcoin de alto volume, pode estar correlacionado "
            "com movimentos do BTC, que possui **volume maior**."
        )

    if token_sel in ['USD', 'USDT', 'XAUT']:
        st.info(
            f"ℹ️ {token_sel}: Stablecoin."
        )


################# CODE #################

st.header('Cryptinho',anchor=False)

with st.sidebar:

    st.text('Para o rebalanceador de carteira:')

    up_ = st.file_uploader("Envie .txt da sua carteira", type=["txt"])
    txt_manual_ = st.text_area(
        "…ou cole aqui",
        height=150,
        placeholder="BTC 0.5\nETH 0.25\n@\nBTC, 0.1, 03/11/2025\nBTC, 0.05, 01/11/2025\nETH, 3, 01/11/2025\nUSD, 1000, 01/11/2025"
    )

    exemplo_txt = """
    # Fazer carteira com, nas primeiras linhas o ativo com a % desejada para ele (o resto será considerado caixa em USD), termine a seção com um '@'. ex.: BTC 0.5
    # Nas linhas seguintes, o ativo, as unidades adquiridas de um ativo e a data de aquisição. ex.: BTC, 0.005, 01/11/2025 

    BTC 0.5
    ETH 0.25
    @
    BTC, 0.1, 03/11/2025
    BTC, 0.05, 01/11/2025
    ETH, 3, 03/11/2025
    USD, 1000, 01/11/2025
    """

    st.download_button(
        "Baixar exemplo de carteira (.txt)",
        data=exemplo_txt,
        file_name="carteira_exemplo.txt",
        mime="text/plain"
    )

    st.divider()

    st.text('Para os indicadores:')

    DATA_MAXIMA = None

    DATA_MAXIMA_CHECK = st.checkbox('Usar última data', value=True, help='Se desmarcado, selecionar última data')

    if DATA_MAXIMA_CHECK:
        DATA_MAXIMA = None
    else:
        DATA_MAXIMA = st.date_input("Selecione última data", value=pd.Timestamp.today().date(), format="DD/MM/YYYY")

    periodo = "36mo"

    periodicidade = st.radio(
        "Periodicidade",
        options=["1D", "1W"],
        index=1,
        horizontal=True,
        help="Escolha entre dados diários (1D) ou semanais (1W)"
    )

    try:
        token_sel = st.selectbox("Selecione o token", options=lista_crypto, index=0)
    except:
        token_sel = st.selectbox("Selecione o token", options=['BTC'], index=0)




for key in ["crypto_1d", "crypto_1w", "crypto_1m", "DATA_MAXIMA_CACHE"]:
    if key not in st.session_state:
        st.session_state[key] = np.nan

if st.session_state["DATA_MAXIMA_CACHE"] != DATA_MAXIMA:
    with st.spinner('Carregando...'):

        # Obtém dados diários, semanais e mensais corretamente
        crypto_1d = fetch_crypto_data(lista_crypto, periodo, "1d", DATA_MAXIMA)  # Diário
        crypto_1w = fetch_crypto_data(lista_crypto, "36mo", "1s", DATA_MAXIMA)  # Semanal (pulando 7 dias)
        crypto_1m = fetch_crypto_data(lista_crypto, "36mo", "1m", DATA_MAXIMA)  # Mensal (pegando o último dia de cada mês)

        # Fear and Greed
        valor_fg, classificacao_fg = mostrar_fear_and_greed()

        st.session_state["crypto_1d"] = crypto_1d
        st.session_state["crypto_1w"] = crypto_1w
        st.session_state["crypto_1m"] = crypto_1m
        st.session_state["valor_fg"] = valor_fg
        st.session_state["classificacao_fg"] = classificacao_fg
        st.session_state["DATA_MAXIMA_CACHE"] = DATA_MAXIMA
else:
    crypto_1d = st.session_state["crypto_1d"]
    crypto_1w = st.session_state["crypto_1w"]
    crypto_1m = st.session_state["crypto_1m"]
    valor_fg = st.session_state["valor_fg"]
    classificacao_fg = st.session_state["classificacao_fg"]

with st.spinner('Processando...'):

    # cria uma Series com o último volume de cada símbolo
    last_vol = pd.Series({sym: df['Volume'].iloc[-1] for sym, df in crypto_1w.items()})

    # aplica as faixas diretamente (vetorizado)
    lista_crypto_25  = last_vol.index[last_vol > 4e8].tolist()
    lista_crypto_50  = last_vol.index[(last_vol > 6e7) & (last_vol <= 4e8)].tolist()
    lista_crypto_100 = last_vol.index[last_vol <= 6e7].tolist()

    (sinais_compra_geral, sinais_venda_geral,
                sinais_compra_25, sinais_venda_25,
                sinais_compra_50, sinais_venda_50,
                sinais_compra_100, sinais_venda_100,
                df_buy_sell_ratio, df_agregado_compra_venda,
                variacoes_preco_1S_str) = analisar_sinais_cripto(crypto_1d, "1d", lista_crypto,
                               crypto_1d, crypto_1w, crypto_1m,
                               lista_crypto_25, lista_crypto_50, lista_crypto_100)

    (sinais_compra_geral_1w, sinais_venda_geral_1w,
                sinais_compra_25_1w, sinais_venda_25_1w,
                sinais_compra_50_1w, sinais_venda_50_1w,
                sinais_compra_100_1w, sinais_venda_100_1w,
                df_buy_sell_ratio_1w, df_agregado_compra_venda_1w,
                variacoes_preco_1S_str_1w) = analisar_sinais_cripto(crypto_1w, "1w", lista_crypto,
                               crypto_1d, crypto_1w, crypto_1m,
                               lista_crypto_25, lista_crypto_50, lista_crypto_100)


    tab_carteira, tab_indicadores = st.tabs(
        ["Carteira","Indicadores"]
    )

    with tab_indicadores:

        col1, col2 = st.columns([2, 1])

        if periodicidade == '1D':
            with col1:
                st.subheader(f"📈 Gráficos {periodicidade}")
                plot_weekly_with_indicators(
                    token_sel=token_sel,
                    s_close=crypto_1d[token_sel]['Close'],
                    s_date=crypto_1d[token_sel]['Date'],
                    compute_missing = True  # calcula EMA50, EMA200 e RSI(14)
                )
                plot_with_semicircles(crypto_1w[token_sel]['Close'], crypto_1w[token_sel]['Date'], token_sel=token_sel)
            with col2:
                exibir_sinais_streamlit(
                    sinais_compra_25, sinais_venda_25,
                    sinais_compra_50, sinais_venda_50,
                    sinais_compra_100, sinais_venda_100,
                    variacoes_preco_1S_str,
                    sinais_compra_geral, sinais_venda_geral,
                    df_buy_sell_ratio, df_agregado_compra_venda, token_sel, periodicidade
                )

                print_volume_review(token_sel)

                st.subheader("⚡ Fear & Greed Index")
                st.write(f"**Valor:** {valor_fg}")
                st.write(f"**Classificação:** {classificacao_fg}")

                print_links_uteis()

        elif periodicidade == '1W':
            with col1:
                st.subheader(f"📈 Gráficos {periodicidade}")
                plot_weekly_with_indicators(
                    token_sel=token_sel,
                    s_close=crypto_1w[token_sel]['Close'],
                    s_date=crypto_1w[token_sel]['Date'],
                    compute_missing = True  # calcula EMA50, EMA200 e RSI(14)
                )
                plot_with_semicircles(crypto_1w[token_sel]['Close'], crypto_1w[token_sel]['Date'], token_sel=token_sel)
            with col2:
                exibir_sinais_streamlit(
                    sinais_compra_25_1w, sinais_venda_25_1w,
                    sinais_compra_50_1w, sinais_venda_50_1w,
                    sinais_compra_100_1w, sinais_venda_100_1w,
                    variacoes_preco_1S_str_1w,
                    sinais_compra_geral_1w, sinais_venda_geral_1w,
                    df_buy_sell_ratio_1w, df_agregado_compra_venda_1w, token_sel, periodicidade
                )

                print_volume_review(token_sel)

                st.subheader("⚡ Fear & Greed Index")
                st.write(f"**Valor:** {valor_fg}")
                st.write(f"**Classificação:** {classificacao_fg}")

                print_links_uteis()


    ###

    with tab_carteira:

        st.header("🧮 Rebalanceador de Carteira")
        exibir_rebalanceamento(crypto_1d, up_, txt_manual_)






