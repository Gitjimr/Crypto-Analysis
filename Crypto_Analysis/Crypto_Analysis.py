import functools
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from io import StringIO
from typing import Callable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import fontes                  # de onde vêm os preços (Yahoo no PC; retrato + Binance no navegador)
import linhas as ln
import ml
from ativos import ACOES_B3, ACOES_EUA, MERCADOS, PASTA
from ativos import ler_simbolos as _ler_simbolos

st.set_page_config(page_title="CRYPTINHO", layout="wide")

# Mantém valores de widgets entre páginas (hack conhecido de apps multipágina).
# Botões de formulário são somente leitura: regravá-los quebra o st.form.
for _k in list(st.session_state.keys()):
    if not str(_k).startswith("FormSubmitter:"):
        st.session_state[_k] = st.session_state[_k]

ANOS_PADRAO = 30     # por padrão, só os últimos 30 anos (ações dos EUA têm dados desde os anos 60–80: fica pesado)
MESES_GRAFICO = 36   # período exibido por padrão nos gráficos (o mesmo do app original)
TTL = "30m"          # tempo de vida do cache de dados de mercado


def _copia(obj):
    """Cópia independente (sem pickle) de DataFrames/Series/arrays dentro de dicts, listas e tuplas."""
    if isinstance(obj, (pd.DataFrame, pd.Series, np.ndarray)):
        return obj.copy()
    if isinstance(obj, dict):
        return {k: _copia(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)) and not hasattr(obj, "_fields"):
        return type(obj)(_copia(v) for v in obj)
    return obj


def cache_dados(**opcoes):
    """``st.cache_data`` no PC. No navegador (Pyodide, 32 bits) o pandas às vezes falha ao reler do pickle
    colunas de data — e o cache_data guarda tudo em pickle. Lá usamos ``st.cache_resource`` (guarda o objeto
    em memória, sem pickle) e devolvemos uma cópia, para ninguém alterar o que está no cache."""
    if not fontes.NO_NAVEGADOR:
        return st.cache_data(**opcoes)

    def decorar(func):
        guardada = st.cache_resource(**opcoes)(func)

        @functools.wraps(func)
        def com_copia(*args, **kwargs):
            return _copia(guardada(*args, **kwargs))
        com_copia.clear = guardada.clear
        return com_copia
    return decorar


# Leitura da lista de símbolos

@cache_dados(show_spinner=False)
def ler_simbolos(caminho: str = "add_symbols.txt") -> tuple[str, ...]:
    return _ler_simbolos(caminho)


# =========================================================================== #
# Indicadores
# =========================================================================== #

def calcular_adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14):
    up, down = high.diff(), -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)     # compara com os valores ORIGINAIS
    atr = ln.atr_wilder(high, low, close, n)
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), plus_di, minus_di


def adicionar_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    """Todos os indicadores de uma vez (vetorizado, um único concat)."""
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    n = len(df)
    x = {}
    x["RSI"] = ln.rsi_wilder(c)                       # Wilder = igual ao TradingView
    for span in (12, 26, 50, 111, 200, 350):
        x[f"EMA{span}"] = c.ewm(span=span, adjust=False).mean()
    x["EMA350x2"] = 2 * x["EMA350"]
    macd = x["EMA12"] - x["EMA26"]
    sinal = macd.ewm(span=9, adjust=False).mean()
    x.update(MACD=macd, MACD_Signal=sinal, MACD_Hist=macd - sinal)

    x["Price_Change"] = pc = c.diff()
    x["OBV"] = obv = (np.sign(pc).fillna(0) * v).cumsum()
    x["OBV_SMA20"] = obv.rolling(20).mean()

    sma, std = c.rolling(20).mean(), c.rolling(20).std()
    largura = 4 * std
    x.update(SMA20=sma, STD20=std, Upper_BB=sma + 2 * std, Lower_BB=sma - 2 * std,
             BB_Width=largura, BB_Contraction=largura.diff())
    x["BB_Contraction_Status"] = np.where(largura.diff() < 0, "Contraindo", "Expandindo")

    hh, ll = h.rolling(14).max(), l.rolling(14).min()
    x["Williams_%R"] = -100 * (hh - c) / (hh - ll + 1e-10)
    r = x["RSI"]
    rmin, rmax = r.rolling(14).min(), r.rolling(14).max()
    x["StochRSI"] = (r - rmin) / (rmax - rmin + 1e-10)
    x["VolumeSpike"] = v > 2 * v.rolling(20).mean()
    x["ADX"], x["DI+"], x["DI-"] = calcular_adx(h, l, c)
    x["ATR"] = ln.atr_wilder(h, l, c)

    # Divergências por pivôs (sinal ativo por 3 barras após a confirmação)
    div = ln.divergencias(c.to_numpy(float), l.to_numpy(float), h.to_numpy(float), r.to_numpy(float))
    for tipo, col in (("Bull", "BullishDivergence"), ("Bear", "BearishDivergence")):
        arr = np.zeros(n, dtype=bool)
        for conf in div.loc[div["tipo"] == tipo, "confirmacao"]:
            arr[conf: conf + 3] = True
        x[col] = arr

    # Fibonacci do último ano (constantes; guardadas como colunas por compatibilidade)
    ano = df.loc[df.index > df.index.max() - pd.DateOffset(years=1)]
    hi1, lo1 = float(ano["High"].max()), float(ano["Low"].min())
    amp = hi1 - lo1
    for k, f in {"23.6%": .236, "38.2%": .382, "50%": .5, "61.8%": .618, "78.6%": .786}.items():
        x[f"Fib_{k}"] = hi1 - amp * f
    for k, f in {"61.8%": .618, "100%": 1.0, "161.8%": 1.618, "261.8%": 2.618, "423.6%": 4.236}.items():
        x[f"Ext_{k}"] = hi1 + amp * f

    return pd.concat([df, pd.DataFrame(x, index=df.index)], axis=1)


# Dados

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


@cache_dados(ttl=TTL, show_spinner=False)
def baixar_ohlc(simbolos: tuple[str, ...], periodo: str, mercado: str = "Cripto") -> dict[str, pd.DataFrame]:
    """Todos os ativos do mercado de uma vez (ver fontes.py)."""
    return fontes.ohlc(simbolos, periodo, mercado)


def reamostrar(df: pd.DataFrame, regra: str) -> pd.DataFrame:
    """OHLC correto (abertura/máx/mín/fech/volume somado). Semana começa na segunda,
    como no TradingView; o último candle é a semana/mês em andamento."""
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    kw = {"label": "left", "closed": "left"} if regra.startswith("W") else {}
    return df.resample(regra, **kw).agg(agg).dropna(subset=["Close"])


@st.cache_resource(ttl=TTL, max_entries=6, show_spinner="Baixando e calculando indicadores...")
def preparar_dados(simbolos: tuple[str, ...], periodo: str, data_maxima: str | None,
                   mercado: str = "Cripto") -> dict:
    """{'1D'|'1W'|'1M': {ativo: DataFrame}}. Somente leitura (é compartilhado)."""
    brutos = baixar_ohlc(simbolos, periodo, mercado)
    corte = pd.Timestamp(data_maxima) if data_maxima else None

    def um_token(df):
        if corte is not None:
            df = df.loc[:corte]
        if len(df) < 60:
            return None
        return (adicionar_indicadores(df), adicionar_indicadores(reamostrar(df, "W-MON")),
                adicionar_indicadores(reamostrar(df, "MS")))

    if fontes.NO_NAVEGADOR:   # o Pyodide não tem threads: um ativo por vez
        resultados = {s: um_token(df) for s, df in brutos.items()}
    else:                    # tokens em paralelo (as rotinas numéricas do pandas/numpy liberam o GIL)
        with ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 2))) as ex:
            resultados = dict(zip(brutos, ex.map(um_token, brutos.values())))
    dados = {"1D": {}, "1W": {}, "1M": {}}
    for s, r in resultados.items():
        if r is not None:
            dados["1D"][s], dados["1W"][s], dados["1M"][s] = r
    return dados


def faixas_volume(d1: dict) -> tuple[list, list, list]:
    """Faixas de liquidez pela média de volume diário dos últimos 7 dias."""
    vol = pd.Series({s: df["Volume"].iloc[-7:].mean() for s, df in d1.items()}, dtype=float)
    return (vol.index[vol > 4e8].tolist(),
            vol.index[(vol > 6e7) & (vol <= 4e8)].tolist(),
            vol.index[vol <= 6e7].tolist())


# =========================================================================== #
# Sinais (tabela de regras, avaliadas para todos os tokens de uma vez)
# =========================================================================== #

f = ln.fmt


@dataclass(frozen=True)
class Regra:
    lado: str                                   # "Compra" | "Venda"
    icone: str
    cond: Callable[[pd.DataFrame], pd.Series]   # S (1 linha por token) -> bool
    msg: Callable[[pd.Series, str], str]        # (linha do token, período) -> texto
    conta: bool = True                          # False = indicador de suporte: aparece, mas NÃO entra no ratio


def Suporte(*args) -> Regra:
    """Regra só informativa. Testadas no BTC (2015–2026, 30 dias), estas ERRARAM mais que o acaso como sinal de
    virada — sobrecompra costuma continuar subindo e sobrevenda continuar caindo —, então ficam de fora da conta."""
    return Regra(*args, conta=False)


def _obv_msg(r, P, compra):
    if compra:
        return (f"({P}) OBV acima da média de 20 e preço subindo → alta confiável" if r.Price_Change > 0
                else f"({P}) OBV acima da média de 20 e preço caindo → possível suporte/recuperação")
    return (f"({P}) OBV abaixo da média de 20 e preço subindo → divergência, alerta de reversão" if r.Price_Change > 0
            else f"({P}) OBV abaixo da média de 20 e preço caindo → baixa confirmada")


REGRAS: list[Regra] = [
    # ------------------------------ COMPRA ------------------------------ #
    Regra("Compra", "🟢", lambda S: S.EMA200 < S.EMA50,
          lambda r, P: f"({P}) EMA200 ({f(r.EMA200)}) menor que EMA50 ({f(r.EMA50)}) → Bull"),
    Suporte("Compra", "🔽", lambda S: (S.EMA200 < S.EMA50) & (S.Close < S.EMA50) & (S.Close < S.EMA111)
          & (S.Close <= S.EMA200) & (S.Close >= S.EMA350),
          lambda r, P: f"({P}) Preço abaixo das EMAs 50, 111 e 200, mas acima do suporte da EMA350 ({f(r.EMA350)})"),
    Regra("Compra", "📉", lambda S: S.RSI_1D <= 30, lambda r, P: f"RSI 1D em sobrevenda ({r.RSI_1D:.2f})"),
    Regra("Compra", "📉", lambda S: S.RSI_1W <= 30, lambda r, P: f"RSI 1S em sobrevenda ({r.RSI_1W:.2f})"),
    Regra("Compra", "📉", lambda S: S.RSI_1M <= 30, lambda r, P: f"RSI 1M em sobrevenda ({r.RSI_1M:.2f})"),
    Regra("Compra", "📈", lambda S: S.MACD_Hist > 0, lambda r, P: f"({P}) MACD acima da linha de sinal → Bull"),
    Regra("Compra", "📊", lambda S: S.OBV > S.OBV_SMA20, lambda r, P: _obv_msg(r, P, True)),
    Suporte("Compra", "📉", lambda S: S.Close < S.Lower_BB,
          lambda r, P: f"({P}) Preço abaixo da Bollinger inferior ({f(r.Lower_BB)}) → "
                       + ("contraindo → movimento forte se aproximando" if r.BB_Contraction_Status == "Contraindo"
                          else "expandindo → alta volatilidade")),
    Suporte("Compra", "🔺", lambda S: (S["Fib_38.2%"] >= S.Close) & (S.Close >= S["Fib_61.8%"]),
          lambda r, P: f"({P}) Preço entre Fibo 38,2% ({f(r['Fib_38.2%'])}, compra arrojada) e "
                       f"61,8% ({f(r['Fib_61.8%'])}, golden pocket)"),
    Suporte("Compra", "🟢", lambda S: S["Williams_%R"] <= -80,
          lambda r, P: f"({P}) Williams %R ({r['Williams_%R']:.2f}) ≤ -80 (sobrevenda)"),
    Suporte("Compra", "🟢", lambda S: S.StochRSI <= 0.2,
          lambda r, P: f"({P}) StochRSI ({r.StochRSI:.2f}) ≤ 0,2 (sobrevenda)"),
    Regra("Compra", "🟢", lambda S: S.VolumeSpike.astype(bool) & (S.Close > S.Close_ant),
          lambda r, P: f"({P}) Pico de volume com preço subindo (acumulação)"),
    Suporte("Compra", "🟢", lambda S: (S.ADX > 25) & (S["DI+"] > S["DI-"]),
          lambda r, P: f"({P}) ADX ({r.ADX:.2f}) > 25 e DI+ > DI- (tendência de alta)"),
    Regra("Compra", "🟢", lambda S: S.BullishDivergence.astype(bool),
          lambda r, P: f"({P}) Divergência de alta RSI x preço"),
    Regra("Compra", "🟢", lambda S: (S["DI+"] > S["DI-"]) & (S["DI+_ant"] < S["DI-_ant"]),
          lambda r, P: f"({P}) DI+ cruzou acima do DI- (mudança de tendência)"),
    Suporte("Compra", "🟢", lambda S: (S["Williams_%R"] > -80) & (S["Williams_%R_ant"] <= -80),
          lambda r, P: f"({P}) Williams %R cruzando acima de -80 (saindo da sobrevenda)"),
    Suporte("Compra", "🟢", lambda S: (S.StochRSI > 0.2) & (S.StochRSI_ant <= 0.2),
          lambda r, P: f"({P}) StochRSI cruzando acima de 0,2 (momentum positivo)"),
    # ------------------------------ VENDA ------------------------------- #
    Regra("Venda", "🔴", lambda S: S.EMA200 > S.EMA50,
          lambda r, P: f"({P}) EMA200 ({f(r.EMA200)}) > EMA50 ({f(r.EMA50)}) → Bear"),
    Suporte("Venda", "🔴", lambda S: (S.Close > S.EMA50) & (S.Close > S.EMA111) & (S.Close > S.EMA200)
          & (S.Close > S.EMA350),
          lambda r, P: f"({P}) Preço acima das EMAs 50, 111, 200 e 350"),
    Regra("Venda", "🔴", lambda S: S.Close > S.EMA350x2,
          lambda r, P: f"({P}) Preço acima da EMA350×2 ({f(r.EMA350x2)}) → atenção ao cruzamento com EMA111 ({f(r.EMA111)})"),
    Regra("Venda", "⚠️", lambda S: S.EMA111 / S.EMA350x2 >= 0.9,
          lambda r, P: f"({P}) Pi Cycle Top em zona de alerta (EMA111 {f(r.EMA111)} x EMA350×2 {f(r.EMA350x2)})"),
    Suporte("Venda", "📈", lambda S: S.RSI_1D >= 70, lambda r, P: f"RSI 1D em sobrecompra ({r.RSI_1D:.2f})"),
    Suporte("Venda", "📈", lambda S: S.RSI_1W >= 70, lambda r, P: f"RSI 1S em sobrecompra ({r.RSI_1W:.2f})"),
    Suporte("Venda", "📈", lambda S: S.RSI_1M >= 70, lambda r, P: f"RSI 1M em sobrecompra ({r.RSI_1M:.2f})"),
    Regra("Venda", "🚨", lambda S: S.RSI_1M >= 90,
          lambda r, P: f"RSI 1M em sobrecompra EXTREMA ({r.RSI_1M:.2f} ≥ 90) — raro, topo de ciclo em 2017 e 2021"),
    Regra("Venda", "📉", lambda S: S.MACD_Hist < 0, lambda r, P: f"({P}) MACD abaixo da linha de sinal → Bear"),
    Regra("Venda", "🔻", lambda S: S.OBV < S.OBV_SMA20, lambda r, P: _obv_msg(r, P, False)),
    Suporte("Venda", "📈", lambda S: S.Close > S.Upper_BB,
          lambda r, P: f"({P}) Preço acima da Bollinger superior ({f(r.Upper_BB)}) → "
                       + ("contraindo → movimento forte se aproximando" if r.BB_Contraction_Status == "Contraindo"
                          else "expandindo → alta volatilidade")),
    Regra("Venda", "📉", lambda S: S["Fib_78.6%"] > S.Close,
          lambda r, P: f"({P}) Preço abaixo da retração de Fibo 78,6% ({f(r['Fib_78.6%'])})"),
    Regra("Venda", "📉", lambda S: (S["Ext_161.8%"] >= S.Close) & (S.Close >= S["Ext_100%"]),
          lambda r, P: f"({P}) Preço entre as extensões de Fibo 100% ({f(r['Ext_100%'])}) e 161,8% ({f(r['Ext_161.8%'])})"),
    Suporte("Venda", "🔴", lambda S: S["Williams_%R"] >= -20,
          lambda r, P: f"({P}) Williams %R ({r['Williams_%R']:.2f}) ≥ -20 (sobrecompra)"),
    Suporte("Venda", "🔴", lambda S: S.StochRSI >= 0.8,
          lambda r, P: f"({P}) StochRSI ({r.StochRSI:.2f}) ≥ 0,8 (sobrecompra)"),
    Suporte("Venda", "🔴", lambda S: S.VolumeSpike.astype(bool) & (S.Close < S.Close_ant),
          lambda r, P: f"({P}) Pico de volume com preço caindo (distribuição)"),
    Suporte("Venda", "🔴", lambda S: (S.ADX > 25) & (S["DI-"] > S["DI+"]),
          lambda r, P: f"({P}) ADX ({r.ADX:.2f}) > 25 e DI- > DI+ (tendência de baixa)"),
    Regra("Venda", "🔴", lambda S: (S["DI-"] > S["DI+"]) & (S["DI-_ant"] < S["DI+_ant"]),
          lambda r, P: f"({P}) DI- cruzou acima do DI+ (mudança de tendência)"),
    Suporte("Venda", "🔴", lambda S: (S["Williams_%R"] < -20) & (S["Williams_%R_ant"] >= -20),
          lambda r, P: f"({P}) Williams %R cruzando abaixo de -20 (saindo da sobrecompra)"),
    Suporte("Venda", "🔴", lambda S: (S.StochRSI < 0.8) & (S.StochRSI_ant >= 0.8),
          lambda r, P: f"({P}) StochRSI cruzando abaixo de 0,8 (momentum negativo)"),
    Regra("Venda", "🔴", lambda S: S.BearishDivergence.astype(bool),
          lambda r, P: f"({P}) Divergência de baixa RSI x preço"),
]
TOTAL = {lado: sum(r.lado == lado and r.conta for r in REGRAS) for lado in ("Compra", "Venda")}   # só as que contam


def _snapshot(dados: dict, freq: str) -> pd.DataFrame:
    """Uma linha por token com o último candle, o anterior (sufixo _ant) e RSI 1D/1W/1M."""
    linhas = {}
    for s, df in dados[freq].items():
        if len(df) < 3:
            continue
        r = df.iloc[-1].to_dict()
        ant = df.iloc[-2]
        for k in ("Close", "DI+", "DI-", "Williams_%R", "StochRSI"):
            r[f"{k}_ant"] = ant[k]
        for fr in ("1D", "1W", "1M"):
            r[f"RSI_{fr}"] = dados[fr][s]["RSI"].iloc[-1] if s in dados[fr] else np.nan
        linhas[s] = r
    return pd.DataFrame.from_dict(linhas, orient="index").infer_objects()


RATIO_COMPRA = 2.0        # Buy/Sell Ratio ≥ 2 = COMPRA, ≤ 1/2 = VENDA (ranking dos métodos)


def historico_ratio(d1: pd.DataFrame, d1w: pd.DataFrame | None, d1m: pd.DataFrame | None) -> pd.DataFrame:
    """As mesmas regras da aba Indicadores (diário) avaliadas em TODOS os dias do passado, só com o que se sabia
    no dia: RSI semanal/mensal do último candle já fechado e Fibonacci da máxima/mínima dos 365 dias anteriores."""
    S = d1.copy()
    for k in ("Close", "DI+", "DI-", "Williams_%R", "StochRSI"):
        S[f"{k}_ant"] = S[k].shift(1)
    S["RSI_1D"] = S["RSI"]
    for nome, d_, desloc in (("RSI_1W", d1w, pd.Timedelta(days=7)), ("RSI_1M", d1m, pd.DateOffset(months=1))):
        if d_ is None or d_.empty:
            S[nome] = np.nan
            continue
        r = d_["RSI"].copy()
        r.index = r.index + desloc                     # só vale depois que o candle fecha
        S[nome] = r.reindex(S.index.union(r.index)).ffill().reindex(S.index)
    hi1 = S["High"].rolling("365D", min_periods=30).max()
    lo1 = S["Low"].rolling("365D", min_periods=30).min()
    amp = hi1 - lo1
    for k, fr in {"23.6%": .236, "38.2%": .382, "50%": .5, "61.8%": .618, "78.6%": .786}.items():
        S[f"Fib_{k}"] = hi1 - amp * fr
    for k, fr in {"61.8%": .618, "100%": 1.0, "161.8%": 1.618, "261.8%": 2.618, "423.6%": 4.236}.items():
        S[f"Ext_{k}"] = hi1 + amp * fr
    cont = {lado: pd.Series(0, index=S.index) for lado in TOTAL}
    for regra in REGRAS:
        if regra.conta:
            cont[regra.lado] += regra.cond(S).fillna(False).astype(bool).astype(int)
    ratio = (cont["Compra"] + 1) / (cont["Venda"] + 1)
    sinal = np.where(ratio >= RATIO_COMPRA, 1, np.where(ratio <= 1 / RATIO_COMPRA, -1, 0))
    return pd.DataFrame({"compra": cont["Compra"], "venda": cont["Venda"], "ratio": ratio, "sinal": sinal},
                        index=S.index).iloc[200:]          # aquecimento das EMAs/ADX


@cache_dados(ttl=TTL, show_spinner=False)
def analisar_sinais(chave: tuple, freq: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna (sinais, resumo). ``chave`` = (simbolos, periodo, data_maxima)."""
    dados = preparar_dados(*chave)
    S = _snapshot(dados, freq)
    P = freq.lower()
    registros = []
    cont = {lado: pd.Series(0, index=S.index) for lado in TOTAL}
    for regra in REGRAS:
        mask = regra.cond(S).fillna(False).astype(bool)
        if regra.conta:
            cont[regra.lado] += mask.astype(int)
        for tok, r in S[mask].iterrows():                       # só os tokens que dispararam
            registros.append((tok, regra.lado, regra.icone, regra.msg(r, P), freq, regra.conta))
    sinais = pd.DataFrame(registros, columns=["Token", "Lado", " ", "Sinal", "Período", "Conta"])

    # Variação de 7 dias sempre pelo diário (iloc[-8] = 7 candles atrás)
    d1 = dados["1D"]
    atual = pd.Series({s: d1[s]["Close"].iloc[-1] for s in S.index if s in d1})
    antes = pd.Series({s: d1[s]["Close"].asof(d1[s].index[-1] - pd.Timedelta(days=7))
                       for s in S.index if s in d1 and len(d1[s]) >= 8})
    resumo = pd.DataFrame({
        "Token": S.index,
        "Compra": [f"{c}/{TOTAL['Compra']}" for c in cont["Compra"]],
        "Venda": [f"{v}/{TOTAL['Venda']}" for v in cont["Venda"]],
        "Buy/Sell Ratio": ((cont["Compra"] + 1) / (cont["Venda"] + 1)).round(3).to_numpy(),
        "Preço atual (1D)": atual.reindex(S.index).to_numpy(),
        "Preço 7D atrás (1D)": antes.reindex(S.index).to_numpy(),
    })
    resumo["Variação (7D) %"] = (100 * (resumo["Preço atual (1D)"] / resumo["Preço 7D atrás (1D)"] - 1)).round(2)
    return sinais, resumo


def texto_telegram(sinais: pd.DataFrame, tokens: list | None = None) -> str:
    """Mesmo formato HTML de antes, caso você ainda envie para o Telegram."""
    s = sinais if tokens is None else sinais[sinais["Token"].isin(tokens)]
    if "Conta" in s:
        s = s[s["Conta"]]                       # só os que contam no ratio
    partes = []
    for lado in ("Compra", "Venda"):
        bloco = s[s["Lado"] == lado]
        if bloco.empty:
            continue
        per = bloco["Período"].iloc[0]
        partes.append(f"<b>* SINAIS DE {lado.upper()} ({per}): *</b>\n")
        partes += [f"{r[' ']} <b><u>{r.Token}</u></b> {r.Sinal}" for _, r in bloco.iterrows()]
        partes.append("")
    return "\n".join(partes)


# UI: sinais

def _destacar(token_sel: str, coluna: str = "Token"):
    def _f(row):
        return ["background-color: #ff4d4d55; font-weight: bold;" if row[coluna] == token_sel else ""] * len(row)
    return _f


def exibir_sinais(sinais: pd.DataFrame, resumo: pd.DataFrame, token_sel: str, freq: str):
    st.subheader(f"📈 Indicadores {freq}")
    t1, t2, t3, t4 = st.tabs([f"Sinais {token_sel}", "Variação 7D", "Buy/Sell Ratio", "Compra × Venda"])
    with t1:
        s = sinais[sinais["Token"] == token_sel].sort_values(["Lado", " "])
        st.caption(f"Sinais que **contam** no Buy/Sell Ratio ({TOTAL['Compra']} de compra e {TOTAL['Venda']} de venda "
                   "possíveis) — os que, no histórico do BTC, acertaram mais que o acaso.")
        st.dataframe(s[s["Conta"]].drop(columns="Conta"), hide_index=True)
        sup = s[~s["Conta"]].drop(columns="Conta")
        st.caption("**Indicadores de suporte** — aparecem só como informação e **não contam no ratio**. São leituras de "
                   "sobrecompra/sobrevenda (StochRSI, Williams %R, Bollinger, RSI ≥ 70, zona de Fibo 38–62%...) que, "
                   "testadas no BTC (2015–2026), erraram mais que o acaso como sinal de virada: no cripto, sobrecomprado "
                   "costuma continuar subindo e sobrevendido continuar caindo.")
        if sup.empty:
            st.caption("(nenhum indicador de suporte ativo agora)")
        else:
            st.dataframe(sup, hide_index=True)
    with t2:
        st.caption("Variação de preço em 7 dias (diário)")
        v = resumo[["Token", "Preço atual (1D)", "Preço 7D atrás (1D)", "Variação (7D) %"]]
        st.dataframe(v.sort_values("Variação (7D) %", ascending=False).style.apply(_destacar(token_sel), axis=1),
                     hide_index=True)
    with t3:
        st.caption(f"Buy/Sell Ratio = (sinais de compra + 1) ÷ (sinais de venda + 1), só com os que contam. Quanto "
                   f"maior, mais 'atrativo'. No ranking dos métodos: ≥ {RATIO_COMPRA:g} = COMPRA, ≤ 1/{RATIO_COMPRA:g} = VENDA.")
        r = resumo[["Token", "Buy/Sell Ratio"]].sort_values("Buy/Sell Ratio", ascending=False)
        st.dataframe(r.style.apply(_destacar(token_sel), axis=1), hide_index=True)
    with t4:
        st.caption("Contadores por token (só os sinais que contam no ratio)")
        st.dataframe(resumo[["Token", "Compra", "Venda"]].style.apply(_destacar(token_sel), axis=1),
                     hide_index=True)


# =========================================================================== #
# UI: gráficos
# =========================================================================== #

def preco_atual(d1: pd.DataFrame | None, token: str, moeda: str = "USD"):
    """Linha discreta acima dos gráficos de preço: último preço, variação do dia e a data do candle."""
    if d1 is None or d1.empty:
        return
    c = d1["Close"].dropna()
    if c.empty:
        return
    ult = float(c.iloc[-1])
    var = (ult / float(c.iloc[-2]) - 1) * 100 if len(c) > 1 and c.iloc[-2] else np.nan
    cor = "green" if var > 0 else ("red" if var < 0 else "gray")
    seta = "▲" if var > 0 else ("▼" if var < 0 else "•")
    dia = "" if np.isnan(var) else f" · :{cor}[{seta} {var:+.2f}% no dia]"
    st.caption(f"**{token}** · último preço **{moeda} {ln.fmt(ult)}**{dia} · candle de {c.index[-1]:%d/%m/%Y}")


def grafico_indicadores(df: pd.DataFrame, token: str, freq: str, moeda: str = "USD"):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name=f"{token} Close", line=dict(width=2)),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA50", line=dict(width=1.5, dash="dot")),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA200"], name="EMA200", line=dict(width=1.5, dash="dash")),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(width=1.5)), secondary_y=True)
    for lvl, dash in ((30, "dot"), (50, "dash"), (70, "dot")):
        fig.add_hline(y=lvl, line=dict(width=1, dash=dash), secondary_y=True)
    fig.update_layout(title=f"{token} {freq}: Close, EMAs e RSI", template="plotly_dark",
                      hovermode="x unified", margin=dict(l=60, r=60, t=60, b=40), legend=dict(x=0.01, y=0.99))
    fig.update_yaxes(title_text=f"Preço ({moeda})", secondary_y=False)
    fig.update_yaxes(title_text="RSI", range=[0, 100], secondary_y=True)
    st.plotly_chart(fig, theme=None)


# --------------------------------------------------------------------------- #
# Ondas (semiciclos): parâmetros no ondas.txt e ajuste fino no navegador
# --------------------------------------------------------------------------- #

ARQ_ONDAS = "ondas.txt"
ONDAS_PADRAO = {"inicio": "2025-03-02", "largura_semanas": 28.0}
MODELO_ONDAS_TXT = """# Ondas do gráfico "Close + Ondas Simétricas" (aba Indicadores)
# Edite os valores abaixo e recarregue o app. Linhas com # são comentários.
#
# inicio          = data (AAAA-MM-DD) em que UMA das ondas começa; as outras se repetem
#                   para trás e para frente a partir dela
# largura_semanas = duração de cada onda, do vale ao vale seguinte (aceita decimal, ex.: 27.5)
#
# Dica: ajuste com os sliders abaixo do gráfico e copie para cá os valores mostrados.

inicio = {inicio}
largura_semanas = {largura_semanas:g}
"""


def ler_ondas(caminho: str = ARQ_ONDAS) -> tuple[dict, str | None]:
    """Lê o ondas.txt (cria com os valores padrão se não existir). Devolve (parâmetros, aviso)."""
    arq = caminho if os.path.isabs(caminho) else os.path.join(PASTA, caminho)
    if not os.path.exists(arq):
        try:
            with open(arq, "w", encoding="utf-8") as f:
                f.write(MODELO_ONDAS_TXT.format(**ONDAS_PADRAO))
        except OSError:
            pass
        return dict(ONDAS_PADRAO), None
    lido = {}
    with open(arq, encoding="utf-8") as f:
        for linha in f:
            linha = linha.split("#", 1)[0].strip()
            if "=" in linha:
                k, v = (p.strip() for p in linha.split("=", 1))
                lido[k.lower()] = v
    out, erros = dict(ONDAS_PADRAO), []
    try:
        out["inicio"] = pd.Timestamp(lido.get("inicio", out["inicio"])).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        erros.append(f"inicio = {lido.get('inicio')!r}")
    try:
        larg = float(str(lido.get("largura_semanas", out["largura_semanas"])).replace(",", "."))
        if not 1 <= larg <= 520:
            raise ValueError
        out["largura_semanas"] = larg
    except ValueError:
        erros.append(f"largura_semanas = {lido.get('largura_semanas')!r}")
    aviso = (f"{ARQ_ONDAS}: valor inválido ({'; '.join(erros)}) — usei o padrão." if erros else None)
    return out, aviso


def _cor_tema(opcao: str, padrao: str) -> str:
    try:
        return st.get_option(opcao) or padrao
    except Exception:  # noqa: BLE001
        return padrao


JS_ONDAS = """
<div class="ondas">
  <div class="ctl"><label>Deslocar ondas <b id="v_desl"></b></label>
    <input type="range" id="desl" min="-364" max="364" step="1"></div>
  <div class="ctl"><label>Largura de cada onda <b id="v_larg"></b></label>
    <input type="range" id="larg" step="0.5"></div>
  <div class="txt"><span id="txt"></span>
    <button id="copiar" title="copiar as duas linhas">copiar</button>
    <button id="voltar" title="voltar aos valores de base">↺ __BASE__</button></div>
</div>
<style>
  html,body{margin:0;background:transparent;font-family:"Source Sans Pro","Source Sans 3",system-ui,sans-serif;color:__TEXTO__}
  .ondas{display:grid;grid-template-columns:1fr 1fr;gap:4px 28px;padding:6px 6px 0}
  .ctl label{font-size:14px;display:flex;justify-content:space-between}
  .ctl label b{font-weight:600;color:__COR__}
  input[type=range]{width:100%;accent-color:__COR__;margin:6px 0 2px;cursor:pointer}
  .txt{grid-column:1/-1;font:12px/1.5 ui-monospace,Consolas,monospace;opacity:.55;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
  .txt button{font:inherit;color:inherit;background:none;border:1px solid currentColor;border-radius:4px;padding:0 6px;cursor:pointer;opacity:.8}
  .txt button:hover{opacity:1}
  @media(max-width:560px){.ondas{grid-template-columns:1fr}}
</style>
<script>
(function(){
  const P = __PARAMS__;                     // vem do Python: ondas.txt + janela de datas do gráfico
  const DIA = 864e5, N = 60;
  const div = document.getElementById("__DIV__");
  const desl = document.getElementById("desl"), larg = document.getElementById("larg");
  larg.min = Math.min(4, P.largura, P.ini_larg); larg.max = Math.max(104, P.largura, P.ini_larg);
  desl.min = -Math.max(364, Math.abs(P.ini_desl)); desl.max = Math.max(364, Math.abs(P.ini_desl));
  const CHAVE = "cryptinho.ondas";
  // lembra o ajuste entre reruns (troca de período, de ativo...) enquanto o ondas.txt não mudar
  let salvo = null;
  try { salvo = JSON.parse(localStorage.getItem(CHAVE) || "null"); } catch (e) {}
  const mesmaBase = salvo && salvo.inicio === P.inicio && salvo.largura_txt === P.largura &&
                    salvo.ini_desl === P.ini_desl && salvo.ini_larg === P.ini_larg;
  desl.value = mesmaBase ? salvo.desl : P.ini_desl;
  larg.value = mesmaBase ? salvo.largura : P.ini_larg;

  function iso(ms){ return new Date(ms).toISOString().slice(0, 10); }
  function desenhar(){
    const d = +desl.value, w = +larg.value, Pms = w * 7 * DIA;
    const ini = Date.parse(P.inicio) + d * DIA;
    const xmin = P.x0, xmax = P.x1 + Pms;          // janela do gráfico + uma onda de projeção
    const k0 = Math.floor((xmin - ini) / Pms), k1 = Math.ceil((xmax - ini) / Pms);
    const xs = [], ys = [];
    for (let k = k0; k < k1; k++)
      for (let j = 0; j < N; j++) {
        const t = j / (N - 1);
        xs.push(new Date(ini + (k + t) * Pms).toISOString());
        ys.push(P.pico * Math.sin(Math.PI * t));
      }
    Plotly.restyle(div, {x: [xs], y: [ys]}, [1]);
    Plotly.relayout(div, {"xaxis.range": [new Date(xmin).toISOString(), new Date(xmax).toISOString()]});
    document.getElementById("v_desl").textContent = (d > 0 ? "+" : "") + d + " dias";
    document.getElementById("v_larg").textContent = w.toLocaleString("pt-BR") + " semanas";
    const linhas = "inicio = " + iso(ini) + "   largura_semanas = " + w;
    const igual = d === P.ini_desl && w === P.ini_larg;
    document.getElementById("txt").textContent = igual
      ? (P.desc || ("ondas.txt: inicio = " + P.inicio + " · largura_semanas = " + P.largura + " (valores do arquivo)"))
      : "para fixar, no ondas.txt →  " + linhas;
    document.getElementById("voltar").style.display = igual ? "none" : "";
    try { localStorage.setItem(CHAVE, JSON.stringify({inicio: P.inicio, largura_txt: P.largura, ini_desl: P.ini_desl, ini_larg: P.ini_larg, desl: d, largura: w})); } catch (e) {}
  }
  desl.addEventListener("input", desenhar);
  larg.addEventListener("input", desenhar);
  document.getElementById("voltar").onclick = () => { desl.value = P.ini_desl; larg.value = P.ini_larg; desenhar(); };
  document.getElementById("copiar").onclick = async (ev) => {
    const ini = Date.parse(P.inicio) + (+desl.value) * DIA;
    const txt = "inicio = " + iso(ini) + "\\nlargura_semanas = " + (+larg.value);
    try { await navigator.clipboard.writeText(txt); ev.target.textContent = "copiado ✓"; }
    catch (e) { ev.target.textContent = "selecione e copie"; }
    setTimeout(() => ev.target.textContent = "copiar", 1500);
  };
  desenhar();
})();
</script>
"""


def config_ondas() -> tuple[bool, int]:
    """(automático?, tamanho dos ciclos) da aba Indicadores. Guardado em chaves próprias porque o Streamlit apaga o
    estado dos widgets de uma aba que não está aberta — assim o ranking (aba de ML) também enxerga a escolha."""
    ss = st.session_state
    auto = ss["ondas_auto"] if "ondas_auto" in ss else ss.get("_ondas_auto", True)
    ordem = ss["ondas_escala"] if "ondas_escala" in ss else ss.get("_ondas_escala", 10)
    ss["_ondas_auto"], ss["_ondas_escala"] = bool(auto), int(ordem)
    return bool(auto), int(ordem)


def ajustar_ondas(close: pd.Series, ordem: int = 10) -> dict | None:
    """Ondas que melhor encaixam nos fundos e topos do período: procura JUNTOS a largura (perto do ciclo médio,
    de 0,6× a 1,6× a mediana dos intervalos entre fundos/topos seguidos) e a posição que deixam os fundos mais
    perto dos vales e os topos mais perto dos picos (erro relativo à largura, para não favorecer ondas curtas)."""
    fundos, topos = ml.extremos(close, ordem)
    sem = lambda ds: np.diff(np.array(ds, dtype="datetime64[D]")).astype(int) / 7     # noqa: E731
    esp = np.concatenate([sem(fundos), sem(topos)]) if len(fundos) + len(topos) else np.array([])
    if len(esp) < 2:
        return None
    med = float(np.median(esp))
    ref = close.index[0]
    tf = np.array([(f - ref).days / 7 for f in fundos])
    tt = np.array([(t - ref).days / 7 for t in topos])
    melhor = None
    for W in np.arange(max(4.0, np.ceil(1.2 * med) / 2), 1.6 * med + 0.01, 0.5):
        fases = np.arange(0, W, 0.5)[:, None]
        circ = lambda x: np.abs((x + W / 2) % W - W / 2)                                # noqa: E731
        erro = np.hstack([circ(tf[None, :] - fases), circ(tt[None, :] - fases - W / 2)]).mean(axis=1)
        k = int(erro.argmin())
        if melhor is None or erro[k] / W < melhor[0]:
            melhor = (erro[k] / W, float(W), float(fases[k, 0]), float(erro[k]))
    _, W, v0, err = melhor
    return {"inicio": ref + pd.Timedelta(weeks=v0), "largura_semanas": W, "mediana": med,
            "n_fundos": len(fundos), "n_topos": len(topos), "esp_min": float(esp.min()), "esp_max": float(esp.max()),
            "erro": err, "acaso": W / 4, "fundos": fundos, "topos": topos}


AJUDA_ONDAS_AUTO = (
    "Acha os fundos e topos do período escolhido no slider e procura, juntos, a **largura** (perto do ciclo médio "
    "entre fundos/topos seguidos) e o **deslocamento** que deixam os fundos nos vales e os topos nos picos. Os dois "
    "sliders vão para esses valores (medidos a partir do ondas.txt) e continuam ajustando dali.\n\n"
    "⚠️ Descreve o passado. Os ciclos do BTC são irregulares: testando desde 2018 (ajustar com 3 anos e prever o "
    "próximo fundo), o erro mediano foi de ~5 meses — o mesmo que sortear a data.")


def grafico_semiciclos(df: pd.DataFrame, token: str, moeda: str = "USD", y_pico: float = 10):
    """Close + ondas senoidais (semiciclos). Início e largura vêm do ondas.txt; os sliders abaixo do gráfico
    deslocam e esticam as ondas no próprio navegador (JavaScript), sem rerun do Streamlit."""
    if df.empty:
        st.info("Sem dados no período para o gráfico de ondas.")
        return
    ondas, aviso = ler_ondas()
    if aviso:
        st.warning(aviso)
    auto, ordem = config_ondas()
    st.session_state.setdefault("ondas_auto", auto)          # padrão: automático ligado
    if auto:
        st.session_state.setdefault("ondas_escala", ordem)
    fit = ajustar_ondas(df["Close"], ordem) if auto else None
    desc, base_rot = None, "ondas.txt"
    ini_desl, ini_larg = 0, ondas["largura_semanas"]
    if fit:
        W = fit["largura_semanas"]
        # deslocamento a partir do início do ondas.txt; somar múltiplos de W não muda as ondas -> fica em ±W/2
        dif = (fit["inicio"] - pd.Timestamp(ondas["inicio"])).days
        ini_desl = int(round(((dif + W * 3.5) % (W * 7)) - W * 3.5))
        ini_larg = W
        base_rot = "automático"
        inicio_auto = (pd.Timestamp(ondas["inicio"]) + pd.Timedelta(days=ini_desl)).strftime("%Y-%m-%d")
        desc = (f"automático: inicio = {inicio_auto} · largura_semanas = {W:g} — encaixe em {fit['n_fundos']} fundos e "
                f"{fit['n_topos']} topos (intervalos de {fit['esp_min']:.0f} a {fit['esp_max']:.0f} semanas); erro médio "
                f"até a onda {fit['erro']:.1f} sem (ao acaso ~{fit['acaso']:.1f})")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="gold", width=2)))
    fig.add_trace(go.Scatter(x=[], y=[], name="Semiciclos", yaxis="y2",       # o JavaScript desenha
                             line=dict(color="cyan", width=3, dash="dot")))
    if fit:                                                                    # o que entrou no ajuste
        for nome, ds, simb, cor in (("fundos usados", fit["fundos"], "triangle-up", "#26a69a"),
                                    ("topos usados", fit["topos"], "triangle-down", "#ef5350")):
            if ds:
                fig.add_trace(go.Scatter(x=ds, y=df["Close"].loc[ds], mode="markers", name=nome,
                                         marker=dict(symbol=simb, size=11, color=cor, line=dict(color="#fff", width=1))))
    fig.update_layout(title=f"{token} Close + Ondas Simétricas", template="plotly_dark", hovermode="x unified",
                      yaxis=dict(title=f"Preço ({moeda})", tickfont=dict(color="gold")),
                      yaxis2=dict(title="Amplitude das Ondas", tickfont=dict(color="cyan"),
                                  overlaying="y", side="right", range=[0, y_pico * 1.05]),
                      legend=dict(x=0.02, y=0.98), margin=dict(l=60, r=60, t=60, b=40), height=450)
    id_div = "ondas_grafico"
    params = {"inicio": ondas["inicio"], "largura": ondas["largura_semanas"], "pico": y_pico, "desc": desc,
              "ini_desl": ini_desl, "ini_larg": ini_larg,
              "x0": int(df.index[0].timestamp() * 1000), "x1": int(df.index[-1].timestamp() * 1000)}
    pagina = (fig.to_html(full_html=False, include_plotlyjs="cdn", div_id=id_div,
                          config={"displaylogo": False, "responsive": True}, default_width="100%")
              + JS_ONDAS.replace("__PARAMS__", json.dumps(params)).replace("__DIV__", id_div).replace("__BASE__", base_rot)
              .replace("__COR__", _cor_tema("theme.primaryColor", "#9B7F5F"))
              .replace("__TEXTO__", _cor_tema("theme.textColor",
                                              "#31333F" if _cor_tema("theme.base", "dark") == "light" else "#FAFAFA")))
    if hasattr(st, "iframe"):                   # Streamlit ≥ 1.6x
        st.iframe(pagina, height=605)
    else:                                       # versões anteriores
        import streamlit.components.v1 as components
        components.html(pagina, height=605)
    ca, cb = st.columns([3, 2])
    ca.checkbox("Ajuste automático das ondas (ciclo médio)", key="ondas_auto",
                help=AJUDA_ONDAS_AUTO)
    if auto:
        cb.radio("Tamanho dos ciclos", [6, 10, 13], key="ondas_escala", horizontal=True,
                 format_func={6: "curtos", 10: "médios", 13: "longos"}.get,
                 help="Que balanço conta como fundo/topo: mínimo/máximo de ±6, ±10 ou ±13 semanas. O 'ciclo' "
                      "muda bastante com essa escolha — mais um sinal de que ele não é regular.")
    if auto and not fit:
        st.caption("Poucos fundos/topos no período para achar um ciclo médio — usando o ondas.txt. Aumente o período.")


# =========================================================================== #
# Rebalanceador
# =========================================================================== #

CAIXAS = ("USD", "BRL")


def _parse_txt(text: str):
    if not text:
        return {}, pd.DataFrame(columns=["Ativo", "Qt", "Data"])
    txt = "\n".join(l_ for l_ in text.splitlines() if l_.strip() and not l_.strip().startswith("#"))
    alvo_txt, _, trade_txt = txt.partition("@")

    pares = re.findall(r"(?mi)^\s*([A-Za-z0-9_.\-]+)\s+([0-9]*\.?[0-9]+)\s*$", alvo_txt)
    alvos = pd.DataFrame(pares, columns=["Ativo", "Pct"])
    alvos["Pct"] = pd.to_numeric(alvos["Pct"], errors="coerce")
    alvos["Ativo"] = alvos["Ativo"].str.upper().str.replace(r"\.SA$", "", regex=True)
    alvos = alvos.dropna(subset=["Pct"]).groupby("Ativo", as_index=False)["Pct"].sum()
    targets = dict(zip(alvos["Ativo"], alvos["Pct"]))
    soma = float(alvos["Pct"].sum())

    if trade_txt.strip():
        trades = pd.read_csv(StringIO(trade_txt.strip()), sep=r"[,\s]+", engine="python", header=None,
                             names=["Ativo", "Qt", "Data"], usecols=[0, 1, 2])
        trades["Ativo"] = trades["Ativo"].str.upper().str.replace(r"\.SA$", "", regex=True)
        trades["Qt"] = pd.to_numeric(trades["Qt"], errors="coerce")
        trades["Data"] = pd.to_datetime(trades["Data"], dayfirst=True, errors="coerce")
        trades = trades.dropna(subset=["Ativo", "Qt"])
    else:
        trades = pd.DataFrame(columns=["Ativo", "Qt", "Data"])

    if soma < 1 - 1e-9:
        # o que sobra vira caixa: em BRL se você só tem caixa em reais; senão em USD
        tem = set(trades["Ativo"])
        caixa = "BRL" if ("BRL" in tem or "BRL" in targets) and "USD" not in tem else "USD"
        targets[caixa] = round(targets.get(caixa, 0) + 1 - soma, 12)
    elif soma > 1 + 1e-9:
        st.warning(f"Soma dos alvos = {soma:.4f} (>1). Ajuste seu TXT.")
    return targets, trades


def mercado_do_ativo(s: str, criptos: tuple) -> str:
    """Em qual mercado procurar o preço. Ordem: caixa > listas conhecidas > padrão de ticker da B3 > cripto/EUA."""
    if s in CAIXAS:
        return "Caixa"
    if s in ACOES_B3:
        return "Ações B3"
    if s in ACOES_EUA:
        return "Ações EUA"
    if s in criptos:
        return "Cripto"
    if re.fullmatch(r"[A-Z]{4}\d{1,2}", s):          # ex.: BBDC4, KNRI11, AAPL34
        return "Ações B3"
    return "?"                                        # tenta cripto e, se não achar, ação dos EUA


@cache_dados(ttl="15m", show_spinner="Buscando preços e câmbio...")
def precos_carteira(itens: tuple[tuple[str, str], ...]) -> tuple[dict, float]:
    """Último preço de cada ativo na moeda original + câmbio (BRL por 1 USD, ticker BRL=X do Yahoo)."""
    return fontes.ultimos_precos(itens)


def _converter(preco: float, de: str, para: str, fx: float) -> float:
    if de == para:
        return preco
    if de == "USD" and para == "BRL":
        return preco * fx
    return preco / fx                                 # BRL -> USD


def _rebalancear(targets: dict, trades: pd.DataFrame, precos: pd.Series):
    qty = trades.groupby("Ativo")["Qt"].sum() if not trades.empty else pd.Series(dtype=float)
    idx = pd.Index(sorted(set(qty.index) | set(targets)))
    qty = qty.reindex(idx, fill_value=0.0)
    preco = precos.reindex(idx)
    val = qty * preco
    total = float(val.replace([np.inf, -np.inf], np.nan).sum(skipna=True))
    cur = (val / total).fillna(0.0) if total > 0 else pd.Series(0.0, index=idx)
    tgt = pd.Series(targets, dtype=float).reindex(idx, fill_value=0.0)
    diff = tgt * total - val
    out = pd.DataFrame({"Ativo": idx, "Qt": qty.values, "Preco": preco.values, "Valor": val.values,
                        "% Atual": cur.values, "% Alvo": tgt.values, "Ajuste": diff.values,
                        "DeltaUnits": (diff / preco.replace(0, np.nan)).values})
    return out, total


def exibir_rebalanceamento(criptos: tuple, up, txt_manual: str):
    # getvalue() (e não read()) — read() esvazia o arquivo nos reruns seguintes
    content = up.getvalue().decode("utf-8", errors="ignore") if up else (txt_manual or "").strip()
    if not content:
        st.info("Envie um arquivo ou cole o conteúdo na barra lateral. Pode misturar cripto, ações da B3 "
                "(ex.: ITUB4), ações dos EUA (ex.: NVDA) e caixa em USD ou BRL.")
        return
    targets, trades = _parse_txt(content)
    base = st.radio("Ver carteira em", ["USD", "BRL"], horizontal=True,
                    help="Tudo é convertido para esta moeda pelo câmbio do dia (USD/BRL do Yahoo Finance, "
                         "ticker BRL=X). Cripto e ações dos EUA são cotadas em USD; ações e FIIs da B3, em BRL.")
    sim = "US$" if base == "USD" else "R$"

    st.subheader("🧾 Aquisições")
    st.dataframe(trades.sort_values("Data"), hide_index=True)

    syms = sorted(set(targets) | set(trades["Ativo"]))
    itens = tuple((s_, mercado_do_ativo(s_, criptos)) for s_ in syms)
    cot, fx = precos_carteira(itens)
    if not np.isfinite(fx):
        st.error("Não consegui o câmbio USD/BRL agora; ativos em moedas diferentes não podem ser somados.")
        return
    precos = pd.Series({s_: _converter(p, m, base, fx) for s_, (p, m, _) in cot.items()}, dtype=float)
    out, total = _rebalancear(targets, trades, precos)
    out.insert(1, "Mercado", out["Ativo"].map(lambda a: cot.get(a, (0, "", "?"))[2]))
    out.insert(3, "Preço original", out["Ativo"].map(
        lambda a: f"{cot[a][1]} {ln.fmt(cot[a][0])}" if a in cot and np.isfinite(cot[a][0]) else "—"))
    sem_preco = out.loc[out["Preco"].isna(), "Ativo"].tolist()
    if sem_preco:
        st.warning("Sem preço (ficaram FORA do total): " + ", ".join(sem_preco) + ". Confira o código do ativo." +
                   (" Na versão web, ações só têm preço se estiverem nas listas embutidas no HTML."
                    if fontes.NO_NAVEGADOR else ""))

    st.subheader(f"💰 Valor total: **{sim} {total:,.2f}**")
    origem_fx = fontes.STATUS.get("Câmbio", "BRL=X do retrato") if fontes.NO_NAVEGADOR else "Yahoo Finance, BRL=X"
    st.caption(f"Câmbio usado: 1 USD = {fx:,.4f} BRL ({origem_fx}).")

    show = out.assign(**{"% Atual": (out["% Atual"] * 100).round(2), "% Alvo": (out["% Alvo"] * 100).round(2)})
    st.subheader("Plano de Rebalanceamento")
    st.dataframe(show.rename(columns={"Qt": "Qtd atual", "Preco": f"Preço ({base})", "Valor": f"Valor ({base})",
                                      "Ajuste": f"Ajuste ({base})", "DeltaUnits": "Δ Unidades"}),
                 hide_index=True, column_config={
                     f"Preço ({base})": st.column_config.NumberColumn(format="%.4g"),
                     f"Valor ({base})": st.column_config.NumberColumn(format="%.2f"),
                     f"Ajuste ({base})": st.column_config.NumberColumn(format="%.2f")})

    st.subheader("Ações sugeridas (exclui caixa)")
    ativos = ~out["Ativo"].isin(CAIXAS) & out["Ajuste"].notna()
    c1, c2 = st.columns(2)
    for col, titulo, mask in ((c1, "**Comprar**", out["Ajuste"] > 0), (c2, "**Vender**", out["Ajuste"] < 0)):
        with col:
            st.write(titulo)
            st.dataframe(out.loc[ativos & mask, ["Ativo", "Mercado", "Ajuste", "DeltaUnits"]]
                         .rename(columns={"Ajuste": f"Ajuste ({base})", "DeltaUnits": "Δ Unidades"})
                         .round({f"Ajuste ({base})": 2}), hide_index=True)
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(show, values="% Atual", names="Ativo", title="Distribuição Atual",
                     color_discrete_sequence=px.colors.sequential.Tealgrn)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig.update_layout(showlegend=False))
    with c2:
        alvo = pd.DataFrame({"Ativo": list(targets), "% Alvo": list(targets.values())})
        fig = px.pie(alvo, values="% Alvo", names="Ativo", title="Distribuição da Carteira Planejada",
                     color_discrete_sequence=px.colors.sequential.RdBu)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig.update_layout(showlegend=False))


# =========================================================================== #
# Extras
# =========================================================================== #

@cache_dados(ttl="1h", show_spinner=False)
def fear_and_greed():
    return fontes.fear_greed_atual()


def links_uteis():
    st.markdown("""
### 🔗 Links Úteis — Análise de Mercado
**📊 Indicadores Gerais**
- [**TVL (Total Value Locked)** — DefiLlama](https://defillama.com/)
- [**Market Cap & Dados** — CoinGecko](https://www.coingecko.com/pt)

**🌐 Situação Macro da Criptoeconomia**
- [**Dashboard Macro Cripto** — SoSoValue](https://sosovalue.com/pt/dashboard/charts)

**⚠️ Indicadores de Alerta**
- [**AHR999 Indicator** — linha verde = zona de oportunidade](https://sosovalue.com/pt/dashboard/ahr-999-indicator)
- [**ETF Fund Flow (1W)** — atenção a saídas 2 semanas seguidas](https://sosovalue.com/pt/dashboard/total-crypto-spot-etf-fund-flow-1w)
""")


def revisao_volume(token: str, top50: list, top100: list):
    if token in ("USD", "USDT", "XAUT"):
        st.info(f"ℹ️ {token}: Stablecoin.")
    elif token in top100:
        st.warning(f"⚠️ {token}: Altcoin de **volume muito baixo**, geralmente pode estar relacionado ao "
                   "movimento de ativos com **alto volume**. Requer atenção redobrada.")
    elif token in top50:
        st.info(f"ℹ️ {token}: Altcoin de **volume relativamente baixo**, pode estar correlacionado com "
                "movimentos de ativos de **alto volume**.")
    elif token != "BTC":
        st.info(f"ℹ️ {token}: Altcoin de alto volume, pode estar correlacionado com movimentos do BTC, "
                "que possui **volume maior**.")


# =========================================================================== #
# Período dos gráficos (slider de duas pontas, abaixo dos gráficos)
# =========================================================================== #

def periodo_grafico(chave: str, idx: pd.DatetimeIndex, meses: int = MESES_GRAFICO) -> tuple:
    """Período escolhido no slider ``chave``, ajustado aos limites dos dados.

    É lido ANTES de desenhar o gráfico e o slider é criado DEPOIS (embaixo dele):
    o valor vem do session_state da interação anterior. Padrão: últimos ``meses``.
    Se o fim estava "colado" no último candle, continua colado quando chegam dados novos.
    """
    lo, hi = idx[0].date(), idx[-1].date()
    padrao = (max(lo, (idx[-1] - pd.DateOffset(months=meses)).date()), hi)
    val = st.session_state.get(chave)
    if val and len(val) == 2:
        a, b = val
        if b == st.session_state.get(f"{chave}_hi"):
            b = hi
        a, b = min(max(a, lo), hi), min(max(b, lo), hi)
        val = (a, b) if a < b else padrao
    else:
        val = padrao
    st.session_state[chave] = val
    st.session_state[f"{chave}_hi"] = hi
    return pd.Timestamp(val[0]), pd.Timestamp(val[1])


def slider_periodo(chave: str, idx: pd.DatetimeIndex, ajuda: str | None = None):
    st.slider("Período dos gráficos", min_value=idx[0].date(), max_value=idx[-1].date(), key=chave,
              format="DD/MM/YYYY", help=ajuda or f"Padrão: últimos {MESES_GRAFICO} meses. Os indicadores e "
                                                 "sinais continuam calculados com todo o histórico.")


# =========================================================================== #
# Aba: Análise por Linhas
# =========================================================================== #

@cache_dados(ttl=TTL, show_spinner=False, max_entries=500)
def calcular_linhas(chave: tuple, freq: str, token: str, p: ln.ParamsLinhas, backtest: bool,
                    ini: str | None = None, fim: str | None = None) -> dict:
    """Linhas calculadas só no período escolhido no slider (padrão: últimos 36 meses)."""
    df = preparar_dados(*chave)[freq][token].loc[ini:fim]
    return ln.analisar(df, p, backtest=backtest)


MODOS = {"toques": "Mais toques = mais forte",
         "virgem": "Menos testada = mais forte",
         "calibrado": "Calibrado pelo backtest do ativo"}


@cache_dados(ttl=TTL, show_spinner=False)
def tendencia_linhas_token(chave: tuple, token: str) -> tuple[pd.DataFrame, dict]:
    tl = ln.tendencia_linhas(preparar_dados(*chave)["1D"][token])
    return tl, dict(tl.attrs)                     # attrs à parte: o cache nem sempre os preserva


EVENTOS_LINHAS = [   # (evento, lado, toques mínimos, nome, ícone)
    ("rompimento", 1, 0, "Rompimento de LTB/resistência para cima", "🚀"),
    ("rompimento", 1, ln.TEND_TOQUES, f"Rompimento de LTB/resistência para cima (linha com ≥ {ln.TEND_TOQUES} toques)", "🚀"),
    ("rompimento", -1, 0, "Perda de LTA/suporte", "⚠️"),
    ("rompimento", -1, ln.TEND_TOQUES, f"Perda de LTA/suporte (linha com ≥ {ln.TEND_TOQUES} toques)", "⚠️"),
    ("toque", 1, 0, "Toque em LTA/suporte", "🟢"),
    ("toque", -1, 0, "Toque em LTB/resistência", "🔴"),
    ("reteste", 1, 0, "Reteste da LTB/resistência rompida", "🎯"),
    ("reteste", -1, 0, "Reteste da LTA/suporte perdida", "🎯"),
    ("rompimento falso", 1, 0, "Rompimento falso de suporte (voltou para cima)", "↩️"),
    ("rompimento falso", -1, 0, "Rompimento falso de resistência (voltou para baixo)", "↩️"),
    ("divergência", 1, 0, "Divergência de alta RSI × preço", "🟢"),
    ("divergência", -1, 0, "Divergência de baixa RSI × preço", "🔴"),
]
PRAZOS_LINHAS = (7, 14, 30)


@cache_dados(ttl=TTL, show_spinner="Medindo o que funcionou nas linhas deste ativo...")
def estudo_linhas_token(chave: tuple, token: str) -> pd.DataFrame:
    """Cada tipo de evento das linhas, no diário e no semanal, medido no histórico do ativo: no dia do evento, o preço
    7/14/30 dias depois foi para o lado do sinal mais vezes que o acaso? (mesma régua do ranking da aba de ML)"""
    d = preparar_dados(*chave)["1D"][token].dropna(subset=["Close"])
    pr, idx = d["Close"].astype(float), d.index
    p = ln.ParamsLinhas()
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    sem = d.resample("W-MON", label="left", closed="left").agg(agg).dropna(subset=["Close"])
    ew = ln.eventos_sinal(sem, replace(p, gap_min=4, span_max=150, janela_reteste=30))
    ew["Date"] = ew["Date"] + pd.Timedelta(days=7)            # o candle semanal fecha 7 dias depois do rótulo
    ew = ew[ew["Date"] <= idx[-1]]
    ew["Date"] = idx[np.minimum(idx.searchsorted(ew["Date"]), len(idx) - 1)]
    linhas_ = []

    def medir(grafico, nome, icone, lado, sinal):
        r = {"Gráfico": grafico, "Evento": f"{icone} {nome}", "Sinal": "🟢 COMPRA" if lado > 0 else
             ("🔴 VENDA" if lado < 0 else "🟢/🔴"),
             "Vezes": int(((sinal != 0) & (sinal != sinal.shift())).sum())}   # entradas de sinal (combinado) ou eventos
        for P in PRAZOS_LINHAS:
            a = ml.avaliar_metodo(sinal, pr, P)
            r[f"{P} dias"] = 100 * a["vantagem"] if a.get("n") else np.nan
            if P == 14:
                r["Acerto 14 dias %"] = 100 * a["acerto"] if a.get("n") else np.nan
        r["Média"] = np.nanmean([r[f"{P} dias"] for P in PRAZOS_LINHAS])
        linhas_.append(r)

    tl = ln.tendencia_linhas(d)
    medir("Semanal + diário", "Sinal combinado (o do quadro acima e do ranking)", "⭐", 0, tl["sinal"])
    for grafico, e in (("Diário", ln.eventos_sinal(d, p)), ("Semanal", ew)):
        for ev, lado, tmin, nome, icone in EVENTOS_LINHAS:
            g = e[(e["evento"] == ev) & (e["lado"] == lado) & (e["toques"] >= tmin)]
            if len(g) < 5:
                continue
            sinal = pd.Series(0, index=idx)
            sinal.loc[g["Date"].unique()] = lado
            medir(grafico, nome, icone, lado, sinal)
    return pd.DataFrame(linhas_)


def destaques_linhas(chave: tuple, token: str):
    """Texto com o que mais e o que menos funcionou nas linhas deste ativo (+ tabela completa)."""
    df = preparar_dados(*chave)["1D"].get(token)
    if df is None or len(df) < 400:
        return
    tab = estudo_linhas_token(chave, token)
    if tab.empty:
        return
    fmt_ = lambda r: " / ".join(f"{r[f'{P} dias']:+.1f}" for P in PRAZOS_LINHAS)        # noqa: E731
    conf = tab[tab["Vezes"] >= 15]
    eventos_ = conf[conf["Gráfico"] != "Semanal + diário"]
    bons = eventos_[(eventos_["Média"] >= 2) & (eventos_[[f"{P} dias" for P in PRAZOS_LINHAS]].min(axis=1) > -2)]
    bons = bons.sort_values("Média", ascending=False).head(4)
    ruins = eventos_[eventos_["Média"] <= -2].sort_values("Média").head(3)
    comb = tab[tab["Gráfico"] == "Semanal + diário"]
    txt = [f"**O que mais funcionou em {token}** — pontos acima do acaso em 7 / 14 / 30 dias, no histórico todo:"]
    if not comb.empty:
        c_ = comb.iloc[0]
        txt.append(f"- ⭐ **Sinal combinado (semanal + diário)**: {fmt_(c_)} p.p.")
    for r in bons.to_dict("records"):
        txt.append(f"- {r['Evento']} no **{r['Gráfico'].lower()}** → {r['Sinal']}: {fmt_(r)} p.p. ({r['Vezes']} vezes)")
    if bons.empty:
        txt.append("- Nenhum evento isolado ficou acima do acaso com folga (≥ 2 p.p. na média).")
    if not ruins.empty:
        txt.append("\n**O que não funcionou** (ficou abaixo do acaso — melhor não seguir sozinho):")
        for r in ruins.to_dict("records"):
            txt.append(f"- {r['Evento']} no **{r['Gráfico'].lower()}** → {r['Sinal']}: {fmt_(r)} p.p. ({r['Vezes']} vezes)")
    st.markdown("\n".join(txt))
    with st.expander(f"Todos os eventos das linhas em {token}, medidos um a um"):
        st.caption("Para cada tipo de evento: nos dias em que ele aconteceu (só com o que já se sabia naquele dia), o "
                   "preço N dias depois foi para o lado do sinal? 'Acima do acaso' desconta quanto acertaria quem desse "
                   "a mesma quantidade de compras/vendas em dias sorteados. Eventos com poucas vezes (< 15) são "
                   "anedóticos. Usa todas as linhas possíveis (não só as desenhadas no gráfico) e os parâmetros padrão.")
        cfg = {f"{P} dias": st.column_config.NumberColumn(f"Acima do acaso {P}d", format="%+.1f") for P in PRAZOS_LINHAS}
        cfg |= {"Média": st.column_config.NumberColumn(format="%+.1f"),
                "Acerto 14 dias %": st.column_config.NumberColumn(format="%.0f"),
                "Evento": st.column_config.TextColumn(width="large")}
        st.dataframe(tab.sort_values(["Gráfico", "Média"], ascending=[True, False]), hide_index=True, column_config=cfg,
                     column_order=["Gráfico", "Evento", "Sinal", "Vezes", "Acerto 14 dias %", "7 dias", "14 dias",
                                   "30 dias", "Média"])


def leitura_tendencia_linhas(chave: tuple, token: str):
    """Resumo do sinal de COMPRA/VENDA das linhas (o mesmo do ranking na aba de ML), sempre no diário + semanal,
    independente da periodicidade e dos controles desta aba."""
    df = preparar_dados(*chave)["1D"].get(token)
    if df is None or len(df) < 200:
        return
    tl, a = tendencia_linhas_token(chave, token)
    d = lambda x: "—" if pd.isna(x) else f"{x:%d/%m/%Y}"                 # noqa: E731
    u = tl.iloc[-1]
    sw, sd = int(u["semanal"]), int(u["diario"])
    nome = {1: "de alta", -1: "de baixa", 0: "indefinida"}
    corpo = (f"Semanal: tendência **{nome[sw]}** (última LTB/resistência rompida para cima: {d(a['ult_sem_alta'])}; "
             f"última LTA/suporte perdida: {d(a['ult_sem_baixa'])}). Diário: "
             + (f"rompimento recente **para {'cima' if sd > 0 else 'baixo'}**" if sd else
                f"nenhum rompimento nos últimos {ln.TEND_VALIDADE} dias")
             + f" (para cima: {d(a['ult_dia_alta'])}; para baixo: {d(a['ult_dia_baixa'])}). "
             f"Só contam linhas com ≥ {ln.TEND_TOQUES} toques.")
    s_ = int(u["sinal"])
    if s_ > 0:
        st.success(f"**Sinal das linhas: 🟢 COMPRA** — semanal e diário romperam para cima.  \n{corpo}")
    elif s_ < 0:
        st.error(f"**Sinal das linhas: 🔴 VENDA** — semanal e diário perderam suporte.  \n{corpo}")
    else:
        st.info(f"**Sinal das linhas: ⚪ sem sinal** — semanal e diário não concordam.  \n{corpo}")


@st.fragment
def aba_linhas(chave: tuple, freq: str, token: str, tokens: list):
    """Fragmento: mexer nos controles daqui reexecuta só esta aba."""
    diario = freq == "1D"
    with st.expander("⚙️ Parâmetros das linhas", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            sens = st.slider("Sensibilidade dos pivôs (barras de cada lado)", 2, 15, 3,
                             help="Maior = só topos/fundos mais relevantes (e confirmação mais atrasada).")
            prom = st.slider("Proeminência mínima do pivô (× tolerância)", 0.0, 6.0, 1.0, 0.5)
        with c2:
            tol = st.slider("Tolerância do preço (× ATR)", 0.2, 1.5, 0.5, 0.1,
                            help="Distância máxima para contar como toque, e para considerar rompimento.")
            tol_rsi = st.slider("Tolerância do RSI (pontos)", 0.5, 5.0, 2.0, 0.5)
        with c3:
            max_l = st.slider("Linhas ativas por tipo", 1, 6, 3)
            modo = st.radio("Força das linhas", list(MODOS), format_func=MODOS.get)
        with c4:
            log = st.toggle("Preço em escala log", True)
            zonas = st.toggle("Zonas horizontais", True)

    p = ln.ParamsLinhas(
        esquerda=sens, direita=sens, prom_min=prom, tol_atr=tol, tol_rsi=tol_rsi, max_linhas=max_l,
        modo_forca=modo, log=log, gap_min=max(2 * sens, 6),
        span_max=400 if diario else 150, meia_vida=90 if diario else 26,
        janela_rompidas=40 if diario else 12, projecao=30 if diario else 12,
        janela_reteste=120 if diario else 26,
    )
    df_todo = preparar_dados(*chave)[freq][token]
    ini, fim = periodo_grafico("periodo_linhas", df_todo.index)
    df = df_todo.loc[ini:fim]
    if len(df) < 60:
        st.warning("Período curto demais para traçar linhas: amplie o intervalo no seletor abaixo.")
        slider_periodo("periodo_linhas", df_todo.index)
        return
    faixa = (ini.date().isoformat(), fim.date().isoformat())
    res = calcular_linhas(chave, freq, token, p, False, *faixa)

    f1, f2, f3 = st.columns([1, 2, 1])
    usar_fibo = f1.toggle("📐 Fibonacci", value=False,
                          help="Retração de Fibonacci entre a mínima e a máxima do PERÍODO escolhido no seletor "
                               "abaixo do gráfico. Mude o período para ancorar em outro fundo/topo.")
    fibo = None
    if usar_fibo:
        dir_fibo = f2.radio("Direção", ["auto", "fundo_topo", "topo_fundo"], horizontal=True,
                            format_func={"auto": "Automática", "fundo_topo": "Fundo → topo",
                                         "topo_fundo": "Topo → fundo"}.get,
                            help="Automática: segue a ordem em que o fundo e o topo aconteceram no período. "
                                 "Fundo → topo: 0 no fundo e 1 no topo (extensões acima). "
                                 "Topo → fundo: 0 no topo e 1 no fundo (extensões abaixo).")
        ext_fibo = f3.checkbox("Extensões (1,272 · 1,618 · 2 · 2,618)", value=False)
        fibo = ln.fibonacci(df, dir_fibo, ext_fibo)

    preco_atual(preparar_dados(*chave)["1D"].get(token), token, MERCADOS.get(chave[3], "USD") if len(chave) > 3 else "USD")
    st.plotly_chart(ln.figura_linhas(df, res, token, freq, len(df), mostrar_zonas=zonas, fibo=fibo), theme=None)
    if fibo:
        baixo_, alto_ = fibo["entre"]
        desc = (f"fundo {ln.fmt(fibo['baixo'])} em {fibo['i_baixo']:%d/%m/%Y} → topo {ln.fmt(fibo['alto'])} em "
                f"{fibo['i_alto']:%d/%m/%Y}" if fibo["direcao"] == "fundo_topo" else
                f"topo {ln.fmt(fibo['alto'])} em {fibo['i_alto']:%d/%m/%Y} → fundo {ln.fmt(fibo['baixo'])} em "
                f"{fibo['i_baixo']:%d/%m/%Y}")
        if baixo_ and alto_:
            onde = (f"entre **{baixo_[0]:g}** ({ln.fmt(baixo_[1])}) e **{alto_[0]:g}** ({ln.fmt(alto_[1])})")
        elif baixo_:
            onde = f"acima do nível **{baixo_[0]:g}** ({ln.fmt(baixo_[1])})"
        else:
            onde = f"abaixo do nível **{alto_[0]:g}** ({ln.fmt(alto_[1])})"
        st.caption(f"Fibonacci: {desc}. Preço atual ({ln.fmt(float(df['Close'].iloc[-1]))}) {onde}.")
    slider_periodo("periodo_linhas", df_todo.index,
                   ajuda="As linhas são calculadas só com os candles deste período.")
    leitura_tendencia_linhas(chave, token)        # sinal geral (diário + semanal) e o que funcionou neste ativo
    destaques_linhas(chave, token)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Linhas selecionadas**")
        st.dataframe(ln.tabela_linhas(res), hide_index=True, column_config={
            "Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
            "Valor agora": st.column_config.NumberColumn(format="%.4g"),
        })
        st.caption("Distância em unidades de tolerância (preço: ×ATR; RSI: pontos). Positiva = do lado "
                   "'certo' da linha (acima do suporte / abaixo da resistência; numa linha ROMPIDA, do lado "
                   "novo: acima da LTB rompida, abaixo da LTA perdida). No gráfico, a linha pontilhada é a "
                   "continuação da rompida, já com o papel trocado; ✕ = rompimento, ◆R = reteste.")
    with c2:
        st.markdown("**Sinais por linhas (candle atual)**")
        if res["sinais"].empty:
            st.info("Nenhum toque ou rompimento relevante agora.")
        else:
            st.dataframe(res["sinais"], hide_index=True, column_config={
                "Força": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f")})


# =========================================================================== #
# Aba: Indicadores × ML
# =========================================================================== #

@cache_dados(ttl="6h", show_spinner=False)
def fear_greed_historico() -> pd.Series:
    """Histórico diário do Medo & Ganância (alternative.me, desde 01/02/2018)."""
    return fontes.fear_greed_historico()


@cache_dados(ttl=TTL, show_spinner=False, max_entries=12)
def base_indicadores(chave: tuple, prazo: int, so_token: str | None) -> pd.DataFrame:
    d1 = preparar_dados(*chave)["1D"]
    diarios = {s_: df[OHLCV] for s_, df in d1.items() if so_token in (None, s_)}
    cripto = len(chave) < 4 or chave[3] == "Cripto"
    return ml.montar_base(diarios, prazo, fear_greed_historico() if cripto else None)


@st.cache_resource
def _memo_ml() -> dict:
    return {}


VERSAO_ML = (ml.EMBARGO, tuple(ml.INDICADORES))      # mudou o ML -> o resultado guardado na memória é refeito


def analisar_ml(chave: tuple, prazo: int, so_token: str | None, grupos: tuple, tarefa: str, familia: str,
                progresso=None):
    """Resultado guardado por configuração (com barra de progresso na 1ª vez)."""
    memo = _memo_ml()
    k = (id(preparar_dados(*chave)), chave, prazo, so_token, grupos, tarefa, familia, VERSAO_ML)
    if k not in memo:
        base = base_indicadores(chave, prazo, so_token)
        cols = [c for c in ml.INDICADORES if ml.GRUPO[c] in grupos]
        memo[k] = ml.analisar(base, cols, prazo, tarefa, familia, progresso)
        while len(memo) > 24:
            memo.pop(next(iter(memo)))
    return memo[k]


def historico_ml(chave: tuple, prazo: int, so_token: str | None, grupos: tuple, tarefa: str, familia: str,
                 res: dict, progresso=None) -> pd.DataFrame:
    """Sinal refeito mês a mês com o modelo escolhido (guardado por configuração)."""
    memo = _memo_ml()
    k = ("hist", id(preparar_dados(*chave)), chave, prazo, so_token, grupos, tarefa, familia, VERSAO_ML)
    if k not in memo:
        base = base_indicadores(chave, prazo, so_token)
        passo = 60 if res["fam"] == "rf" or fontes.NO_NAVEGADOR else 30     # Random Forest é o mais lento
        memo[k] = ml.historico_sinal(base, res["cols"], prazo, tarefa, res["fam"], res["valor"], passo=passo,
                                     anos=99, progresso=progresso)
        while len(memo) > 24:
            memo.pop(next(iter(memo)))
    return memo[k]


# --------------------------------------------------------------------------- #
# Ranking dos métodos (item 6 da aba de ML): todos os sinais do app na mesma régua
# --------------------------------------------------------------------------- #

def sinal_ondas(datas: pd.DatetimeIndex, inicio, largura_semanas: float, prazo: int, limiar: float = 0.1) -> np.ndarray:
    """O que as ondas 'dizem' para os próximos `prazo` dias: COMPRA se a onda sobe (indo do vale para o pico),
    VENDA se desce. Perto do pico ou do vale (variação < `limiar` da altura) = sem sinal."""
    W = float(largura_semanas)
    t = (datas - pd.Timestamp(inicio)).days.to_numpy() / 7.0
    altura = lambda x: np.sin(np.pi * ((x % W) / W))                       # noqa: E731
    d = altura(t + prazo / 7.0) - altura(t)
    return np.where(d > limiar, 1, np.where(d < -limiar, -1, 0))


def historico_ondas_auto(preco: pd.Series, prazo: int, anos_ajuste: int = 3, passo: int = 30,
                         ordem: int = 10) -> pd.Series:
    """Ajuste automático das ondas refeito mês a mês, só com os 3 anos anteriores (sem ver o futuro)."""
    sem = preco.resample("W-MON", label="left", closed="left").last().dropna()
    out = pd.Series(0, index=preco.index)
    inicio = preco.index[0] + pd.Timedelta(days=365 * anos_ajuste)
    for D in pd.date_range(inicio, preco.index[-1], freq=f"{passo}D"):
        jan = sem.loc[D - pd.Timedelta(weeks=52 * anos_ajuste): D - pd.Timedelta(days=7)]
        fit = ajustar_ondas(jan, ordem) if len(jan) > 60 else None
        m = (out.index > D) & (out.index <= D + pd.Timedelta(days=passo))
        if fit and m.any():
            out[m] = sinal_ondas(out.index[m], fit["inicio"], fit["largura_semanas"], prazo)
    return out


def historico_linhas(df: pd.DataFrame) -> pd.Series:
    """Leitura agregada das linhas (semanal dá a tendência, diário confirma) — ver ln.tendencia_linhas."""
    return ln.tendencia_linhas(df)["sinal"]


SETA = {1: "↑", -1: "↓", 0: "—"}


def texto_tendencia_linhas(tl: pd.DataFrame) -> str:
    """'semanal ↑ · diário ↑' para a coluna 'Sinal hoje'."""
    u = tl.iloc[-1]
    return f"semanal {SETA[int(u['semanal'])]} · diário {SETA[int(u['diario'])]}"


DESC_METODOS = {
    "Modelo de ML": "Sinal do item 5 (COMPRA/VENDA = 25% maiores/menores chances), refeito mês a mês.",
    "Alerta de fundo (◆ laranja)": "COMPRA nos dias de alerta de fundo (RSI < 35 e queda forte em desvios).",
    "Alerta de fundo FORTE (◆ vermelho)": f"COMPRA nos dias de alerta de fundo com queda ainda maior (≥ "
                                           f"{ml.FUNDO_FORTE_Z:g} desvios): mais raro, acerta mais.",
    "Alerta de topo (◆ roxo)": f"VENDA nos dias de alerta de topo (divergência de baixa, ou perdeu a EMA{ml.TOPO_EMA} depois de "
                               "uma alta forte).",
    "Análise de linhas": f"Rompimentos de linhas com ≥ {ln.TEND_TOQUES} toques: o último rompimento no SEMANAL dá a "
                         f"tendência e o diário confirma com um rompimento do mesmo lado nos últimos {ln.TEND_VALIDADE} "
                         "dias. COMPRA = LTB/resistência rompida para cima nos dois; VENDA = LTA/suporte perdida nos "
                         "dois; se discordam, sem sinal. Toques na linha não contam (erraram mais que o acaso).",
    "Buy/Sell Ratio (diário)": f"Regras que contam no ratio (aba Indicadores) em cada dia: ratio ≥ {RATIO_COMPRA:g} = "
                               f"COMPRA, ≤ 1/{RATIO_COMPRA:g} = VENDA. Indicadores de suporte não entram.",
    "Tabela de faixas (item 5)": "Média do 'Subiu %' das faixas de cada indicador (só com resultados já conhecidos "
                                 "no dia) acima do normal = COMPRA; abaixo = VENDA.",
    "Sempre comprar (referência)": "Para comparar: COMPRA todos os dias.",
}


# muda quando a definição de algum método muda -> o ranking guardado na memória é refeito
VERSAO_METODOS = (4, RATIO_COMPRA, tuple((r.lado, r.conta) for r in REGRAS),
                  ml.ALERTA_RSI, ml.ALERTA_Z, "topo-perda-de-força", ml.TOPO_Z, ml.TOPO_DIAS, ml.TOPO_EMA, ml.FUNDO_FORTE_Z)


DESC_ONDAS_AUTO = ("Como está na aba Indicadores: ajuste automático (ciclos {escala}). Onda subindo nos próximos "
                   "dias = COMPRA; descendo = VENDA. Para não ver o futuro, o ajuste é refeito mês a mês com os 3 anos "
                   "anteriores.")
DESC_ONDAS_TXT = ("Como está na aba Indicadores: valores do ondas.txt (inicio = {inicio}, largura = {largura_semanas:g} "
                  "semanas). Onda subindo = COMPRA; descendo = VENDA. ⚠️ Otimista: valores escolhidos olhando o passado. "
                  "Ajustes só nos sliders não entram — salve no ondas.txt.")


def ranking_metodos(chave: tuple, token: str, prazo: int, ml_: dict | None, ondas_cfg: dict,
                    progresso=None) -> pd.DataFrame:
    """`ml_` = resultado do modelo (itens 1–5) ou None com o ML desligado (aí o ML e a tabela de faixas ficam fora)."""
    prog = progresso or (lambda *_: None)
    d = preparar_dados(*chave)
    df = d["1D"][token]
    pr = df["Close"].astype(float)
    sinais, hoje = {}, {}
    rot = {1: "🟢 COMPRA", -1: "🔴 VENDA", 0: "⚪ —"}

    if ml_:
        prog(0.05, "Ranking: modelo de ML")
        res, hist = ml_["res"], ml_["hist"]
        h = hist[hist["token"] == token].set_index("Date")["sinal"].map({"COMPRA": 1, "VENDA": -1}).fillna(0)
        sinais["Modelo de ML"] = h
        ag = res["agora"]
        hoje["Modelo de ML"] = ag.loc[ag["Token"] == token, "Sinal"].iloc[0] if (ag["Token"] == token).any() else "—"

    prog(0.15, "Ranking: alertas de fundo e topo")
    rsi = df["RSI"]
    af = ml.alerta_fundo(pr, rsi).astype(int)
    aff = ml.alerta_fundo_forte(pr, rsi).astype(int)
    at = -ml.alerta_topo(pr, rsi, df["High"], df["Low"]).astype(int)
    sinais["Alerta de fundo (◆ laranja)"], sinais["Alerta de topo (◆ roxo)"] = af, at
    hoje["Alerta de fundo (◆ laranja)"], hoje["Alerta de topo (◆ roxo)"] = rot[int(af.iloc[-1])], rot[int(at.iloc[-1])]
    if aff.any():
        sinais["Alerta de fundo FORTE (◆ vermelho)"] = aff
        hoje["Alerta de fundo FORTE (◆ vermelho)"] = rot[int(aff.iloc[-1])]

    auto, ordem = ondas_cfg["auto"], ondas_cfg["ordem"]
    if auto:
        prog(0.25, "Ranking: ondas simétricas (refazendo o ajuste automático mês a mês)")
        on = historico_ondas_auto(pr, prazo, ordem=ordem)
        fit = ajustar_ondas(pr.resample("W-MON", label="left", closed="left").last().dropna()
                            .loc[pr.index[-1] - pd.Timedelta(weeks=156):], ordem)
        hoje["Ondas simétricas"] = (rot[int(sinal_ondas(pr.index[-1:], fit["inicio"], fit["largura_semanas"],
                                                        prazo)[0])] if fit else "—")
    else:
        on = pd.Series(sinal_ondas(pr.index, ondas_cfg["inicio"], ondas_cfg["largura_semanas"], prazo), index=pr.index)
        hoje["Ondas simétricas"] = rot[int(on.iloc[-1])]
    sinais["Ondas simétricas"] = on

    prog(0.55, "Ranking: análise de linhas")
    tl = ln.tendencia_linhas(df)
    sinais["Análise de linhas"] = tl["sinal"]
    hoje["Análise de linhas"] = f"{rot[int(tl['sinal'].iloc[-1])]} ({texto_tendencia_linhas(tl)})"

    prog(0.7, "Ranking: Buy/Sell Ratio")
    rb = historico_ratio(df, d["1W"].get(token), d["1M"].get(token))
    if len(rb):                                       # precisa de ~200 dias de aquecimento
        sinais["Buy/Sell Ratio (diário)"] = rb["sinal"]
        hoje["Buy/Sell Ratio (diário)"] = f"{rot[int(rb['sinal'].iloc[-1])]} ({rb['ratio'].iloc[-1]:.2f})"

    if ml_:
        prog(0.8, "Ranking: tabela de faixas")
        base = ml_["base"]
        bt = base[base["token"] == token]
        sf = ml.sinal_faixas(bt, res["faixas"], res["cols_todas"], prazo) if len(bt) else pd.DataFrame()
        if len(sf):
            sinais["Tabela de faixas (item 5)"] = sf["sinal"]
        hoje["Tabela de faixas (item 5)"] = (f"{rot[int(sf['sinal'].iloc[-1])]} ({100 * sf['media'].iloc[-1]:.0f}% vs "
                                             f"{100 * sf['normal'].iloc[-1]:.0f}%)" if len(sf) else "—")

    sinais["Sempre comprar (referência)"] = pd.Series(1, index=pr.index)
    hoje["Sempre comprar (referência)"] = "🟢 COMPRA"

    linhas_ = []
    for i, (nome, s) in enumerate(sinais.items()):
        prog(0.85 + 0.15 * i / len(sinais), f"Ranking: avaliando {nome}")
        a = ml.avaliar_metodo(s, pr, prazo)
        if not a.get("n"):
            continue
        desc = DESC_METODOS.get(nome, "")
        if nome == "Ondas simétricas":
            desc = (DESC_ONDAS_AUTO.format(escala={6: "curtos", 10: "médios", 13: "longos"}.get(ordem, ordem))
                    if auto else DESC_ONDAS_TXT.format(**ondas_cfg))
        s_ = s.reindex(pr.index).fillna(0)
        ult = lambda lado: (s_.index[s_.to_numpy() == lado].max().date() if (s_ == lado).any() else None)  # noqa: E731
        linhas_.append({
            "Método": nome, "Sinal hoje": hoje.get(nome, "—"),
            "_hoje": int(np.sign(s_.iloc[-1])) if len(s_) else 0,          # numérico, para a pontuação dos ativos
            "Última COMPRA": ult(1), "Última VENDA": ult(-1),
            "% de acertos": 100 * a["acerto"], "Acaso %": 100 * a["acaso"],
            "Acima do acaso (p.p.)": 100 * a["vantagem"],
            "COMPRA → subiu %": 100 * a["subiu_compra"], "VENDA → caiu %": 100 * a["caiu_venda"],
            "Dias com sinal": a["n"], "% do tempo": 100 * a["cobertura"],
            "Anos acima do acaso": f"{a['anos_bons']}/{a['anos']}", "Desde": a["inicio"].date(),
            "Como lê": desc})
    prog(1.0, "Pronto")
    return pd.DataFrame(linhas_)


def fig_ranking(tab: pd.DataFrame) -> go.Figure:
    t = tab.iloc[::-1]
    cores = ["#888" if "referência" in m else ("#26a69a" if v > 0 else "#ef5350")
             for m, v in zip(t["Método"], t["Acima do acaso (p.p.)"])]
    fig = go.Figure(go.Bar(
        x=t["Acima do acaso (p.p.)"], y=t["Método"], orientation="h", marker_color=cores,
        text=[f"{a:.0f}% de acertos (acaso {b:.0f}%)" for a, b in zip(t["% de acertos"], t["Acaso %"])],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>%{x:+.1f} p.p. acima do acaso<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="#bbb", width=1))
    lim = max(5.0, float(t["Acima do acaso (p.p.)"].abs().max()) * 1.9)
    fig.update_layout(template="plotly_dark", height=60 + 46 * len(t), xaxis=dict(range=[-lim, lim],
                      title="pontos percentuais acima (ou abaixo) do acaso"), margin=dict(l=10, r=20, t=20, b=40))
    return fig


AJUDA_PRAZO = """
Quantos dias à frente o modelo olha.

- **7 dias**: movimentos curtos — mais casos para aprender, mas muito ruído.
- **14 dias**: meio-termo (padrão).
- **30 dias**: tendência — menos ruído, mas menos casos independentes no histórico (e o mercado muda de
  fase no meio do caminho).

Referência: em cripto o preço sobe em ~50–55% dos períodos de 14 dias, então acertar 60% já é muito.
"""

AJUDA_MODELO = """
**Padrão: Regressão logística.** Refazendo o passado do BTC (2017–2026), foi o único modelo que ficou acima
do acaso nos dois prazos e nas duas metades do período (cerca de +3 a +7 p.p.). Gradient Boosting e Random
Forest acertam mais na validação, mas "decoram" a tendência mais recente e erram nas viradas (Random Forest
em 30 dias: −5 p.p.).

**Automático**: treina os três abaixo, cada um em três níveis de complexidade, e fica com o
que teve o **melhor AUC (compra × venda) ou a melhor correlação de Spearman (variação %) na validação**
(dados que ele não viu).

**Gradient Boosting** — 100 árvores de decisão pequenas, treinadas em sequência: cada uma corrige o erro
das anteriores. Bom para regras combinadas como *"RSI < 30 E Medo & Ganância < 25"*. Aceita dado faltando.

**Random Forest** — 100 árvores independentes, cada uma com uma amostra diferente dos dados; a resposta é a
média. Mais estável, decora menos o passado.

**Regressão logística / linear (Ridge)** — soma ponderada dos indicadores (cada um ganha um peso, positivo
ou negativo). O mais simples e fácil de interpretar, mas não capta combinações.

Complexidade: nas árvores, quantos níveis de perguntas cada árvore pode fazer (1 nível = uma pergunta só,
tipo *"RSI < 35?"*); no linear, quanta liberdade os pesos têm. Mais complexo aprende mais detalhes — e
também mais ruído.
"""

AJUDA_MOEDAS = """
**Todos juntos** (recomendado): um único modelo aprende com o histórico de todos os ativos do mercado
selecionado na barra lateral (cripto, B3 ou EUA; stablecoins ficam de fora). São muito mais exemplos e
vários ciclos de alta e baixa, então o padrão encontrado tende a ser mais confiável.

**Só o ativo selecionado**: aprende apenas com o histórico dele. Pode captar um comportamento próprio do
ativo, mas com poucos dados o risco de achar padrão por acaso é bem maior.
"""

AJUDA_GRUPOS = """
Grupos que o modelo pode usar (ele mesmo testa quais combinações ajudam):

- **RSI**: RSI diário (14 candles diários), RSI semanal (14 candles semanais, último candle já fechado) e
  divergência RSI × preço nos últimos 10 dias.
- **Médias móveis**: distância do preço às MME 50 e 200, MME 50 vs MME 200 (cruzamento), Pi Cycle
  (MME 111 ÷ 2×MME 350).
- **Volume**: volume do dia ÷ média de 20 dias; pico de volume (> 2× a média) nos últimos 3 dias, com alta
  ou com queda.
- **Medo & Ganância**: valor do índice (0–100) e quanto ele mudou em 7 dias (dados desde 2018).
- **LTA / LTB**: distância do preço à LTA e à LTB mais recentes (retas pelos 2 últimos fundos/topos) e
  distância do RSI às LTA/LTB do próprio RSI.
- **MACD**: MACD (12, 26) e MACD − linha de sinal (9), em % do preço; cruzamento do MACD nos últimos 5 dias.
- **OBV / fluxo**: OBV vs as médias de 20 e 50 dias (em desvios típicos) e o Chaikin Money Flow de 20 dias
  (compra ou venda pesando o volume pela posição do fechamento no candle).

Já vêm marcados os grupos que melhor funcionaram no BTC refazendo o passado (Médias móveis, LTA / LTB e
OBV / fluxo: cerca de +4 a +6 p.p. acima do acaso em 14 e 30 dias). Com todos os grupos marcados ficou em
+1 a +2 p.p. — indicador a mais vira ruído. (Medo & Ganância não entrou nesse teste: o índice só existe
desde 2018.)
"""

AJUDA_NOTA = {
    "cls": "AUC (área sob a curva ROC): a chance de o modelo dar uma probabilidade de alta maior para um dia "
           "que de fato subiu do que para um dia que caiu. 0,50 = cara ou coroa; 1,00 = perfeito. "
           "Em mercado, 0,53–0,56 já é considerado bom.",
    "reg": "Correlação de Spearman entre a variação prevista e a real: mede se o modelo ORDENA bem quem vai "
           "subir mais (usa a posição no ranking, não o valor, então não é distorcida por dias extremos). "
           "0 = acaso; 1 = ordem perfeita; negativo = ordem invertida. Em mercado, 0,05–0,10 já é bom.",
}


def _legenda_alertas(h_tok: pd.DataFrame, token: str, prazo: int):
    """Resumo do alerta de fundo (◆ laranja) e de topo (◆ roxo) no histórico do ativo."""
    if len(h_tok) > 200:
        pr = h_tok["preco"]
        av = ml.avaliar_alerta(pr, ml.alerta_fundo(pr, h_tok["rsi"]))
        avf = ml.avaliar_alerta(pr, ml.alerta_fundo_forte(pr, h_tok["rsi"]))
        at = ml.avaliar_alerta(pr, ml.alerta_topo(pr, h_tok["rsi"], h_tok.get("maxima"), h_tok.get("minima")), "topo")
        at_r = ml.avaliar_metodo(-ml.alerta_topo(pr, h_tok["rsi"], h_tok.get("maxima"), h_tok.get("minima"))
                                 .astype(int), pr, prazo)
        linhas_ = []
        if av["n"]:
            linhas_.append(
                f"🔶 **Alerta de fundo (em tempo real, sem usar o futuro):** RSI diário < {ml.ALERTA_RSI} e queda desde a "
                f"máxima de 90 dias de pelo menos {ml.ALERTA_Z:g} desvios típicos de 30 dias de {token} (a régua se "
                f"adapta à volatilidade). Neste período: **{av['n']} alertas**, **{av['acertos']}** seguidos de um fundo "
                f"em até 6 semanas; {av['cobertos']} dos {av['alvos']} fundos marcados tiveram alerta. Mas em "
                f"**{100 * av['caiu10']:.0f}%** das vezes o preço ainda caiu mais 10% ou mais antes de virar, e 60 dias "
                f"depois a variação mediana foi {100 * av['var_med']:+.0f}% (num dia qualquer: {100 * av['base_med']:+.0f}%). "
                "Ele acha **fundos locais**, inclusive os de cada perna de uma queda longa — não diz se a queda maior acabou."
                + (f" **◆ vermelho = alerta FORTE** (queda de ≥ {ml.FUNDO_FORTE_Z:g} desvios): {avf['n']} alertas, "
                   f"{avf['acertos']} seguidos de fundo; pega menos fundos ({avf['cobertos']} dos {avf['alvos']}), mas "
                   "acerta mais." if avf["n"] else ""))
        if at["n"]:
            linhas_.append(
                f"🟣 **Alerta de topo (perda de força):** divergência de baixa RSI × preço, **ou** o preço fechando abaixo "
                f"da EMA{ml.TOPO_EMA} depois de uma alta forte (≥ {ml.TOPO_Z:g} desvios nos {ml.TOPO_DIAS} dias anteriores). Ele não "
                "tenta vender a euforia — espera o impulso quebrar. Neste período: "
                f"**{at['n']} alertas**, {at['acertos']} perto de um topo marcado; {at['cobertos']} dos {at['alvos']} topos "
                "tiveram alerta"
                + (f"; o preço estava mais baixo {prazo} dias depois em **{100 * at_r['acerto']:.0f}%** dos dias de alerta "
                   f"(acaso {100 * at_r['acaso']:.0f}%)" if at_r.get("n") else "")
                + ". Muitos alertas são correções no meio da alta, não o topo do ciclo.")
        for l_ in linhas_:
            st.caption(l_)


def _analise_ml(chave: tuple, token: str, prazo: int) -> dict | None:
    """Itens 1 a 5 (o modelo de Machine Learning). Só roda com a opção ligada no topo da aba."""
    st.caption("Treina um modelo com **RSI, médias móveis, volume, Medo & Ganância, LTA/LTB, MACD e OBV** "
               "(candles diários) e confere se ele acerta em meses que **não viu**. O treino sempre deixa de fora o "
               f"**último ano** ({ml.EMBARGO} dias): o ano mais recente fazia o modelo só repetir a tendência que acabou "
               "de acontecer e errar nas viradas. Passe o mouse no ❔ de cada opção para detalhes.")
    c3, c4 = st.columns([1.2, 1.1])
    tarefa = "cls"                       # só compra × venda (a regressão saiu: muita informação e pouco acerto)
    opcoes_mod = ["auto"] + list(ml.FAMILIAS)
    familia = c3.selectbox("Modelo", opcoes_mod, help=AJUDA_MODELO, index=opcoes_mod.index("lin"),
                           format_func=lambda f: "Automático (melhor na validação)" if f == "auto"
                           else ml.nome_modelo(f, tarefa))
    so_tok = c4.radio("Ativos", ["Todos juntos", f"Só {token}"], help=AJUDA_MOEDAS)
    cripto = len(chave) < 4 or chave[3] == "Cripto"
    grupos_ok = ml.GRUPOS if cripto else [g for g in ml.GRUPOS if g != "Medo & Ganância"]
    grupos = st.multiselect("Indicadores que o modelo pode usar", grupos_ok, help=AJUDA_GRUPOS,
                            default=[g for g in ml.GRUPOS_PADRAO if g in grupos_ok])
    if not cripto:
        st.caption("O índice de Medo & Ganância (alternative.me) é do mercado cripto, então não entra para ações.")
    if not grupos:
        st.info("Escolha pelo menos um grupo de indicadores.")
        return None
    so_token = None if so_tok == "Todos juntos" else token
    barra = st.progress(0.0, "Montando os indicadores...")
    try:
        res = analisar_ml(chave, prazo, so_token, tuple(grupos), tarefa, familia,
                          progresso=lambda f, t: barra.progress(min(f, 1.0), t))
    except ValueError as e:
        barra.empty()
        st.error(str(e))
        return None
    barra.empty()
    base = base_indicadores(chave, prazo, so_token)
    t = res["teste"]
    cls = tarefa == "cls"
    if res["sem_dado"]:
        st.warning("Sem dados para: " + ", ".join(ml.NOME[c] for c in res["sem_dado"]) +
                   ". (O Medo & Ganância vem de api.alternative.me; se o site não respondeu, ele fica de fora.)")

    # ---------------- modelo usado
    fam = res["fam"]
    st.info(f"**Modelo usado: {res['modelo']}** ({ml.descricao_complexidade(fam, res['valor'])}) · prevendo "
            f"**{ml.TAREFAS[tarefa].lower()}** em **{prazo} dias** · indicadores: "
            f"**{' + '.join(res['grupos_escolhidos']) if res['grupos_escolhidos'] else ' + '.join(grupos)}**"
            f"{' (escolhidos pelo ML)' if res['grupos_escolhidos'] else ''}.\n\n{ml.FAMILIAS[fam]['desc']}")

    # ---------------- 1. dá para confiar? (refazendo o passado mês a mês)
    st.subheader("1 · Ele acertou em meses que não viu?")
    barra_h = st.progress(0.0, "Refazendo o passado...")
    hist = historico_ml(chave, prazo, so_token, tuple(grupos), tarefa, familia, res,
                        progresso=lambda f, t_: barra_h.progress(min(f, 1.0), t_))
    barra_h.empty()
    r_tok = ml.resumo_historico(hist, tarefa, token)
    r_all = ml.resumo_historico(hist, tarefa) if so_token is None else r_tok
    r_uso = r_tok if r_tok and r_tok.get("n", 0) >= 300 else r_all
    ok_, fraco = ml.sinal_confiavel(r_uso, tarefa)
    ok_all, _ = ml.sinal_confiavel(r_all, tarefa)
    met = ml.METRICA[tarefa]
    st.caption(f"Para cada mês do passado (a partir de 2 anos de dados), o modelo foi **treinado de novo só com o que já se sabia naquela "
               f"data** e deu o sinal de cada dia seguinte — exatamente como a 'Leitura de agora' faz hoje. "
               f"Depois comparamos com o que o preço fez em {prazo} dias.")
    if not r_uso:
        st.warning("Histórico curto demais para refazer o passado.")
    else:
        if ok_:
            st.success(f"✅ Refazendo o passado, o sinal de {token} acertou mais que o acaso. Ainda é um sinal fraco "
                       "(mercado é muito aleatório): use como filtro junto com sua análise, não como gatilho.")
        elif fraco:
            st.warning("🟡 Sinal fraco: refazendo o passado, ficou só um pouco acima do acaso. Use com cautela.")
        else:
            st.error(f"❌ Refazendo o passado, o sinal de {token} **não acertou mais que o acaso** — por isso a "
                     "'Leitura de agora' (item 5) mostra *sem sinal confiável*. As tabelas e o cruzamento de "
                     "indicadores continuam valendo como estatística histórica.")
        m1, m2, m3, m4 = st.columns(4)
        anos_txt = " · ".join(f"{a}: {v:.2f}" for a, v in r_uso["por_ano"].items())
        m1.metric(f"{met} refazendo o passado", f"{r_uso['nota']:.3f}", f"acaso = {ml.ACASO[tarefa]:.2f}",
                  delta_color="off", help=AJUDA_NOTA[tarefa] + f"\n\nAno a ano: {anos_txt}")
        if cls:
            m2.metric("Quando deu 🟢 COMPRA, subiu", f"{100 * r_uso['subiu_COMPRA']:.0f}%",
                      f"{100 * (r_uso['subiu_COMPRA'] - r_uso['subiu_todos']):+.0f} p.p. vs um dia qualquer",
                      help=f"{r_uso['n_COMPRA']} dias com COMPRA. Num dia qualquer subiu {100 * r_uso['subiu_todos']:.0f}%.")
            m3.metric("Quando deu 🔴 VENDA, caiu", f"{100 * (1 - r_uso['subiu_VENDA']):.0f}%",
                      f"{100 * (r_uso['subiu_todos'] - r_uso['subiu_VENDA']):+.0f} p.p. vs um dia qualquer",
                      help=f"{r_uso['n_VENDA']} dias com VENDA. Num dia qualquer caiu {100 * (1 - r_uso['subiu_todos']):.0f}%.")
        else:
            m2.metric("Previu alta → variação média", f"{100 * r_uso['ret_COMPRA']:+.1f}%")
            m3.metric("Previu queda → variação média", f"{100 * r_uso['ret_VENDA']:+.1f}%")
        fz = r_uso["fases"]
        if "fundo" in fz.index and "topo" in fz.index:
            m4.metric("Perto de fundos, deu VENDA", f"{fz.loc['fundo'].get('VENDA', 0):.0f}% dos dias",
                      f"perto de topos deu COMPRA: {fz.loc['topo'].get('COMPRA', 0):.0f}%", delta_color="off",
                      help="Fundos e topos = mínimos e máximos de ±10 semanas (só dá para saber depois); conta os dias "
                           "a até 3 semanas deles. Num modelo que acerta viradas, esses números seriam baixos.")
        preco_atual(preparar_dados(*chave)["1D"].get(token), token,
                    MERCADOS.get(chave[3], "USD") if len(chave) > 3 else "USD")
        st.plotly_chart(ml.fig_historico(hist, token, tarefa, MERCADOS.get(chave[3], "USD") if len(chave) > 3 else "USD"),
                        theme=None)
        h_tok = hist[hist["token"] == token].set_index("Date")
        _legenda_alertas(h_tok, token, prazo)
        tend = r_uso.get("segue_tendencia", np.nan)
        if not np.isnan(tend) and tend >= 0.2 and r_uso["nota"] < ml.ACASO[tarefa] + 0.01:
            st.info(f"🔁 **Por que ele erra nas viradas:** a previsão acompanha o que o preço **acabou de fazer** "
                    f"(correlação de {tend:+.2f} com a variação dos últimos {prazo} dias). Os exemplos mais "
                    "parecidos com o dia de hoje são os dias recentes, e o rótulo deles é o movimento que acabou de "
                    "acontecer — depois de meses de queda, ele 'aprendeu' que esse cenário cai; depois de meses de alta, "
                    "que sobe. Nos fundos e topos é exatamente quando isso se inverte.")
    with st.expander("Teste único (o modelo treinado uma vez e testado nos últimos 20% das datas)"):
        c1_, c2_, c3_ = st.columns(3)
        c1_.metric(f"{met} no teste", f"{t['nota']:.3f}", f"{t['nota'] - t['nota_val']:+.3f} vs validação",
                   delta_color="off")
        if cls:
            c2_.metric("COMPRA → subiu", "—" if np.isnan(t["acerto_compra"]) else f"{100 * t['acerto_compra']:.0f}%")
            c3_.metric("VENDA → caiu", "—" if np.isnan(t["acerto_venda"]) else f"{100 * t['acerto_venda']:.0f}%")
        else:
            c2_.metric("20% maiores previsões", f"{100 * t['ret_top']:+.1f}%")
            c3_.metric("20% menores previsões", f"{100 * t['ret_fundo']:+.1f}%")
        st.caption(f"Teste: {t['inicio']:%d/%m/%Y} a {t['fim']:%d/%m/%Y} · {t['n']:,} dias-ativo · {res['n_tokens']} "
                   "ativo(s). Este teste usa um modelo só, treinado uma vez, e costuma parecer melhor que o histórico "
                   "acima: o modelo foi escolhido entre 9 opções olhando a validação, e ele não sofre com o efeito de "
                   "'aprender o movimento mais recente', que é justamente o que pesa no sinal de hoje.")

    with st.expander("Como o modelo e os indicadores foram escolhidos (treino × validação)"):
        st.markdown(
            "**Passo 1 — indicadores.** Começa sem nada e adiciona um grupo por vez (com o Gradient Boosting mais "
            f"simples), só se o **{ml.METRICA[tarefa]} de validação** melhorar mais que o ruído. Validação = média de 4 janelas de "
            "tempo do período anterior ao teste, sempre treinando com o passado e testando no futuro.")
        if not res["selecao"].empty:
            st.dataframe(res["selecao"], hide_index=True, column_config={
                c: st.column_config.NumberColumn(format="%.3f") for c in (f"{ml.METRICA[tarefa]} validação", "± erro padrão")})
        else:
            st.caption("Nenhum grupo melhorou a validação sozinho; o modelo usa todos os escolhidos por você.")
        st.markdown(
            "**Passo 2 — modelo e complexidade.** Cada modelo é testado em 3 níveis de complexidade. "
            "**Treino muito acima da validação = decorou o passado (overfitting); os dois baixos = simples demais "
            "(underfitting).** Em cada modelo fica o nível mais simples que não perde para os outros; entre os "
            "modelos, vence a maior validação (⭐).")
        cmp_ = res["comparacao"]
        st.plotly_chart(ml.fig_comparacao(cmp_, tarefa), theme=None)
        met = ml.METRICA[tarefa]
        ren = {"Treino": f"{met} treino", "Validação": f"{met} validação", "Pior janela": f"{met} pior janela"}
        st.dataframe(cmp_[["Modelo", "Complexidade", "Treino", "Validação", "± erro padrão", "Pior janela", "Escolhido"]]
                     .rename(columns=ren), hide_index=True,
                     column_config={c: st.column_config.NumberColumn(format="%.3f")
                                    for c in list(ren.values()) + ["± erro padrão"]})

    # ---------------- 2. o que pesa e como
    st.subheader("2 · O que mais pesa e para que lado")
    a, b = st.columns([2, 3])
    a.plotly_chart(ml.fig_importancia(res["importancia"], tarefa), theme=None)
    a.caption(f"Peso = quanto o {ml.METRICA[tarefa]} cai quando o indicador é embaralhado (importância por "
              "permutação, medida no teste). Zero ou negativo = não ajudou.")
    ref = 100 * (t["subiu_todos"] if cls else t["ret_todos"])
    b.plotly_chart(ml.fig_efeitos(res, ref=ref), theme=None)
    b.caption(("Como a chance de alta muda" if cls else "Como a variação prevista muda") +
              " conforme cada indicador (os outros fixos). Pontilhado = média de todos os dias. "
              "Curva quase reta = o indicador pouco muda a previsão.")

    # ---------------- 3. combinações históricas
    st.subheader("3 · Combinações de indicadores que mais funcionaram")
    at = res["comb"].attrs
    st.caption("Faixas de 1 ou 2 indicadores em que o preço " + ("subiu mais (ou menos) que o normal" if cls else
               "variou mais (ou menos) que o normal") + " no período **antes** do teste "
               f"(média: {100 * at.get('base_dev', np.nan):.0f}% de altas, {100 * at.get('ret_dev', np.nan):+.1f}% "
               "de variação) e o que aconteceu **no teste**. Como muitas combinações são testadas, algumas parecem "
               "boas por sorte: confie mais nas ✅. 'Dias' conta dias-ativo; dias seguidos se parecem, então o "
               "número real de casos independentes é menor.")
    ca, cb = st.tabs(["🟢 Favoráveis à compra", "🔴 Favoráveis à venda"])
    for aba_, lado in ((ca, "alta"), (cb, "queda")):
        with aba_:
            tab = ml.tabela_combinacoes(res["comb"], lado, tarefa, 15, res["cods_agora"])
            if tab.empty:
                st.info("Poucos dados para combinações.")
            else:
                st.dataframe(tab, hide_index=True)

    # ---------------- 4. explorar um par
    st.subheader("4 · Cruzar dois indicadores")
    opcoes = [c for c in res["cols_todas"] if c in res["faixas"]]
    if len(opcoes) >= 2:
        k1, k2, k3 = st.columns([2, 2, 1])
        pa = k1.selectbox("Linhas", opcoes, index=opcoes.index("rsi") if "rsi" in opcoes else 0,
                          format_func=ml.NOME.get)
        resto = [c for c in opcoes if c != pa]
        pref = "fg" if "fg" in resto else ("dist_ltb" if "dist_ltb" in resto else resto[0])
        pb = k2.selectbox("Colunas", resto, index=resto.index(pref), format_func=ml.NOME.get)
        per = k3.radio("Período", ["todo", "antes", "teste"],
                       help="'antes' = onde se procura o padrão; 'teste' = meses recentes, para ver se se repetiu.")
        mp = ml.mapa_par(base, res, pa, pb, per)
        cods = res["cods_agora"].get(token, {})
        ia, ib = cods.get(pa, -1), cods.get(pb, -1)
        if ia >= 0 and ib >= 0:
            n, sb, cb_ = mp["cnt"][ia, ib], mp["sub"][ia, ib], mp["cai"][ia, ib]
            onde = (f"**📍 {token} agora:** {ml.NOME[pa]} **{mp['rot_a'][ia]}** · {ml.NOME[pb]} **{mp['rot_b'][ib]}**")
            if n >= 30:
                st.info(f"{onde} → nesse cruzamento, {prazo} dias depois o preço **subiu {100 * sb / n:.0f}%** e "
                        f"**caiu {100 * cb_ / n:.0f}%** das vezes ({sb} / {cb_} dias); variação média "
                        f"{mp['ret_med'][ia, ib]:+.1f}%. Média geral: subiu {mp['base_pct']:.0f}%.")
            else:
                st.info(f"{onde} → poucos dias históricos nesse cruzamento ({n}) para tirar conclusão.")
        else:
            st.info(f"{token} está sem dado em um dos dois indicadores agora.")
        st.plotly_chart(ml.fig_mapa(mp, ml.NOME[pa], ml.NOME[pb], token, (ia, ib), mostrar_ret=not cls),
                        theme=None)
        st.caption(f"Cada célula: **▲ % de vezes que subiu / ▼ % que caiu** em {prazo} dias; embaixo, "
                   f"**dias de alta / dias de queda**" + (" e a variação média" if not cls else "") +
                   f". Verde = sobe mais que a média ({mp['base_pct']:.0f}%), vermelho = sobe menos. "
                   "Células com menos de 30 dias ficam sem cor.")

    # ---------------- 5. agora
    st.subheader("5 · Leitura de agora")
    ag = res["agora"].copy()
    if not ok_all:          # o modelo (de todos os ativos) não passou refazendo o passado: não mostra COMPRA/VENDA
        ag["Sinal do modelo"] = ag["Sinal"]
        ag["Sinal"] = "⚪ sem sinal confiável"
    linha = ag[ag["Token"] == token]
    if not linha.empty:
        r0 = linha.iloc[0]
        normal = 100 * (r_uso.get("subiu_todos", np.nan) if r_uso else np.nan)
        if cls:
            bruto = res["agora"].loc[res["agora"]["Token"] == token, "Sinal"].iloc[0]
            sinal_tok = bruto if ok_ or fraco else "⚪ sem sinal confiável"
            st.metric(f"{token} · sinal do modelo ({prazo} dias)", sinal_tok,
                      f"chance de alta {r0['Chance de alta %']:.0f}% · normal {normal:.0f}%", delta_color="off",
                      help=f"COMPRA se a chance ≥ {100 * res['limites_agora'][0]:.0f}%, VENDA se ≤ "
                           f"{100 * res['limites_agora'][1]:.0f}% (25% mais altas / mais baixas que o modelo costuma "
                           f"dar) — e só quando o item 1 aprova o modelo. 'Normal' = quanto {token} subiu num dia "
                           "qualquer do histórico refeito.")
            if not (ok_ or fraco):
                st.caption(f"Sem o filtro do item 1, o modelo diria **{bruto}** — mas refazendo o passado esse tipo "
                           "de sinal não acertou mais que o acaso, então não deve guiar decisão.")
        else:
            st.metric(f"{token} · variação prevista em {prazo} dias" + ("" if ok_ or fraco else " (sem acerto comprovado)"),
                      f"{r0['Variação prevista %']:+.1f}%", f"erro típico ±{100 * t['erro_tipico']:.0f} p.p.",
                      delta_color="off")
    a, b = st.columns([3, 2])
    with a:
        st.markdown(f"**{token}: cada indicador hoje e o que aconteceu historicamente nessa faixa**")
        lt = ml.leitura_token(res, base, token)
        if lt.empty:
            st.info(f"{token} não entra nesta análise (stablecoin ou histórico curto).")
        else:
            fmt = {c: st.column_config.NumberColumn(format="%.1f") for c in lt.columns if c.startswith("Subiu")}
            fmt |= {c: st.column_config.NumberColumn(format="%+.2f") for c in lt.columns if c.startswith("Variação")}
            st.dataframe(lt, hide_index=True, column_config=fmt)
            if so_token is None:
                st.caption(f"Com 'Todos juntos', compare as duas colunas: o mesmo cenário pode ter terminado bem "
                           f"diferente para {token} e para a média dos ativos (ex.: altcoins em tendência de baixa "
                           "costumam continuar caindo; o BTC, historicamente, se recuperou mais).")
    with b:
        st.markdown("**Todos os ativos**")
        num = {c: "{:.1f}" for c in ("Chance de alta %",) if c in ag} | {c: "{:+.2f}" for c in ("Variação prevista %",) if c in ag}
        st.dataframe(ag.style.apply(_destacar(token), axis=1).format(num), hide_index=True)
        st.caption("O sinal só aparece quando o item 1 aprova o modelo (refazendo o passado).")
    return {"res": res, "hist": hist, "base": base, "so_token": so_token, "grupos": tuple(grupos), "familia": familia}


def _ranking(chave: tuple, token: str, prazo: int, ml_: dict | None):
    # ---------------- 6. ranking dos métodos
    st.subheader("6 · Ranking dos métodos" if ml_ else "Ranking dos métodos")
    st.caption(f"Todos os sinais do app medidos com a **mesma régua**, em todo o passado de {token}: em cada dia em que "
               f"o método deu COMPRA ou VENDA, o preço {prazo} dias depois foi para o lado certo? Cada método só usa o que "
               "já se sabia naquele dia. **Acaso** = o que acertaria quem desse a mesma quantidade de compras e vendas em "
               "dias sorteados — como o preço sobe na maior parte do tempo, só comprar já acerta bastante, então a coluna "
               "que separa sorte de habilidade é **acima do acaso**." + ("" if ml_ else " O modelo de ML e a tabela de faixas "
               "entram quando o ML está ligado (no topo da aba)."))
    ondas_txt, _ = ler_ondas()
    auto_o, ordem_o = config_ondas()
    ondas_cfg = {**ondas_txt, "auto": auto_o, "ordem": ordem_o}
    memo = _memo_ml()
    cfg_ml = (ml_["so_token"], ml_["grupos"], ml_["familia"]) if ml_ else ("sem ML",)
    k_rank = ("rank", id(preparar_dados(*chave)), chave, token, prazo, cfg_ml,
              tuple(sorted(ondas_cfg.items())), VERSAO_METODOS)
    if k_rank not in memo:
        barra_r = st.progress(0.0, "Montando o ranking...")
        memo[k_rank] = ranking_metodos(chave, token, prazo, ml_, ondas_cfg,
                                       progresso=lambda f, t_: barra_r.progress(min(f, 1.0), t_))
        barra_r.empty()
    rank = memo[k_rank]
    if rank.empty:
        st.info("Histórico curto demais para comparar os métodos.")
        return
    crit = st.radio("Ordenar por", ["Acima do acaso (p.p.)", "% de acertos"], horizontal=True, key="ranking_ordem",
                    help="'% de acertos' favorece quem vive dando COMPRA num mercado que sobe. 'Acima do acaso' desconta "
                         "isso e mostra se o método realmente sabe a hora de comprar e de vender.")
    rank = rank.sort_values(crit, ascending=False, ignore_index=True)
    rank.insert(0, "#", np.arange(1, len(rank) + 1))
    st.plotly_chart(fig_ranking(rank), theme=None)
    num = {c: st.column_config.NumberColumn(format="%.1f") for c in
           ("% de acertos", "Acaso %", "COMPRA → subiu %", "VENDA → caiu %", "% do tempo")}
    num["Acima do acaso (p.p.)"] = st.column_config.NumberColumn(format="%+.1f")
    num["Como lê"] = st.column_config.TextColumn(width="large")
    num["Última COMPRA"] = st.column_config.DateColumn(format="DD/MM/YYYY", help="Último dia em que o método deu COMPRA")
    num["Última VENDA"] = st.column_config.DateColumn(format="DD/MM/YYYY", help="Último dia em que o método deu VENDA")
    num["Desde"] = st.column_config.DateColumn(format="DD/MM/YYYY", help="Primeiro dia com sinal (início da avaliação)")
    st.dataframe(rank.drop(columns="_hoje"), hide_index=True, column_config=num)
    fora = "referência" if auto_o else "referência|Ondas"      # o ondas.txt foi ajustado olhando o passado
    reais = rank[~rank["Método"].str.contains(fora)]
    melhor = reais.sort_values("Acima do acaso (p.p.)", ascending=False).iloc[0]
    if melhor["Acima do acaso (p.p.)"] >= 3:
        st.success(f"Em {token}, o método que mais acertou acima do acaso foi **{melhor['Método']}** "
                   f"({melhor['% de acertos']:.0f}% de acertos, {melhor['Acima do acaso (p.p.)']:+.1f} p.p.; acima do acaso em "
                   f"{melhor['Anos acima do acaso']} anos). Hoje ele diz: **{melhor['Sinal hoje']}**.")
    else:
        st.warning(f"Em {token}, nenhum método passou de 3 p.p. acima do acaso em {prazo} dias — o melhor foi "
                   f"**{melhor['Método']}** ({melhor['Acima do acaso (p.p.)']:+.1f} p.p.). Use os sinais como contexto, "
                   "não como gatilho.")
    st.caption("Dias seguidos com o mesmo sinal contam várias vezes e se parecem entre si, então diferenças de 1–2 p.p. "
               "são ruído. Confie mais em quem fica acima do acaso na maioria dos anos.")
    st.caption("➡️ A aba **Pontuação dos Ativos** junta todos esses métodos numa nota de compra/venda para cada ativo "
               "do mercado, com o ranking refeito no histórico de cada um.")


AJUDA_USAR_ML = """
**Desligado (padrão):** a aba mostra os alertas de fundo/topo, o **ranking dos métodos** (alertas, análise de
linhas, Buy/Sell Ratio, ondas) e a **pontuação dos ativos** — mais rápido.

**Ligado:** treina e valida o modelo de Machine Learning (itens 1 a 5), e ele (mais a tabela de faixas) entra no
ranking e na pontuação. Leva de alguns segundos a um minuto. Desde a última revisão (OBV, médias móveis e LTA/LTB,
treino sem o último ano) ele ficou acima do acaso no BTC (≈ +4 a +5 p.p. em 30 dias), mas ainda abaixo dos
alertas, da análise de linhas e do Buy/Sell Ratio.
"""


ICONE_SINAL = {1: "🟢", -1: "🔴", 0: "⚪"}
MIN_DIAS_PESO = 30          # método com menos dias de sinal que isso no histórico do ativo não ganha peso (pouco caso)
PESO_CAUTELA = 10.0         # somado ao divisor: com poucos métodos fracos (ex.: um só, de +1,5 p.p.) a pontuação não vai a ±100


def _ml_padrao(chave: tuple, prazo: int, progresso=None) -> dict | None:
    """O modelo de ML na configuração padrão da aba Indicadores × ML (todos os ativos juntos, grupos padrão,
    regressão logística), sem desenhar nada. Se a aba de ML já rodou com essa configuração, reaproveita."""
    cripto = len(chave) < 4 or chave[3] == "Cripto"
    grupos = tuple(g for g in ml.GRUPOS_PADRAO if cripto or g != "Medo & Ganância")
    try:
        res = analisar_ml(chave, prazo, None, grupos, "cls", "lin", progresso=progresso)
    except ValueError as e:
        st.warning(f"Não deu para treinar o ML: {e}")
        return None
    hist = historico_ml(chave, prazo, None, grupos, "cls", "lin", res, progresso=progresso)
    return {"res": res, "hist": hist, "base": base_indicadores(chave, prazo, None), "so_token": None,
            "grupos": grupos, "familia": "lin"}


@st.fragment
def aba_pontuacao(chave: tuple, token: str):
    ss = st.session_state                # o Streamlit apaga o estado dos widgets de abas fechadas: guarda uma cópia
    ss.setdefault("pont_prazo", ss.get("_pont_prazo", 30))
    ss.setdefault("pont_ml", ss.get("_pont_ml", False))
    c1, c2 = st.columns([0.8, 2.3])
    prazo = c1.selectbox("Prazo", [7, 14, 30], key="pont_prazo", format_func=lambda d: f"{d} dias", help=AJUDA_PRAZO)
    c2.write("")
    usar_ml = c2.toggle("🧠 Incluir o modelo de Machine Learning (mais lento)", key="pont_ml",
                        help="Treina o ML na configuração padrão da aba Indicadores × ML (todos os ativos juntos, grupos "
                             "padrão, regressão logística); ele e a tabela de faixas entram como mais dois métodos.")
    ss["_pont_prazo"], ss["_pont_ml"] = prazo, usar_ml
    ml_ = None
    if usar_ml:
        barra = st.progress(0.0, "Treinando o ML...")
        ml_ = _ml_padrao(chave, prazo, progresso=lambda f, t_: barra.progress(min(f, 1.0), t_))
        barra.empty()
    ondas_txt, _ = ler_ondas()
    auto_o, ordem_o = config_ondas()
    _pontuacao_ativos(chave, token, prazo, ml_, {**ondas_txt, "auto": auto_o, "ordem": ordem_o})


def _pontuacao_ativos(chave: tuple, token: str, prazo: int, ml_: dict | None, ondas_cfg: dict):
    """Para cada ativo: o ranking dos métodos refeito no histórico DELE; peso = pontos acima do acaso (só os
    positivos); pontuação = média dos sinais de hoje ponderada por esses pesos."""
    mercado = chave[3] if len(chave) > 3 else "Cripto"
    st.caption(f"Para cada ativo de **{mercado}** (painel lateral), o ranking dos métodos (aba Indicadores × ML) é refeito **no histórico do próprio "
               f"ativo** (prazo de {prazo} dias). Cada método que ficou acima do acaso **nele** ganha peso = quantos pontos "
               "acertou acima do acaso; os outros não entram. Pontuação = soma de (sinal de hoje × peso) ÷ (soma dos "
               f"pesos + {PESO_CAUTELA:g}) × 100, onde o sinal é 🟢 +1, ⚪ 0 ou 🔴 −1. Perto de **+100** = vários métodos "
               "fortes no ativo em compra; perto de **−100** = em venda; **0** = sem sinal ou dividido. O "
               f"+{PESO_CAUTELA:g} no divisor é uma cautela: um ativo em que só um método fraco funciona não chega a ±100.")
    toks = [t for t in preparar_dados(*chave)["1D"] if t not in ml.ESTAVEIS]
    memo = _memo_ml()
    cfg_ml = (ml_["so_token"], ml_["grupos"], ml_["familia"]) if ml_ else ("sem ML",)
    k = ("pontos", id(preparar_dados(*chave)), chave, prazo, tuple(sorted(ondas_cfg.items())), cfg_ml, VERSAO_METODOS)
    if k not in memo:
        def um(t):
            try:
                return t, ranking_metodos(chave, t, prazo, ml_, ondas_cfg)
            except Exception:                               # um ativo com dado ruim não derruba a lista
                return t, pd.DataFrame()
        barra = st.progress(0.0, f"Refazendo o ranking de cada ativo (0/{len(toks)})...")
        rankings = {}
        if fontes.NO_NAVEGADOR:
            for i, t in enumerate(toks):
                barra.progress(i / max(len(toks), 1), f"Refazendo o ranking de cada ativo: {t} ({i + 1}/{len(toks)})")
                rankings.update([um(t)])
        else:
            with ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 2))) as ex:
                for i, (t, r) in enumerate(ex.map(um, toks)):
                    rankings[t] = r
                    barra.progress((i + 1) / max(len(toks), 1), f"Refazendo o ranking de cada ativo ({i + 1}/{len(toks)})")
        barra.empty()
        memo[k] = rankings
    rankings = memo[k]
    # colunas dos métodos: do que mais funcionou (média entre os ativos) para o que menos
    todos = pd.concat([r for r in rankings.values() if r is not None and len(r)], ignore_index=True) \
        if any(r is not None and len(r) for r in rankings.values()) else pd.DataFrame(columns=["Método"])
    todos = todos[~todos["Método"].str.contains("referência")]
    nomes = list(todos.groupby("Método")["Acima do acaso (p.p.)"].mean().sort_values(ascending=False).index)
    linhas_ = []
    for t, rk in rankings.items():
        if rk is None or rk.empty:
            continue
        rk = rk[~rk["Método"].str.contains("referência")]
        ok = rk[(rk["Acima do acaso (p.p.)"] > 0) & (rk["Dias com sinal"] >= MIN_DIAS_PESO)]
        lin = {"Ativo": t}
        if ok.empty:
            lin |= {"Pontuação": 0.0, "Leitura": "⚪ nenhum método funciona", "Compra": 0, "Venda": 0}
        else:
            w, v = ok["Acima do acaso (p.p.)"].to_numpy(float), ok["_hoje"].to_numpy(float)
            pts = 100 * float((w * v).sum() / (w.sum() + PESO_CAUTELA))
            lin |= {"Pontuação": pts, "Leitura": "🟢 compra" if pts >= 25 else ("🔴 venda" if pts <= -25 else "⚪ neutro"),
                    "Compra": int((v > 0).sum()), "Venda": int((v < 0).sum())}
        lin["Métodos"] = f"{len(ok)}/{len(rk)}"
        lin["Melhor método no ativo"] = (f"{ok.sort_values('Acima do acaso (p.p.)').iloc[-1]['Método'].split(' (')[0]} "
                                        f"({ok['Acima do acaso (p.p.)'].max():+.1f})") if len(ok) else "—"
        por = rk.set_index("Método")
        for m in nomes:
            if m not in por.index:
                lin[m.split(" (")[0]] = ""                                    # não deu para calcular (histórico curto)
            elif m in set(ok["Método"]):
                lin[m.split(" (")[0]] = f"{ICONE_SINAL[int(por.loc[m, '_hoje'])]} {por.loc[m, 'Acima do acaso (p.p.)']:+.1f}"
            else:
                lin[m.split(" (")[0]] = "·"                                   # existe, mas não funciona neste ativo
        linhas_.append(lin)
    if not linhas_:
        st.info("Sem ativos com dados suficientes.")
        return
    tab = pd.DataFrame(linhas_).sort_values(["Pontuação", "Compra"], ascending=[False, False], ignore_index=True)
    tab.insert(0, "#", np.arange(1, len(tab) + 1))
    st.dataframe(tab.style.apply(_destacar(token, "Ativo"), axis=1), hide_index=True, column_config={
        "Pontuação": st.column_config.ProgressColumn(min_value=-100, max_value=100, format="%+.0f",
                                                     help="−100 = todos os métodos que funcionam no ativo em venda · "
                                                          "+100 = todos em compra"),
        "Compra": st.column_config.NumberColumn("🟢", help="Quantos dos métodos que contam estão em COMPRA hoje"),
        "Venda": st.column_config.NumberColumn("🔴", help="Quantos dos métodos que contam estão em VENDA hoje"),
        "Métodos": st.column_config.TextColumn(help="Quantos métodos ficaram acima do acaso NESTE ativo (e por isso "
                                                    "entram na pontuação) / quantos deu para avaliar no histórico dele")})
    st.caption(f"Em cada coluna de método: **🟢/⚪/🔴 + número** = sinal de hoje e o peso (pontos acima do acaso no ativo); "
               f"**·** = o método não funcionou nesse ativo (abaixo do acaso, ou menos de {MIN_DIAS_PESO} dias com sinal) e "
               "não entra; vazio = histórico curto demais para calcular. Os alertas ficam sem sinal quase sempre, então "
               "a pontuação raramente chega perto de ±100. É um resumo de hoje, não uma previsão garantida: com poucos "
               "anos de histórico, os pesos de um ativo são mais incertos.")


def _grafico_alertas(chave: tuple, token: str, prazo: int):
    """Preço com os alertas de fundo/topo (◆) sem precisar do modelo (o gráfico do item 1 já os mostra)."""
    df = preparar_dados(*chave)["1D"].get(token)
    if df is None or len(df) < 200:
        return
    st.subheader("Alertas de fundo e topo")
    h = pd.DataFrame({"Date": df.index, "token": token, "preco": df["Close"].to_numpy(float),
                      "rsi": df["RSI"].to_numpy(float), "maxima": df["High"].to_numpy(float),
                      "minima": df["Low"].to_numpy(float), "sinal": "", "p": np.nan})
    preco_atual(df, token, MERCADOS.get(chave[3], "USD") if len(chave) > 3 else "USD")
    st.plotly_chart(ml.fig_historico(h, token, "cls", MERCADOS.get(chave[3], "USD") if len(chave) > 3 else "USD"),
                    theme=None)
    _legenda_alertas(h.set_index("Date"), token, prazo)


@st.fragment
def aba_ml(chave: tuple, token: str):
    c2, c3 = st.columns([0.8, 2.3])
    prazo = c2.selectbox("Prazo", [7, 14, 30], index=2, format_func=lambda d: f"{d} dias", help=AJUDA_PRAZO)
    ss = st.session_state                # o Streamlit apaga o estado dos widgets de abas fechadas: guarda uma cópia
    ss.setdefault("usar_ml", ss.get("_usar_ml", False))
    c3.write("")
    usar_ml = c3.toggle("🧠 Incluir o modelo de Machine Learning (mais lento)", key="usar_ml", help=AJUDA_USAR_ML)
    ss["_usar_ml"] = usar_ml
    ml_ = None
    if usar_ml:
        ml_ = _analise_ml(chave, token, prazo)
    else:
        st.caption("O modelo de Machine Learning está **desligado** — ligue acima para treiná-lo e incluí-lo no ranking.")
        _grafico_alertas(chave, token, prazo)
    _ranking(chave, token, prazo, ml_)


# =========================================================================== #
# App
# =========================================================================== #

st.header("Cryptinho", anchor=False)

simbolos = ler_simbolos()

with st.sidebar:
    st.text("Para o rebalanceador de carteira:")
    up_ = st.file_uploader("Envie .txt da sua carteira", type=["txt"])
    txt_manual_ = st.text_area(
        "…ou cole aqui", height=150,
        placeholder="BTC 0.4\nETH 0.1\nITUB4 0.2\nNVDA 0.2\n@\nBTC, 0.1, 03/11/2025\nETH, 3, 01/11/2025\n"
                    "ITUB4, 200, 05/11/2025\nNVDA, 15, 05/11/2025\nBRL, 5000, 01/11/2025\nUSD, 1000, 01/11/2025")
    st.download_button("Baixar exemplo de carteira (.txt)", file_name="carteira_exemplo.txt", mime="text/plain",
                       data=("# Nas primeiras linhas, o ativo e a % desejada (o resto vira caixa); termine com '@'.\n"
                             "# Depois: ativo, unidades adquiridas, data. Pode misturar cripto (BTC), ações da B3\n"
                             "# (ITUB4, HGLG11), ações dos EUA (NVDA) e caixa (USD ou BRL).\n\n"
                             "BTC 0.4\nETH 0.1\nITUB4 0.2\nNVDA 0.2\n@\nBTC, 0.1, 03/11/2025\n"
                             "ETH, 3, 03/11/2025\nITUB4, 200, 05/11/2025\nNVDA, 15, 05/11/2025\n"
                             "BRL, 5000, 01/11/2025\nUSD, 1000, 01/11/2025\n"))
    st.divider()
    st.text("Para os indicadores:")
    mercado = st.radio("Mercado", list(MERCADOS), horizontal=True,
                       help="Cripto = lista do add_symbols.txt (cotação em USD). Ações B3 em BRL e Ações EUA em "
                            "USD = listas ACOES_B3 / ACOES_EUA no início do código.")
    so_recentes = st.checkbox(f"Só os últimos {ANOS_PADRAO} anos", value=True,
                              help=f"Marcado (padrão): baixa e analisa só os últimos {ANOS_PADRAO} anos de cada ativo — mais "
                                   "leve, principalmente para ações dos EUA, que têm dados desde os anos 60–80. "
                                   "Desmarcado: usa todo o histórico de todos os ativos (cripto quase não muda: o "
                                   "histórico delas é mais curto que isso).")
    periodo_dados = f"{ANOS_PADRAO}y" if so_recentes else "max"
    usar_ultima = st.checkbox("Usar última data", value=True, help="Se desmarcado, selecionar última data")
    data_maxima = None if usar_ultima else st.date_input(
        "Selecione última data", value=pd.Timestamp.today().date(), format="DD/MM/YYYY").isoformat()
    periodicidade = st.radio("Periodicidade", options=["1D", "1W"], index=0, horizontal=True,
                             help="Dados diários (1D) ou semanais (1W)")

lista = {"Cripto": simbolos, "Ações B3": ACOES_B3, "Ações EUA": ACOES_EUA}[mercado]
moeda = MERCADOS[mercado]
chave = (lista, periodo_dados, data_maxima, mercado)
dados = preparar_dados(*chave)
tokens = list(dados["1D"])
if not tokens:
    st.error(f"Nenhum dado baixado para {mercado}. Verifique a conexão ou a lista de símbolos." +
             (" Na versão web, ações vêm do retrato embutido no HTML: gere-o no PC com `python gerar_html.py`."
              if fontes.NO_NAVEGADOR else ""))
    st.stop()

with st.sidebar:
    token_sel = st.selectbox("Selecione o ativo", options=tokens, index=0)
    st.caption("Dados: " + fontes.descricao_fonte() +
               "".join(f"  \n{m}: {txt}" for m, txt in fontes.STATUS.items() if fontes.NO_NAVEGADOR))
    if st.button("🔄 Atualizar dados agora"):
        baixar_ohlc.clear()
        preparar_dados.clear()
        st.rerun()

top25, top50, top100 = faixas_volume(dados["1D"]) if mercado == "Cripto" else ([], [], [])

ABAS = ["Carteira", "Indicadores", "Análise por Linhas", "Indicadores × ML", "Pontuação dos Ativos"]
try:   # abas preguiçosas (Streamlit recente): só a aberta é executada
    tab_carteira, tab_indicadores, tab_linhas, tab_ml, tab_pontos = st.tabs(ABAS, on_change="rerun")
except TypeError:
    tab_carteira, tab_indicadores, tab_linhas, tab_ml, tab_pontos = st.tabs(ABAS)


def _aberta(tab) -> bool:
    return getattr(tab, "open", True) is not False


if _aberta(tab_indicadores):
    with tab_indicadores:
        sinais, resumo = analisar_sinais(chave, periodicidade)
        df_tok = dados[periodicidade][token_sel]
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader(f"📈 Gráficos {periodicidade}")
            ini, fim = periodo_grafico("periodo_indicadores", df_tok.index)
            preco_atual(dados["1D"].get(token_sel), token_sel, moeda)
            grafico_indicadores(df_tok.loc[ini:fim], token_sel, periodicidade, moeda)
            preco_atual(dados["1D"].get(token_sel), token_sel, moeda)
            grafico_semiciclos(dados["1W"][token_sel].loc[ini:fim], token_sel, moeda=moeda)
            slider_periodo("periodo_indicadores", df_tok.index)
        with col2:
            exibir_sinais(sinais, resumo, token_sel, periodicidade)
            if mercado == "Cripto":
                revisao_volume(token_sel, top50, top100)
                valor_fg, classe_fg = fear_and_greed()
                st.subheader("⚡ Fear & Greed Index")
                st.write(f"**Valor:** {valor_fg}")
                st.write(f"**Classificação:** {classe_fg}")
                links_uteis()
            else:
                st.caption(f"Preços em {moeda}. Ações só negociam em dias úteis: '7D' = 7 dias corridos.")

if _aberta(tab_linhas):
    with tab_linhas:
        aba_linhas(chave, periodicidade, token_sel, tokens)

if _aberta(tab_ml):
    with tab_ml:
        aba_ml(chave, token_sel)

if _aberta(tab_pontos):
    with tab_pontos:
        aba_pontuacao(chave, token_sel)

if _aberta(tab_carteira):
    with tab_carteira:
        st.header("🧮 Rebalanceador de Carteira")
        exibir_rebalanceamento(simbolos, up_, txt_manual_)
