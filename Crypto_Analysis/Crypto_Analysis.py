
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import StringIO
from typing import Callable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

import linhas as ln
import ml

st.set_page_config(page_title="CRYPTINHO", layout="wide")

# Mantém valores de widgets entre páginas (hack conhecido de apps multipágina).
# Botões de formulário são somente leitura: regravá-los quebra o st.form.
for _k in list(st.session_state.keys()):
    if not str(_k).startswith("FormSubmitter:"):
        st.session_state[_k] = st.session_state[_k]

PERIODO = "max"      # baixa todo o histórico (mais dados para o ML e para EMAs longas)
MESES_GRAFICO = 36   # período exibido por padrão nos gráficos (o mesmo do app original)
TTL = "30m"          # tempo de vida do cache de dados de mercado
PASTA = os.path.dirname(os.path.abspath(__file__))

# Outros mercados (conferidos no Yahoo Finance em 14/10/2026).
# EMBR3 virou EMBJ3 (nov/2025) e a JBS saiu da B3 como ação (JBSS3) e voltou como BDR (JBSS32).
ACOES_B3 = ("ITUB4", "BBAS3", "VALE3", "CMIN3", "EQTL3", "PRIO3", "NVDC34", "MSFT34", "GOGL34", "TOTS3",
            "MELI34", "RDOR3", "RADL3", "SLCE3", "JBSS32", "BTLG11", "HGLG11", "MULT3", "EMBJ3", "WEGE3",
            "MDIA3", "PETR4")
ACOES_EUA = ("JPM", "BAC", "FCX", "NEM", "NEE", "XOM", "NVDA", "MSFT", "GOOGL", "AAPL", "LLY", "UNH", "DE",
             "ADM", "PLD", "AMT", "LMT", "RTX")
MERCADOS = {"Cripto": "USD", "Ações B3": "BRL", "Ações EUA": "USD"}      # mercado -> moeda de cotação


def ticker_yahoo(simbolo: str, mercado: str) -> str:
    if mercado == "Ações B3":
        return f"{simbolo}.SA"
    if mercado == "Ações EUA":
        return simbolo
    return "BTC-USD" if simbolo == "USD" else f"{simbolo}-USD"


# Leitura da lista de símbolos

@st.cache_data(show_spinner=False)
def ler_simbolos(caminho: str = "add_symbols.txt") -> tuple[str, ...]:
    for c in (caminho, os.path.join(PASTA, caminho)):
        if os.path.exists(c):
            with open(c, encoding="utf-8") as f:
                txt = "\n".join(ln_ for ln_ in f if not ln_.strip().startswith("#"))
            simbolos = [s.upper() for s in re.split(r"[,;\s]+", txt) if s.strip()]
            return tuple(dict.fromkeys(simbolos))          # remove duplicatas, mantém ordem
    return ("BTC", "ETH")


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


@st.cache_data(ttl=TTL, show_spinner=False)
def baixar_ohlc(simbolos: tuple[str, ...], periodo: str, mercado: str = "Cripto") -> dict[str, pd.DataFrame]:
    """Um único download para todos os ativos do mercado."""
    tickers = sorted({ticker_yahoo(s, mercado) for s in simbolos})
    raw = yf.download(tickers, period=periodo, group_by="ticker", auto_adjust=False,
                      progress=False, threads=True)
    out = {}
    for s in simbolos:
        t = ticker_yahoo(s, mercado)
        if isinstance(raw.columns, pd.MultiIndex):
            if t not in raw.columns.get_level_values(0):
                continue
            df = raw[t]
        else:
            df = raw
        df = df[OHLCV].dropna(subset=["Close"]).astype(float)
        if df.empty:
            continue
        idx = pd.to_datetime(df.index)
        df.index = (idx.tz_localize(None) if idx.tz is not None else idx).rename("Date")
        if mercado == "Cripto" and s == "USD":   # USD medido em BTC: inverte e troca máxima/mínima
            df = pd.DataFrame({"Open": 1 / df["Open"], "High": 1 / df["Low"], "Low": 1 / df["High"],
                               "Close": 1 / df["Close"], "Volume": df["Volume"]}, index=df.index)
        out[s] = df
    return out


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

    # tokens em paralelo (as rotinas numéricas do pandas/numpy liberam o GIL)
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
    Regra("Compra", "🔽", lambda S: (S.EMA200 < S.EMA50) & (S.Close < S.EMA50) & (S.Close < S.EMA111)
          & (S.Close <= S.EMA200) & (S.Close >= S.EMA350),
          lambda r, P: f"({P}) Preço abaixo das EMAs 50, 111 e 200, mas acima do suporte da EMA350 ({f(r.EMA350)})"),
    Regra("Compra", "📉", lambda S: S.RSI_1D <= 30, lambda r, P: f"RSI 1D em sobrevenda ({r.RSI_1D:.2f})"),
    Regra("Compra", "📉", lambda S: S.RSI_1W <= 30, lambda r, P: f"RSI 1S em sobrevenda ({r.RSI_1W:.2f})"),
    Regra("Compra", "📉", lambda S: S.RSI_1M <= 30, lambda r, P: f"RSI 1M em sobrevenda ({r.RSI_1M:.2f})"),
    Regra("Compra", "📈", lambda S: S.MACD_Hist > 0, lambda r, P: f"({P}) MACD acima da linha de sinal → Bull"),
    Regra("Compra", "📊", lambda S: S.OBV > S.OBV_SMA20, lambda r, P: _obv_msg(r, P, True)),
    Regra("Compra", "📉", lambda S: S.Close < S.Lower_BB,
          lambda r, P: f"({P}) Preço abaixo da Bollinger inferior ({f(r.Lower_BB)}) → "
                       + ("contraindo → movimento forte se aproximando" if r.BB_Contraction_Status == "Contraindo"
                          else "expandindo → alta volatilidade")),
    Regra("Compra", "🔺", lambda S: (S["Fib_38.2%"] >= S.Close) & (S.Close >= S["Fib_61.8%"]),
          lambda r, P: f"({P}) Preço entre Fibo 38,2% ({f(r['Fib_38.2%'])}, compra arrojada) e "
                       f"61,8% ({f(r['Fib_61.8%'])}, golden pocket)"),
    Regra("Compra", "🟢", lambda S: S["Williams_%R"] <= -80,
          lambda r, P: f"({P}) Williams %R ({r['Williams_%R']:.2f}) ≤ -80 (sobrevenda)"),
    Regra("Compra", "🟢", lambda S: S.StochRSI <= 0.2,
          lambda r, P: f"({P}) StochRSI ({r.StochRSI:.2f}) ≤ 0,2 (sobrevenda)"),
    Regra("Compra", "🟢", lambda S: S.VolumeSpike.astype(bool) & (S.Close > S.Close_ant),
          lambda r, P: f"({P}) Pico de volume com preço subindo (acumulação)"),
    Regra("Compra", "🟢", lambda S: (S.ADX > 25) & (S["DI+"] > S["DI-"]),
          lambda r, P: f"({P}) ADX ({r.ADX:.2f}) > 25 e DI+ > DI- (tendência de alta)"),
    Regra("Compra", "🟢", lambda S: S.BullishDivergence.astype(bool),
          lambda r, P: f"({P}) Divergência de alta RSI x preço"),
    Regra("Compra", "🟢", lambda S: (S["DI+"] > S["DI-"]) & (S["DI+_ant"] < S["DI-_ant"]),
          lambda r, P: f"({P}) DI+ cruzou acima do DI- (mudança de tendência)"),
    Regra("Compra", "🟢", lambda S: (S["Williams_%R"] > -80) & (S["Williams_%R_ant"] <= -80),
          lambda r, P: f"({P}) Williams %R cruzando acima de -80 (saindo da sobrevenda)"),
    Regra("Compra", "🟢", lambda S: (S.StochRSI > 0.2) & (S.StochRSI_ant <= 0.2),
          lambda r, P: f"({P}) StochRSI cruzando acima de 0,2 (momentum positivo)"),
    # ------------------------------ VENDA ------------------------------- #
    Regra("Venda", "🔴", lambda S: S.EMA200 > S.EMA50,
          lambda r, P: f"({P}) EMA200 ({f(r.EMA200)}) > EMA50 ({f(r.EMA50)}) → Bear"),
    Regra("Venda", "🔴", lambda S: (S.Close > S.EMA50) & (S.Close > S.EMA111) & (S.Close > S.EMA200)
          & (S.Close > S.EMA350),
          lambda r, P: f"({P}) Preço acima das EMAs 50, 111, 200 e 350"),
    Regra("Venda", "🔴", lambda S: S.Close > S.EMA350x2,
          lambda r, P: f"({P}) Preço acima da EMA350×2 ({f(r.EMA350x2)}) → atenção ao cruzamento com EMA111 ({f(r.EMA111)})"),
    Regra("Venda", "⚠️", lambda S: S.EMA111 / S.EMA350x2 >= 0.9,
          lambda r, P: f"({P}) Pi Cycle Top em zona de alerta (EMA111 {f(r.EMA111)} x EMA350×2 {f(r.EMA350x2)})"),
    Regra("Venda", "📈", lambda S: S.RSI_1D >= 70, lambda r, P: f"RSI 1D em sobrecompra ({r.RSI_1D:.2f})"),
    Regra("Venda", "📈", lambda S: S.RSI_1W >= 70, lambda r, P: f"RSI 1S em sobrecompra ({r.RSI_1W:.2f})"),
    Regra("Venda", "📈", lambda S: S.RSI_1M >= 70, lambda r, P: f"RSI 1M em sobrecompra ({r.RSI_1M:.2f})"),
    Regra("Venda", "📉", lambda S: S.MACD_Hist < 0, lambda r, P: f"({P}) MACD abaixo da linha de sinal → Bear"),
    Regra("Venda", "🔻", lambda S: S.OBV < S.OBV_SMA20, lambda r, P: _obv_msg(r, P, False)),
    Regra("Venda", "📈", lambda S: S.Close > S.Upper_BB,
          lambda r, P: f"({P}) Preço acima da Bollinger superior ({f(r.Upper_BB)}) → "
                       + ("contraindo → movimento forte se aproximando" if r.BB_Contraction_Status == "Contraindo"
                          else "expandindo → alta volatilidade")),
    Regra("Venda", "📉", lambda S: S["Fib_78.6%"] > S.Close,
          lambda r, P: f"({P}) Preço abaixo da retração de Fibo 78,6% ({f(r['Fib_78.6%'])})"),
    Regra("Venda", "📉", lambda S: (S["Ext_161.8%"] >= S.Close) & (S.Close >= S["Ext_100%"]),
          lambda r, P: f"({P}) Preço entre as extensões de Fibo 100% ({f(r['Ext_100%'])}) e 161,8% ({f(r['Ext_161.8%'])})"),
    Regra("Venda", "🔴", lambda S: S["Williams_%R"] >= -20,
          lambda r, P: f"({P}) Williams %R ({r['Williams_%R']:.2f}) ≥ -20 (sobrecompra)"),
    Regra("Venda", "🔴", lambda S: S.StochRSI >= 0.8,
          lambda r, P: f"({P}) StochRSI ({r.StochRSI:.2f}) ≥ 0,8 (sobrecompra)"),
    Regra("Venda", "🔴", lambda S: S.VolumeSpike.astype(bool) & (S.Close < S.Close_ant),
          lambda r, P: f"({P}) Pico de volume com preço caindo (distribuição)"),
    Regra("Venda", "🔴", lambda S: (S.ADX > 25) & (S["DI-"] > S["DI+"]),
          lambda r, P: f"({P}) ADX ({r.ADX:.2f}) > 25 e DI- > DI+ (tendência de baixa)"),
    Regra("Venda", "🔴", lambda S: (S["DI-"] > S["DI+"]) & (S["DI-_ant"] < S["DI+_ant"]),
          lambda r, P: f"({P}) DI- cruzou acima do DI+ (mudança de tendência)"),
    Regra("Venda", "🔴", lambda S: (S["Williams_%R"] < -20) & (S["Williams_%R_ant"] >= -20),
          lambda r, P: f"({P}) Williams %R cruzando abaixo de -20 (saindo da sobrecompra)"),
    Regra("Venda", "🔴", lambda S: (S.StochRSI < 0.8) & (S.StochRSI_ant >= 0.8),
          lambda r, P: f"({P}) StochRSI cruzando abaixo de 0,8 (momentum negativo)"),
    Regra("Venda", "🔴", lambda S: S.BearishDivergence.astype(bool),
          lambda r, P: f"({P}) Divergência de baixa RSI x preço"),
]
TOTAL = {lado: sum(r.lado == lado for r in REGRAS) for lado in ("Compra", "Venda")}


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


@st.cache_data(ttl=TTL, show_spinner=False)
def analisar_sinais(chave: tuple, freq: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna (sinais, resumo). ``chave`` = (simbolos, periodo, data_maxima)."""
    dados = preparar_dados(*chave)
    S = _snapshot(dados, freq)
    P = freq.lower()
    registros = []
    cont = {lado: pd.Series(0, index=S.index) for lado in TOTAL}
    for regra in REGRAS:
        mask = regra.cond(S).fillna(False).astype(bool)
        cont[regra.lado] += mask.astype(int)
        for tok, r in S[mask].iterrows():                       # só os tokens que dispararam
            registros.append((tok, regra.lado, regra.icone, regra.msg(r, P), freq))
    sinais = pd.DataFrame(registros, columns=["Token", "Lado", " ", "Sinal", "Período"])

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

def _destacar(token_sel: str):
    def _f(row):
        return ["background-color: #ff4d4d55; font-weight: bold;" if row["Token"] == token_sel else ""] * len(row)
    return _f


def exibir_sinais(sinais: pd.DataFrame, resumo: pd.DataFrame, token_sel: str, freq: str):
    st.subheader(f"📈 Indicadores {freq}")
    t1, t2, t3, t4 = st.tabs([f"Sinais {token_sel}", "Variação 7D", "Buy/Sell Ratio", "Compra × Venda"])
    with t1:
        st.caption("Sinais agregados (compra e venda)")
        s = sinais[sinais["Token"] == token_sel].sort_values(["Lado", " "])
        st.dataframe(s, hide_index=True)
    with t2:
        st.caption("Variação de preço em 7 dias (diário)")
        v = resumo[["Token", "Preço atual (1D)", "Preço 7D atrás (1D)", "Variação (7D) %"]]
        st.dataframe(v.sort_values("Variação (7D) %", ascending=False).style.apply(_destacar(token_sel), axis=1),
                     hide_index=True)
    with t3:
        st.caption("Buy/Sell Ratio (quanto maior, mais 'atrativo' o conjunto de sinais do token)")
        r = resumo[["Token", "Buy/Sell Ratio"]].sort_values("Buy/Sell Ratio", ascending=False)
        st.dataframe(r.style.apply(_destacar(token_sel), axis=1), hide_index=True)
    with t4:
        st.caption("Contadores agregados por token")
        st.dataframe(resumo[["Token", "Compra", "Venda"]].style.apply(_destacar(token_sel), axis=1),
                     hide_index=True)


# =========================================================================== #
# UI: gráficos
# =========================================================================== #

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


def grafico_semiciclos(df: pd.DataFrame, token: str, x0=pd.Timestamp(2025, 3, 2),
                       num_ondas: int = 8, periodo_sem: int = 4 * 7, y_pico: float = 10, moeda: str = "USD"):
    """Ondas senoidais simétricas em torno de x0 (vetorizado)."""
    k = np.arange(-(num_ondas // 2), num_ondas // 2)                 # índices das ondas
    t = np.linspace(0, np.pi, 100)
    inicio = x0 + pd.to_timedelta(k * periodo_sem, unit="W")
    xs = (inicio.values[:, None] + (t / np.pi * periodo_sem * 7 * 86400e9).astype("timedelta64[ns]")).ravel()
    ys = np.tile(y_pico * np.sin(t), len(k))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="gold", width=2)))
    fig.add_trace(go.Scatter(x=xs, y=ys, name="Semiciclos", yaxis="y2",
                             line=dict(color="cyan", width=3, dash="dot")))
    fig.update_layout(title=f"{token} Close + Ondas Simétricas", template="plotly_dark", hovermode="x unified",
                      yaxis=dict(title=f"Preço ({moeda})", tickfont=dict(color="gold")),
                      yaxis2=dict(title="Amplitude das Ondas", tickfont=dict(color="cyan"),
                                  overlaying="y", side="right"),
                      legend=dict(x=0.02, y=0.98), margin=dict(l=60, r=60, t=60, b=40))
    st.plotly_chart(fig, theme=None)


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


@st.cache_data(ttl="15m", show_spinner="Buscando preços e câmbio...")
def precos_carteira(itens: tuple[tuple[str, str], ...]) -> tuple[dict, float]:
    """Último preço de cada ativo na moeda original + câmbio (BRL por 1 USD, ticker BRL=X do Yahoo)."""
    tickers = {"BRL=X"}
    for s, m in itens:
        if m == "Cripto":
            tickers.add(ticker_yahoo(s, "Cripto"))
        elif m in ("Ações B3", "Ações EUA"):
            tickers.add(ticker_yahoo(s, m))
        elif m == "?":
            tickers |= {f"{s}-USD", s}
    raw = yf.download(sorted(tickers), period="10d", interval="1d", group_by="ticker", auto_adjust=False,
                      progress=False, threads=True)

    def ultimo(t):
        try:
            serie = raw[t]["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
            serie = serie.dropna()
            return float(serie.iloc[-1]) if len(serie) else np.nan
        except KeyError:
            return np.nan

    fx = ultimo("BRL=X")
    out = {}
    for s, m in itens:
        if m == "Caixa":
            out[s] = (1.0, s, "caixa")
        elif m == "Cripto":
            out[s] = (ultimo(ticker_yahoo(s, "Cripto")), "USD", "Cripto")
        elif m == "Ações B3":
            out[s] = (ultimo(ticker_yahoo(s, m)), "BRL", "B3")
        elif m == "Ações EUA":
            out[s] = (ultimo(s), "USD", "EUA")
        else:
            p = ultimo(f"{s}-USD")
            out[s] = (p, "USD", "Cripto") if np.isfinite(p) else (ultimo(s), "USD", "EUA")
    return out, fx


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
        st.warning("Sem preço no Yahoo Finance (ficaram FORA do total): " + ", ".join(sem_preco) +
                   ". Confira o código do ativo.")

    st.subheader(f"💰 Valor total: **{sim} {total:,.2f}**")
    st.caption(f"Câmbio usado: 1 USD = {fx:,.4f} BRL (Yahoo Finance, BRL=X).")

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

@st.cache_data(ttl="1h", show_spinner=False)
def fear_and_greed():
    try:
        d = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8).json()["data"][0]
        return d["value"], d["value_classification"]
    except Exception as e:  # noqa: BLE001
        return "—", f"indisponível ({type(e).__name__})"


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

@st.cache_data(ttl=TTL, show_spinner=False, max_entries=500)
def calcular_linhas(chave: tuple, freq: str, token: str, p: ln.ParamsLinhas, backtest: bool,
                    ini: str | None = None, fim: str | None = None) -> dict:
    """Linhas calculadas só no período escolhido no slider (padrão: últimos 36 meses)."""
    df = preparar_dados(*chave)[freq][token].loc[ini:fim]
    return ln.analisar(df, p, backtest=backtest)


def agregar_backtest(tabelas: list[pd.DataFrame]) -> pd.DataFrame:
    """Soma os testes de vários tokens e recalcula taxa, IC e base (ponderada)."""
    tabs = [t for t in tabelas if t is not None and not t.empty]
    if not tabs:
        return pd.DataFrame()
    if len(tabs) == 1:
        return tabs[0]
    t = pd.concat(tabs, ignore_index=True)
    t["rep"] = t["Repique %"] / 100 * t["Testes"]
    t["base"] = t["Base aleatória %"] / 100 * t["Testes"]
    g = t.groupby(["Tipo", "Toques antes do teste"], as_index=False)[["Testes", "rep", "base"]].sum()
    lo, hi = ln.wilson(g["rep"].to_numpy(), g["Testes"].to_numpy(float))
    taxa, base = g["rep"] / g["Testes"], g["base"] / g["Testes"]
    return pd.DataFrame({
        "Tipo": g["Tipo"], "Toques antes do teste": g["Toques antes do teste"], "Testes": g["Testes"],
        "Repique %": (100 * taxa).round(1),
        "IC95% (min-máx)": [f"{100 * a:.0f}–{100 * b:.0f}" for a, b in zip(lo, hi)],
        "Base aleatória %": (100 * base).round(1), "Edge (p.p.)": (100 * (taxa - base)).round(1),
    })


def agregar_reteste(tabelas: list[pd.DataFrame]) -> pd.DataFrame:
    """Soma os casos de rompimento/reteste de vários tokens."""
    tabs = [t for t in tabelas if t is not None and not t.empty]
    if not tabs:
        return pd.DataFrame()
    if len(tabs) == 1:
        return tabs[0]
    t = pd.concat(tabs, ignore_index=True)
    t["ac"] = t["Acerto %"] / 100 * t["Casos"]
    t["bs"] = t["Base aleatória %"] / 100 * t["Casos"]
    g = t.groupby(["Operação", "Entrada"], as_index=False)[["Casos", "ac", "bs"]].sum()
    lo, hi = ln.wilson(g["ac"].to_numpy(), g["Casos"].to_numpy(float))
    taxa, base = g["ac"] / g["Casos"], g["bs"] / g["Casos"]
    return pd.DataFrame({
        "Operação": g["Operação"], "Entrada": g["Entrada"], "Casos": g["Casos"].astype(int),
        "Acerto %": (100 * taxa).round(1), "IC95% (min-máx)": [f"{100 * a:.0f}–{100 * b:.0f}" for a, b in zip(lo, hi)],
        "Base aleatória %": (100 * base).round(1), "Edge (p.p.)": (100 * (taxa - base)).round(1),
    }).sort_values(["Operação", "Entrada"], ascending=[True, False], ignore_index=True)


MODOS = {"toques": "Mais toques = mais forte",
         "virgem": "Menos testada = mais forte",
         "calibrado": "Calibrado pelo backtest do ativo"}


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

    with st.expander("📊 Backtest: toques anteriores × probabilidade de repique (este ativo)"):
        st.caption(
            "Cada vez que o preço volta a uma linha é um teste. Repique = andar 2×tolerância a favor da linha "
            "antes de andar 2×tolerância contra, em até 20 barras. 'Base aleatória' é a mesma conta feita em "
            "qualquer barra: só há vantagem real quando o repique fica acima da base e fora do intervalo de "
            "confiança. Para linhas do RSI o resultado é medido no PREÇO.")
        c_bt, c_ag = st.columns(2)
        rodar = c_bt.toggle("Rodar backtest")
        agregado = c_ag.toggle("Agregar todos os tokens", help="Um ativo sozinho tem poucos testes com 4+ toques; "
                               "somando a lista inteira os intervalos de confiança ficam bem mais estreitos.")
        if rodar:
            alvo = tokens if agregado else [token]
            a, b = st.columns(2)
            resultados = [calcular_linhas(chave, freq, t, p, True, *faixa) for t in alvo]
            st.markdown("**Toques em linhas ativas (repique)**")
            a, b = st.columns(2)
            for col, serie, nome in ((a, "preco", "Linhas no preço"), (b, "rsi", "Linhas no RSI")):
                with col:
                    st.markdown(f"{nome}" + (f" · {len(alvo)} tokens" if agregado else ""))
                    tab = agregar_backtest([r[serie]["tabela"] for r in resultados])
                    if tab.empty:
                        st.info("Poucos eventos.")
                    else:
                        st.dataframe(tab, hide_index=True)
            st.markdown("**Rompimento × reteste: entrar no rompimento ou esperar o reteste?**")
            st.caption("Compra = LTB/resistência rompida para cima; venda = LTA/suporte perdido. Acerto = "
                       "andar 2×tolerância a favor antes de 2×tolerância contra, em até 20 barras.")
            a, b = st.columns(2)
            for col, serie, nome in ((a, "preco", "Linhas no preço"), (b, "rsi", "Linhas no RSI (medido no preço)")):
                with col:
                    st.markdown(nome)
                    tab = agregar_reteste([r[serie]["tabela_reteste"] for r in resultados])
                    if tab.empty:
                        st.info("Poucos eventos.")
                    else:
                        st.dataframe(tab, hide_index=True)

    with st.expander("🔎 Scanner: sinais por linhas em todos os tokens"):
        if st.toggle("Escanear lista inteira"):
            barra = st.progress(0.0)
            partes = []
            for k, tok in enumerate(tokens):
                s = calcular_linhas(chave, freq, tok, p, False, *faixa)["sinais"]
                if not s.empty:
                    partes.append(s.assign(Token=tok))
                barra.progress((k + 1) / len(tokens))
            barra.empty()
            if partes:
                tab = pd.concat(partes, ignore_index=True)[["Token", "Lado", " ", "Sinal", "Força"]]
                st.dataframe(tab.sort_values("Força", ascending=False), hide_index=True, column_config={
                    "Força": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f")})
            else:
                st.info("Nenhum sinal por linhas agora.")


# =========================================================================== #
# Aba: Indicadores × ML
# =========================================================================== #

@st.cache_data(ttl="6h", show_spinner=False)
def fear_greed_historico() -> pd.Series:
    """Histórico diário do Medo & Ganância (alternative.me, desde 01/02/2018)."""
    try:
        dados_ = requests.get("https://api.alternative.me/fng/?limit=0&format=json", timeout=15).json()["data"]
        s_ = pd.Series({pd.to_datetime(int(d["timestamp"]), unit="s").normalize(): float(d["value"])
                        for d in dados_})
        return s_.sort_index()
    except Exception:  # noqa: BLE001
        return pd.Series(dtype=float)


@st.cache_data(ttl=TTL, show_spinner=False, max_entries=12)
def base_indicadores(chave: tuple, prazo: int, so_token: str | None) -> pd.DataFrame:
    d1 = preparar_dados(*chave)["1D"]
    diarios = {s_: df[OHLCV] for s_, df in d1.items() if so_token in (None, s_)}
    cripto = len(chave) < 4 or chave[3] == "Cripto"
    return ml.montar_base(diarios, prazo, fear_greed_historico() if cripto else None)


@st.cache_resource
def _memo_ml() -> dict:
    return {}


def analisar_ml(chave: tuple, prazo: int, so_token: str | None, grupos: tuple, tarefa: str, familia: str,
                progresso=None):
    """Resultado guardado por configuração (com barra de progresso na 1ª vez)."""
    memo = _memo_ml()
    k = (id(preparar_dados(*chave)), chave, prazo, so_token, grupos, tarefa, familia)
    if k not in memo:
        base = base_indicadores(chave, prazo, so_token)
        cols = [c for c in ml.INDICADORES if ml.GRUPO[c] in grupos]
        memo[k] = ml.analisar(base, cols, prazo, tarefa, familia, progresso)
        while len(memo) > 12:
            memo.pop(next(iter(memo)))
    return memo[k]


AJUDA_TAREFA = """
**Compra × venda (classificação)** — o modelo responde a uma pergunta de sim ou não: *"daqui a N dias o
preço vai estar mais alto do que hoje?"*. O resultado é uma **chance de alta (%)** e um **sinal**:

- 🟢 **COMPRA** quando a chance está entre as 25% mais altas que o modelo costuma dar;
- 🔴 **VENDA** quando está entre as 25% mais baixas;
- ⚪ **neutro** no meio.

**Variação em % (regressão)** — o modelo tenta prever **quanto** o preço vai variar em N dias (ex.: +3,2%).
Serve para comparar ativos entre si, mas o erro costuma ser grande: a variação real fica, em geral, bem
longe da prevista. O sinal é só o sentido da previsão (alta ou queda).
"""

AJUDA_PRAZO = """
Quantos dias à frente o modelo olha.

- **7 dias**: movimentos curtos — mais casos para aprender, mas muito ruído.
- **14 dias**: meio-termo (padrão).
- **30 dias**: tendência — menos ruído, mas menos casos independentes no histórico (e o mercado muda de
  fase no meio do caminho).

Referência: em cripto o preço sobe em ~50–55% dos períodos de 14 dias, então acertar 60% já é muito.
"""

AJUDA_MODELO = """
**Automático (recomendado)**: treina os três abaixo, cada um em três níveis de complexidade, e fica com o
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
"""

AJUDA_NOTA = {
    "cls": "AUC (área sob a curva ROC): a chance de o modelo dar uma probabilidade de alta maior para um dia "
           "que de fato subiu do que para um dia que caiu. 0,50 = cara ou coroa; 1,00 = perfeito. "
           "Em mercado, 0,53–0,56 já é considerado bom.",
    "reg": "Correlação de Spearman entre a variação prevista e a real: mede se o modelo ORDENA bem quem vai "
           "subir mais (usa a posição no ranking, não o valor, então não é distorcida por dias extremos). "
           "0 = acaso; 1 = ordem perfeita; negativo = ordem invertida. Em mercado, 0,05–0,10 já é bom.",
}


@st.fragment
def aba_ml(chave: tuple, token: str):
    st.caption("Treina um modelo com **RSI, médias móveis, volume, Medo & Ganância e LTA/LTB** (candles diários) "
               "e confere se ele acerta em meses que **não viu**. Passe o mouse no ❔ de cada opção para detalhes.")
    c1, c2, c3, c4 = st.columns([1.4, 0.8, 1.2, 1.1])
    tarefa = c1.radio("O que prever", list(ml.TAREFAS), format_func=ml.TAREFAS.get, help=AJUDA_TAREFA)
    prazo = c2.selectbox("Prazo", [7, 14, 30], index=1, format_func=lambda d: f"{d} dias", help=AJUDA_PRAZO)
    opcoes_mod = ["auto"] + list(ml.FAMILIAS)
    familia = c3.selectbox("Modelo", opcoes_mod, help=AJUDA_MODELO,
                           format_func=lambda f: "Automático (melhor na validação)" if f == "auto"
                           else ml.nome_modelo(f, tarefa))
    so_tok = c4.radio("Ativos", ["Todos juntos", f"Só {token}"], help=AJUDA_MOEDAS)
    cripto = len(chave) < 4 or chave[3] == "Cripto"
    grupos_ok = ml.GRUPOS if cripto else [g for g in ml.GRUPOS if g != "Medo & Ganância"]
    grupos = st.multiselect("Indicadores que o modelo pode usar", grupos_ok, default=grupos_ok, help=AJUDA_GRUPOS)
    if not cripto:
        st.caption("O índice de Medo & Ganância (alternative.me) é do mercado cripto, então não entra para ações.")
    if not grupos:
        st.info("Escolha pelo menos um grupo de indicadores.")
        return
    so_token = None if so_tok == "Todos juntos" else token
    barra = st.progress(0.0, "Montando os indicadores...")
    try:
        res = analisar_ml(chave, prazo, so_token, tuple(grupos), tarefa, familia,
                          progresso=lambda f, t: barra.progress(min(f, 1.0), t))
    except ValueError as e:
        barra.empty()
        st.error(str(e))
        return
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

    # ---------------- 1. dá para confiar?
    st.subheader("1 · Ele acertou em meses que não viu?")
    if cls:
        ok_ = t["nota"] >= 0.52 and (t["acerto_compra"] - (1 - t["acerto_venda"])) >= 0.04
        fraco = t["nota"] >= 0.51
    else:
        ok_ = t["nota"] >= 0.05 and t["ret_top"] > t["ret_fundo"]
        fraco = t["nota"] >= 0.02 and t["ret_top"] > t["ret_fundo"]
    if ok_:
        st.success("✅ O padrão se repetiu no período de teste. É um sinal fraco (mercado é muito aleatório): "
                   "use como filtro junto com sua análise, não como gatilho sozinho.")
    elif fraco:
        st.warning("🟡 Sinal fraco: no teste ficou só um pouco acima do acaso. Use com cautela.")
    else:
        st.error("❌ No teste não acertou mais que o acaso: não use o sinal deste modelo para decidir. "
                 "As tabelas de combinações abaixo continuam valendo como estatística histórica.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"{ml.METRICA[tarefa]} no teste", f"{t['nota']:.3f}", f"{t['nota'] - t['nota_val']:+.3f} vs validação",
              delta_color="off", help=AJUDA_NOTA[tarefa])
    if cls:
        caiu_todos = 1 - t["subiu_todos"]
        m2.metric("Quando deu 🟢 COMPRA, subiu", "—" if np.isnan(t["acerto_compra"]) else f"{100 * t['acerto_compra']:.0f}%",
                  f"{100 * (t['acerto_compra'] - t['subiu_todos']):+.0f} p.p. vs um dia qualquer",
                  help=f"{t['n_compra']} sinais de compra no teste. Num dia qualquer do teste o preço subiu "
                       f"{100 * t['subiu_todos']:.0f}% das vezes.")
        m3.metric("Quando deu 🔴 VENDA, caiu", "—" if np.isnan(t["acerto_venda"]) else f"{100 * t['acerto_venda']:.0f}%",
                  f"{100 * (t['acerto_venda'] - caiu_todos):+.0f} p.p. vs um dia qualquer",
                  help=f"{t['n_venda']} sinais de venda no teste. Num dia qualquer do teste o preço caiu "
                       f"{100 * caiu_todos:.0f}% das vezes.")
        m4.metric(f"Variação média em {prazo}d após COMPRA", "—" if np.isnan(t["ret_compra"]) else f"{100 * t['ret_compra']:+.1f}%",
                  f"após VENDA: {100 * t['ret_venda']:+.1f}%", delta_color="off",
                  help=f"Em todos os dias do teste: {100 * t['ret_todos']:+.1f}%. Sem taxas.")
    else:
        m2.metric("Acertou a direção", f"{100 * t['acerto_direcao']:.0f}%",
                  help=f"Previu alta e subiu, ou previu queda e caiu. Só 'sempre alta' acertaria "
                       f"{100 * max(t['subiu_todos'], 1 - t['subiu_todos']):.0f}%.")
        m3.metric("20% maiores previsões → variação real", f"{100 * t['ret_top']:+.1f}%",
                  f"20% menores: {100 * t['ret_fundo']:+.1f}%", delta_color="off",
                  help="Se o modelo ordena bem, os ativos/dias com previsão mais alta devem ter subido mais.")
        m4.metric("Erro típico da previsão", f"±{100 * t['erro_tipico']:.1f} p.p.",
                  f"chutando a média: ±{100 * t['erro_ingenuo']:.1f}", delta_color="off",
                  help="Diferença mediana entre a variação prevista e a real. Se não for menor que 'chutando "
                       "a média', a previsão do valor em si não ajuda (mesmo que a ordem ajude).")
    st.caption(f"Teste: {t['inicio']:%d/%m/%Y} a {t['fim']:%d/%m/%Y} · {t['n']:,} dias-ativo · "
               f"{res['n_tokens']} ativo(s). O modelo foi escolhido só com os dados anteriores a esse período.")

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
    ag = res["agora"]
    linha = ag[ag["Token"] == token]
    aviso = "" if (ok_ or fraco) else " — ⚠️ modelo sem acerto comprovado"
    if not linha.empty:
        r0 = linha.iloc[0]
        if cls:
            st.metric(f"{token} · sinal do modelo ({prazo} dias){aviso}", r0["Sinal"], f"chance de alta {r0['Chance de alta %']:.0f}%",
                      delta_color="off", help=f"COMPRA se a chance ≥ {100 * res['limites_agora'][0]:.0f}%, VENDA se "
                                              f"≤ {100 * res['limites_agora'][1]:.0f}% (25% mais altas / mais baixas "
                                              "que o modelo costuma dar).")
        else:
            st.metric(f"{token} · variação prevista em {prazo} dias{aviso}", f"{r0['Variação prevista %']:+.1f}%",
                      f"erro típico ±{100 * t['erro_tipico']:.0f} p.p.", delta_color="off")
    a, b = st.columns([3, 2])
    with a:
        st.markdown(f"**{token}: cada indicador hoje e o que aconteceu historicamente nessa faixa**")
        lt = ml.leitura_token(res, base, token)
        if lt.empty:
            st.info(f"{token} não entra nesta análise (stablecoin ou histórico curto).")
        else:
            st.dataframe(lt, hide_index=True, column_config={
                "Subiu % nessa faixa": st.column_config.NumberColumn(format="%.1f"),
                "Variação média %": st.column_config.NumberColumn(format="%+.2f")})
    with b:
        st.markdown("**Todos os ativos**")
        st.dataframe(ag.style.apply(_destacar(token), axis=1), hide_index=True)
        st.caption("Só use o sinal se o item 1 estiver verde.")


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
    usar_ultima = st.checkbox("Usar última data", value=True, help="Se desmarcado, selecionar última data")
    data_maxima = None if usar_ultima else st.date_input(
        "Selecione última data", value=pd.Timestamp.today().date(), format="DD/MM/YYYY").isoformat()
    periodicidade = st.radio("Periodicidade", options=["1D", "1W"], index=1, horizontal=True,
                             help="Dados diários (1D) ou semanais (1W)")

lista = {"Cripto": simbolos, "Ações B3": ACOES_B3, "Ações EUA": ACOES_EUA}[mercado]
moeda = MERCADOS[mercado]
chave = (lista, PERIODO, data_maxima, mercado)
dados = preparar_dados(*chave)
tokens = list(dados["1D"])
if not tokens:
    st.error(f"Nenhum dado baixado para {mercado}. Verifique a conexão ou a lista de símbolos.")
    st.stop()

with st.sidebar:
    token_sel = st.selectbox("Selecione o ativo", options=tokens, index=0)
    if st.button("🔄 Atualizar dados agora"):
        baixar_ohlc.clear()
        preparar_dados.clear()
        st.rerun()

top25, top50, top100 = faixas_volume(dados["1D"]) if mercado == "Cripto" else ([], [], [])

ABAS = ["Carteira", "Indicadores", "Análise por Linhas", "Indicadores × ML"]
try:   # abas preguiçosas (Streamlit recente): só a aberta é executada
    tab_carteira, tab_indicadores, tab_linhas, tab_ml = st.tabs(ABAS, on_change="rerun")
except TypeError:
    tab_carteira, tab_indicadores, tab_linhas, tab_ml = st.tabs(ABAS)


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
            grafico_indicadores(df_tok.loc[ini:fim], token_sel, periodicidade, moeda)
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

if _aberta(tab_carteira):
    with tab_carteira:
        st.header("🧮 Rebalanceador de Carteira")
        exibir_rebalanceamento(simbolos, up_, txt_manual_)
