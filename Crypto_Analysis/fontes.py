"""
fontes.py — de onde vêm os preços.

O app roda em dois lugares:

1. No PC (Python normal, `streamlit run`): baixa tudo do Yahoo Finance (yfinance)
   e o Medo & Ganância do alternative.me, como sempre.

2. No navegador (versão HTML com stlite/Pyodide): o navegador NÃO consegue falar
   com o Yahoo Finance (o Yahoo não libera CORS e o yfinance depende de bibliotecas
   em C que não existem no Pyodide). Então:
     - usa um RETRATO dos dados embutido no HTML (dados/ohlc.csv.gz), gerado no PC
       pelo `gerar_html.py`;
     - cripto: tenta completar com os candles mais recentes direto da Binance
       (API pública, sem chave, liberada para navegador);
     - Medo & Ganância: tenta o alternative.me; se não der, usa o retrato.
   Se a internet ou essas APIs falharem, o app continua com o retrato.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import sys

import numpy as np
import pandas as pd

from ativos import PASTA, ticker_yahoo

NO_NAVEGADOR = sys.platform == "emscripten"          # True dentro do Pyodide/stlite
ARQ_RETRATO = os.path.join(PASTA, "dados", "ohlc.csv.gz")
ARQ_FG = os.path.join(PASTA, "dados", "fear_greed.csv")
ARQ_META = os.path.join(PASTA, "dados", "meta.json")
OHLCV = ["Open", "High", "Low", "Close", "Volume"]
URL_FG = "https://api.alternative.me/fng/?limit={n}&format=json"
URLS_BINANCE = ("https://data-api.binance.vision/api/v3/klines", "https://api.binance.com/api/v3/klines")

# o que aconteceu na última busca (mostrado na barra lateral da versão web)
STATUS: dict[str, str] = {}


# --------------------------------------------------------------------------- #
# HTTP (PC: requests · navegador: XMLHttpRequest síncrono do Pyodide)
# --------------------------------------------------------------------------- #

def _get_json(url: str, timeout: int = 15):
    if NO_NAVEGADOR:
        from pyodide.http import open_url          # disponível só dentro do Pyodide
        return json.loads(open_url(url).read())
    import requests
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


# --------------------------------------------------------------------------- #
# PC: Yahoo Finance
# --------------------------------------------------------------------------- #

def _normalizar(df: pd.DataFrame) -> pd.DataFrame:
    df = df[OHLCV].dropna(subset=["Close"]).astype(float)
    idx = pd.to_datetime(df.index)
    df.index = (idx.tz_localize(None) if idx.tz is not None else idx).rename("Date")
    return df


def usd_em_btc(b: pd.DataFrame) -> pd.DataFrame:
    """Ativo 'USD' da lista cripto = dólar medido em BTC: inverte o BTC e troca máxima/mínima."""
    return pd.DataFrame({"Open": 1 / b["Open"], "High": 1 / b["Low"], "Low": 1 / b["High"],
                         "Close": 1 / b["Close"], "Volume": b["Volume"]}, index=b.index)


def baixar_yahoo(simbolos: tuple[str, ...], periodo: str = "max", mercado: str = "Cripto") -> dict[str, pd.DataFrame]:
    import yfinance as yf                          # só no PC
    tickers = sorted({ticker_yahoo(s, mercado) for s in simbolos})
    raw = yf.download(tickers, period=periodo, group_by="ticker", auto_adjust=False, progress=False, threads=True)
    out = {}
    for s in simbolos:
        t = ticker_yahoo(s, mercado)
        if isinstance(raw.columns, pd.MultiIndex):
            if t not in raw.columns.get_level_values(0):
                continue
            df = raw[t]
        else:
            df = raw
        df = _normalizar(df)
        if not df.empty:
            out[s] = usd_em_btc(df) if (mercado == "Cripto" and s == "USD") else df
    return out


# --------------------------------------------------------------------------- #
# Navegador: retrato embutido + Binance
# --------------------------------------------------------------------------- #

_cache_retrato: pd.DataFrame | None = None


def retrato() -> pd.DataFrame:
    """Todas as séries do retrato (colunas: mercado, simbolo, Date, OHLCV)."""
    global _cache_retrato
    if _cache_retrato is None:
        if os.path.exists(ARQ_RETRATO):
            with gzip.open(ARQ_RETRATO, "rt", encoding="utf-8") as f:
                _cache_retrato = pd.read_csv(f, parse_dates=["Date"])
        else:
            _cache_retrato = pd.DataFrame(columns=["mercado", "simbolo", "Date"] + OHLCV)
    return _cache_retrato


def meta() -> dict:
    if os.path.exists(ARQ_META):
        with open(ARQ_META, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _do_retrato(mercado: str, simbolos: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    r = retrato()
    r = r[(r["mercado"] == mercado) & r["simbolo"].isin(simbolos)]
    grupos = {s: g.set_index("Date")[OHLCV].sort_index() for s, g in r.groupby("simbolo")}
    return {s: grupos[s] for s in simbolos if s in grupos}          # mesma ordem da lista


def par_binance(simbolo: str) -> str:
    """Código do Yahoo -> par da Binance. O Yahoo põe números para desempatar nomes (SUI20947, UNI7083)."""
    return re.sub(r"(?<=[A-Z])\d{4,}$", "", simbolo) + "USDT"


def _binance_desde(simbolo: str, inicio: pd.Timestamp, par: str | None = None) -> pd.DataFrame:
    """Candles diários SIMBOLO/USDT da Binance a partir de ``inicio`` (volume em USDT)."""
    par = par or par_binance(simbolo)
    linhas, ultimo_erro = [], None
    for base in URLS_BINANCE:
        ms = int(pd.Timestamp(inicio).timestamp() * 1000)
        try:
            linhas = []
            while True:
                lote = _get_json(f"{base}?symbol={par}&interval=1d&startTime={ms}&limit=1000")
                if isinstance(lote, dict):              # {"code": -1121, "msg": "Invalid symbol."}
                    raise LookupError(f"{par}: {lote.get('msg', lote)}")
                if not lote:
                    break
                linhas += lote
                if len(lote) < 1000:
                    break
                ms = lote[-1][0] + 86_400_000
            break
        except LookupError:                             # par não existe na Binance: nem tenta o outro endereço
            raise
        except Exception as e:  # noqa: BLE001 — falha de rede: tenta o próximo endereço
            ultimo_erro = e
    if not linhas:
        if ultimo_erro:
            raise ultimo_erro
        return pd.DataFrame(columns=OHLCV)
    a = np.array([[float(x) for x in (k[0], k[1], k[2], k[3], k[4], k[7])] for k in linhas])
    idx = pd.to_datetime(a[:, 0], unit="ms").rename("Date")
    return pd.DataFrame(a[:, 1:], index=idx, columns=OHLCV)


def ohlc(simbolos: tuple[str, ...], periodo: str = "max", mercado: str = "Cripto",
         ao_vivo: bool = True) -> dict[str, pd.DataFrame]:
    """Séries diárias de todos os ativos do mercado. PC: Yahoo. Navegador: retrato (+ Binance)."""
    if not NO_NAVEGADOR:
        return baixar_yahoo(simbolos, periodo, mercado)
    out = _do_retrato(mercado, simbolos)
    if mercado != "Cripto" or not ao_vivo:
        STATUS[mercado] = "retrato"
        return out
    ok, falhas = 0, []
    for s in simbolos:
        if s == "USD":                       # USD = inverso do BTC (feito abaixo)
            continue
        base = out.get(s)
        # pega 30 dias que o retrato já tem para "costurar" as duas fontes
        inicio = (base.index[max(len(base) - 30, 0)] if base is not None and len(base) else pd.Timestamp("2015-01-01"))
        try:
            novo = _binance_desde(s, inicio)
        except Exception:  # noqa: BLE001
            falhas.append(s)
            continue
        if novo.empty:
            continue
        if base is not None and len(base):
            # o volume do Yahoo soma várias corretoras; o da Binance é só dela -> mesma escala nos dias em comum
            comum = base.index.intersection(novo.index)
            razao = (base.loc[comum, "Volume"] / novo.loc[comum, "Volume"].replace(0, np.nan)).median()
            if np.isfinite(razao) and razao > 0:
                novo = novo.assign(Volume=novo["Volume"] * razao)
            novo = novo[novo.index > base.index[-1] - pd.Timedelta(days=2)]   # refaz só os últimos dias
            out[s] = pd.concat([base[base.index < novo.index[0]], novo]) if len(novo) else base
        else:
            out[s] = novo
        ok += 1
    if "USD" in simbolos and "BTC" in out:
        out["USD"] = usd_em_btc(out["BTC"])
    if not ok:
        STATUS[mercado] = "sem conexão com a Binance: usando só o retrato"
    else:
        STATUS[mercado] = f"Binance ao vivo para {ok} cripto(s)" + (
            f"; fora da Binance (retrato): {', '.join(falhas[:8])}{'…' if len(falhas) > 8 else ''}" if falhas else "")
    return out


# --------------------------------------------------------------------------- #
# Medo & Ganância
# --------------------------------------------------------------------------- #

def _fg_da_api(n: int) -> pd.Series:
    d = _get_json(URL_FG.format(n=n))["data"]
    return pd.Series({pd.to_datetime(int(x["timestamp"]), unit="s").normalize(): float(x["value"])
                      for x in d}).sort_index()


def fear_greed_historico() -> pd.Series:
    try:
        s = _fg_da_api(0)
        if len(s):
            return s
    except Exception:  # noqa: BLE001
        pass
    if os.path.exists(ARQ_FG):                    # retrato embutido (versão web)
        s = pd.read_csv(ARQ_FG, parse_dates=["Date"]).set_index("Date")["fg"]
        return s.sort_index()
    return pd.Series(dtype=float)


CLASSES_FG = ((25, "Extreme Fear"), (45, "Fear"), (55, "Neutral"), (75, "Greed"), (101, "Extreme Greed"))


def fear_greed_atual() -> tuple[str, str]:
    try:
        d = _get_json(URL_FG.format(n=1), timeout=8)["data"][0]
        return d["value"], d["value_classification"]
    except Exception as e:  # noqa: BLE001
        s = fear_greed_historico()
        if len(s):
            v = int(s.iloc[-1])
            classe = next(c for lim, c in CLASSES_FG if v < lim)
            return str(v), f"{classe} (retrato de {s.index[-1]:%d/%m/%Y})"
        return "—", f"indisponível ({type(e).__name__})"


# --------------------------------------------------------------------------- #
# Último preço de cada ativo (carteira)
# --------------------------------------------------------------------------- #

def ultimos_precos(itens: tuple[tuple[str, str], ...]) -> tuple[dict, float]:
    """{simbolo: (preço, moeda, fonte)} e o câmbio (BRL por 1 USD)."""
    if not NO_NAVEGADOR:
        return _ultimos_yahoo(itens)
    r = retrato()
    ult = r.sort_values("Date").groupby(["mercado", "simbolo"])["Close"].last()

    def do_retrato(m, s):
        return float(ult.get((m, s), np.nan))

    fx = do_retrato("FX", "BRL=X")
    try:   # câmbio ao vivo: par USDT/BRL da Binance (1 USDT ≈ 1 USD)
        ult_fx = _binance_desde("", pd.Timestamp.now() - pd.Timedelta(days=3), par="USDTBRL")
        if len(ult_fx):
            fx = float(ult_fx["Close"].iloc[-1])
            STATUS["Câmbio"] = "USDT/BRL da Binance (ao vivo)"
    except Exception:  # noqa: BLE001
        STATUS["Câmbio"] = "BRL=X do retrato"
    out = {}
    for s, m in itens:
        if m == "Caixa":
            out[s] = (1.0, s, "caixa")
            continue
        if m in ("Cripto", "?"):
            p = np.nan
            try:
                ultimos = _binance_desde(s, pd.Timestamp.now() - pd.Timedelta(days=3))
                p = float(ultimos["Close"].iloc[-1]) if len(ultimos) else np.nan
            except Exception:  # noqa: BLE001
                pass
            if not np.isfinite(p):
                p = do_retrato("Cripto", s)
            if np.isfinite(p) or m == "Cripto":
                out[s] = (p, "USD", "Cripto")
                continue
            out[s] = (do_retrato("Ações EUA", s), "USD", "EUA")
        elif m == "Ações B3":
            out[s] = (do_retrato(m, s), "BRL", "B3")
        else:
            out[s] = (do_retrato(m, s), "USD", "EUA")
    return out, fx


def _ultimos_yahoo(itens):
    import yfinance as yf
    tickers = {"BRL=X"}
    for s, m in itens:
        if m in ("Cripto", "Ações B3", "Ações EUA"):
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
    return out, ultimo("BRL=X")


def descricao_fonte() -> str:
    """Texto curto para a barra lateral."""
    if not NO_NAVEGADOR:
        return "Yahoo Finance (ao vivo)"
    m = meta()
    ao_vivo = any(v.startswith("Binance ao vivo") for v in STATUS.values())
    base = f"retrato de {m['gerado_em']}" if m.get("gerado_em") else "sem retrato embutido"
    return base + (" + cripto ao vivo (Binance)" if ao_vivo else "")
