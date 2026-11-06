"""
ativos.py — listas de ativos e conversão para os códigos do Yahoo Finance.
Usado pelo app (Crypto_Analysis.py), pela camada de dados (fontes.py) e pelo
gerador da versão web (gerar_html.py).
"""
import os
import re

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


def ler_simbolos(caminho: str = "add_symbols.txt") -> tuple[str, ...]:
    """Lista de cripto do add_symbols.txt (separados por espaço, vírgula ou ponto e vírgula)."""
    for c in (caminho, os.path.join(PASTA, caminho)):
        if os.path.exists(c):
            with open(c, encoding="utf-8") as f:
                txt = "\n".join(l_ for l_ in f if not l_.strip().startswith("#"))
            simbolos = [s.upper() for s in re.split(r"[,;\s]+", txt) if s.strip()]
            return tuple(dict.fromkeys(simbolos))          # remove duplicatas, mantém ordem
    return ("BTC", "ETH")
