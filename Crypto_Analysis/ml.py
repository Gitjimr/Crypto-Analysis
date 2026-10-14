"""
ml.py — Combinações de indicadores × Machine Learning

Usa só os indicadores que importam na prática:
    RSI · médias móveis · picos de volume · Medo & Ganância · LTA/LTB

Duas tarefas:
- Classificação (compra × venda): o preço estava mais alto N dias depois?
- Regressão: quanto o preço variou (%) em N dias?

Três modelos, comparados automaticamente na validação: Gradient Boosting,
Random Forest e o linear (regressão logística / Ridge).

Leituras:
1. Combinação de grupos de indicadores que ajuda (seleção para a frente).
2. Modelos × complexidade (treino × validação) e o escolhido.
3. Teste final, peso e efeito de cada indicador.
4. Combinações históricas de faixas (sem modelo), conferidas no teste.
5. Leitura de agora (sinal por token).

Validação: tudo separado por DATA (todas as moedas juntas). Desenvolvimento =
primeiros ~80% das datas (walk-forward em 4 dobras, com intervalo de N dias);
teste = últimos ~20%, que o modelo não vê. Todos os atributos são causais
(só usam informação disponível no fechamento do dia).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import linhas as ln

ESTAVEIS = {"USD", "USDT", "USDC", "XAUT", "DAI", "FDUSD", "TUSD", "PAXG"}
GRUPOS = ["RSI", "Médias móveis", "Volume", "Medo & Ganância", "LTA / LTB"]
COR_GRUPO = {"RSI": "#9575cd", "Médias móveis": "#ffb74d", "Volume": "#4fc3f7",
             "Medo & Ganância": "#f06292", "LTA / LTB": "#81c784"}

FX_RSI = ([0, 30, 40, 50, 60, 70, 100], ["< 30", "30–40", "40–50", "50–60", "60–70", "> 70"])
FX_FG = ([0, 25, 45, 55, 75, 100], ["Medo extremo (0–25)", "Medo (25–45)", "Neutro (45–55)",
                                    "Ganância (55–75)", "Ganância extrema (75–100)"])

# chave: (nome, grupo, unidade, faixas) — faixas: (bordas, rótulos) fixas, dict de categorias,
# ou None = quintis calculados no período de desenvolvimento
INDICADORES: dict[str, tuple] = {
    "rsi": ("RSI diário", "RSI", "", FX_RSI),
    "rsi_semanal": ("RSI semanal", "RSI", "", FX_RSI),
    "div_rsi": ("Divergência RSI (10 dias)", "RSI", "", {-1: "de baixa", 0: "nenhuma", 1: "de alta"}),
    "dist_mme50": ("Preço vs MME 50", "Médias móveis", "%", None),
    "dist_mme200": ("Preço vs MME 200", "Médias móveis", "%", None),
    "mme50_vs_200": ("MME 50 vs MME 200", "Médias móveis", "%", None),
    "pi_cycle": ("Pi Cycle (MME111 ÷ 2×MME350)", "Médias móveis", "×", None),
    "vol_rel": ("Volume vs média de 20 dias", "Volume", "×", None),
    "pico_vol": ("Pico de volume (3 dias)", "Volume", "", {-1: "com queda", 0: "nenhum", 1: "com alta"}),
    "fg": ("Medo & Ganância", "Medo & Ganância", "", FX_FG),
    "fg_var7": ("Medo & Ganância: variação 7 dias", "Medo & Ganância", " pts", None),
    "dist_lta": ("Preço vs LTA (+ = acima)", "LTA / LTB", "%", None),
    "dist_ltb": ("Preço vs LTB (+ = rompida)", "LTA / LTB", "%", None),
    "rsi_vs_lta": ("RSI vs LTA do RSI", "LTA / LTB", " pts", None),
    "rsi_vs_ltb": ("RSI vs LTB do RSI", "LTA / LTB", " pts", None),
}
NOME = {k: v[0] for k, v in INDICADORES.items()}
GRUPO = {k: v[1] for k, v in INDICADORES.items()}


# =========================================================================== #
# Atributos (causais)
# =========================================================================== #

def _linha_recente(x, ref, esq, dir_, tipo):
    """Reta pelos 2 últimos pivôs JÁ CONFIRMADOS em cada barra (pivô i só é
    conhecido em i+dir). Retorna o valor da reta em cada barra."""
    n = len(x)
    idx, _ = ln.pivos(x, esq, dir_, tipo)
    out = np.full(n, np.nan)
    if idx.size < 2:
        return out
    t = np.arange(n)
    k = np.searchsorted(idx + dir_, t, side="right") - 1
    ok = k >= 1
    i1, i0 = idx[np.clip(k, 0, None)], idx[np.clip(k - 1, 0, None)]
    m = (x[i1] - x[i0]) / np.maximum(i1 - i0, 1)
    out[ok] = (x[i1] + m * (t - i1))[ok]
    return out


def _rsi_semanal(df: pd.DataFrame) -> np.ndarray:
    """RSI do último candle SEMANAL já fechado (semana segunda→domingo)."""
    sem = df["Close"].resample("W-MON", label="left", closed="left").last().dropna()
    r = ln.rsi_wilder(sem)
    r.index = r.index + pd.Timedelta(days=6)          # conhecido no fechamento de domingo
    return r.reindex(df.index, method="ffill").to_numpy()


def atributos(df: pd.DataFrame, fg: pd.Series | None = None, sens: int = 3) -> pd.DataFrame:
    c, h, l, v = (df[k].astype(float) for k in ("Close", "High", "Low", "Volume"))
    n = len(df)
    x = {}
    rsi = ln.rsi_wilder(c)
    x["rsi"] = rsi
    x["rsi_semanal"] = _rsi_semanal(df)
    div = ln.divergencias(c.to_numpy(), l.to_numpy(), h.to_numpy(), rsi.to_numpy(), sens, sens)
    d = np.zeros(n)
    for r in div.itertuples():                         # ativo por 10 barras após confirmar
        if r.confirmacao < n:
            d[r.confirmacao: r.confirmacao + 10] = 1 if r.tipo == "Bull" else -1
    x["div_rsi"] = d

    ema = {s: c.ewm(span=s, adjust=False).mean() for s in (50, 111, 200, 350)}
    for s in ema:                                      # sem histórico suficiente = desconhecido
        ema[s] = ema[s].where(np.arange(n) >= s)
    x["dist_mme50"] = 100 * (c / ema[50] - 1)
    x["dist_mme200"] = 100 * (c / ema[200] - 1)
    x["mme50_vs_200"] = 100 * (ema[50] / ema[200] - 1)
    x["pi_cycle"] = ema[111] / (2 * ema[350])

    vm = v.rolling(20).mean()
    x["vol_rel"] = v / vm.replace(0, np.nan)
    pico = np.where(v > 2 * vm, np.sign(c.diff().fillna(0)), 0.0)
    x["pico_vol"] = pd.Series(pico, index=df.index).replace(0, np.nan).ffill(limit=2).fillna(0).to_numpy()

    if fg is not None and len(fg):
        f = fg.reindex(df.index.normalize()).to_numpy()
        f = pd.Series(f, index=df.index).ffill(limit=3)
        x["fg"], x["fg_var7"] = f, f - f.shift(7)
    else:
        x["fg"] = x["fg_var7"] = np.full(n, np.nan)

    lc, lh, ll = np.log(c.to_numpy()), np.log(h.to_numpy()), np.log(l.to_numpy())
    x["dist_lta"] = 100 * np.expm1(lc - _linha_recente(ll, lc, sens, sens, "baixo"))
    x["dist_ltb"] = 100 * np.expm1(lc - _linha_recente(lh, lc, sens, sens, "alto"))
    r = rsi.to_numpy()
    x["rsi_vs_lta"] = r - _linha_recente(r, r, sens, sens, "baixo")
    x["rsi_vs_ltb"] = r - _linha_recente(r, r, sens, sens, "alto")
    return pd.DataFrame(x, index=df.index).replace([np.inf, -np.inf], np.nan)


def montar_base(diarios: dict[str, pd.DataFrame], prazo: int, fg: pd.Series | None = None) -> pd.DataFrame:
    """Uma linha por (token, dia). y = 1 se o fechamento N dias depois for maior."""
    partes = []
    for tok, d in diarios.items():
        if tok in ESTAVEIS or len(d) < 120 + prazo:
            continue
        d = d.dropna(subset=["Close"])
        a = atributos(d, fg)
        c = np.log(d["Close"].to_numpy(float))
        ret = np.full(len(c), np.nan)
        ret[:-prazo] = c[prazo:] - c[:-prazo]
        a["ret"] = ret
        a["y"] = np.where(np.isnan(ret), np.nan, (ret > 0).astype(float))
        a = a.iloc[60:]                                # aquecimento do RSI/volume
        a.insert(0, "token", tok)
        partes.append(a)
    if not partes:
        return pd.DataFrame()
    base = pd.concat(partes)
    base.index.name = "Date"
    return base.reset_index()


# =========================================================================== #
# Divisão por data
# =========================================================================== #

def dividir(datas: pd.Series, prazo: int, k: int = 4, frac_teste: float = 0.2):
    gap = pd.Timedelta(days=prazo + 1)
    ud = np.sort(datas.unique())
    corte = pd.Timestamp(ud[int(len(ud) * (1 - frac_teste))])
    d = datas.to_numpy()
    dev = d < np.datetime64(corte - gap)
    blocos = np.array_split(ud[ud < np.datetime64(corte - gap)], k + 1)
    dobras = []
    for j in range(1, k + 1):
        if not len(blocos[j]):
            continue
        tr = d < np.datetime64(pd.Timestamp(blocos[j][0]) - gap)
        va = (d >= blocos[j][0]) & (d <= blocos[j][-1])
        if tr.sum() > 300 and va.sum() > 50:
            dobras.append((tr, va))
    return dobras, dev, d >= np.datetime64(corte), corte


# =========================================================================== #
# Faixas e combinações históricas (sem modelo)
# =========================================================================== #

def _fmt_num(v: float, un: str) -> str:
    a = abs(v)
    s = f"{v:.0f}" if a >= 20 else (f"{v:.1f}" if a >= 2 else f"{v:.2f}")
    return s.replace(".", ",") + un


def faixas(base_dev: pd.DataFrame, cols: list[str]) -> dict:
    """Bordas e rótulos de cada indicador (quintis do desenvolvimento quando não são fixas)."""
    out = {}
    for c in cols:
        _, _, un, fx = INDICADORES[c]
        if isinstance(fx, dict):
            out[c] = ("cat", list(fx), list(fx.values()))
        elif fx is not None:
            out[c] = ("num", list(fx[0][1:-1]), fx[1])
        else:
            q = np.unique(np.round(np.nanquantile(base_dev[c], [0.2, 0.4, 0.6, 0.8]), 4))
            if q.size == 0:
                continue
            rot = [f"< {_fmt_num(q[0], un)}"] + [f"{_fmt_num(a, un)} a {_fmt_num(b, un)}"
                                                  for a, b in zip(q[:-1], q[1:])] + [f"> {_fmt_num(q[-1], un)}"]
            out[c] = ("num", list(q), rot)
    return out


def codigos(serie: pd.Series | np.ndarray, spec) -> np.ndarray:
    """Código inteiro da faixa (−1 = sem dado)."""
    x = np.asarray(serie, dtype=float)
    tipo, bordas, _ = spec
    if tipo == "cat":
        cod = np.full(x.shape, -1)
        for i, k in enumerate(bordas):
            cod[x == k] = i
        return cod
    cod = np.searchsorted(np.asarray(bordas, float), x, side="right")
    return np.where(np.isnan(x), -1, cod)


def combinacoes(base: pd.DataFrame, cols: list[str], fx: dict, dev, te, min_dev: int, min_te: int = 30) -> pd.DataFrame:
    """Para cada faixa de 1 indicador e cada par de faixas de 2 indicadores:
    % de vezes que subiu e variação média, antes (desenvolvimento) e no teste."""
    y = base["y"].to_numpy()
    ok = ~np.isnan(y)
    yv = np.nan_to_num(y)
    rv = np.nan_to_num(np.expm1(base["ret"].to_numpy()))
    cod = {c: codigos(base[c], fx[c]) for c in cols if c in fx}
    base_dev, base_te = y[dev & ok].mean(), y[te & ok].mean()
    ret_dev, ret_te = rv[dev & ok].mean(), rv[te & ok].mean()

    def medir(chave: np.ndarray, nk: int) -> dict:
        r = {}
        for nome, m in (("dev", dev & ok), ("te", te & ok)):
            mm = m & (chave >= 0)
            r[nome] = (np.bincount(chave[mm], minlength=nk),
                       np.bincount(chave[mm], weights=yv[mm], minlength=nk),
                       np.bincount(chave[mm], weights=rv[mm], minlength=nk))
        return r

    linhas = []
    for c in cod:
        nk = len(fx[c][2])
        r = medir(cod[c], nk)
        for i in range(nk):
            linhas.append((NOME[c] + ": " + fx[c][2][i], (c,), (i,), *(r["dev"][q][i] for q in range(3)),
                           *(r["te"][q][i] for q in range(3))))
    for a, b in combinations(cod, 2):
        na, nb = len(fx[a][2]), len(fx[b][2])
        chave = np.where((cod[a] >= 0) & (cod[b] >= 0), cod[a] * nb + cod[b], -1)
        r = medir(chave, na * nb)
        for i in range(na):
            for j in range(nb):
                k = i * nb + j
                if r["dev"][0][k] >= min_dev:
                    linhas.append((f"{NOME[a]}: {fx[a][2][i]}  ·  {NOME[b]}: {fx[b][2][j]}", (a, b), (i, j),
                                   *(r["dev"][q][k] for q in range(3)), *(r["te"][q][k] for q in range(3))))
    t = pd.DataFrame(linhas, columns=["Condição", "cols", "cods", "n_dev", "s_dev", "r_dev", "n_te", "s_te", "r_te"])
    t = t[(t["n_dev"] >= min_dev) & (t["n_te"] >= min_te)].copy()
    t["taxa_dev"], t["taxa_te"] = t["s_dev"] / t["n_dev"], t["s_te"] / t["n_te"]
    t["ret_dev"], t["ret_te"] = t["r_dev"] / t["n_dev"], t["r_te"] / t["n_te"]
    t["edge_dev"], t["edge_te"] = t["taxa_dev"] - base_dev, t["taxa_te"] - base_te
    t["redge_dev"], t["redge_te"] = t["ret_dev"] - ret_dev, t["ret_te"] - ret_te
    t.attrs.update(base_dev=base_dev, base_te=base_te, ret_dev=ret_dev, ret_te=ret_te)
    return t


def _confirmou(edge_dev: float, edge_te: float) -> str:
    if np.sign(edge_te) == np.sign(edge_dev) and abs(edge_te) >= 0.5 * abs(edge_dev):
        return "✅ manteve"
    if np.sign(edge_te) == np.sign(edge_dev):
        return "🟡 enfraqueceu"
    return "❌ não se repetiu"


def tabela_combinacoes(t: pd.DataFrame, lado: str, tarefa: str = "cls", top: int = 12,
                       agora: dict | None = None) -> pd.DataFrame:
    """lado: 'alta' (favoráveis à compra) ou 'queda' (favoráveis à venda)."""
    if t.empty:
        return pd.DataFrame()
    e_dev, e_te = ("edge_dev", "edge_te") if tarefa == "cls" else ("redge_dev", "redge_te")
    s = t.sort_values(e_dev, ascending=(lado == "queda")).head(top)
    if tarefa == "cls":
        met = {"Subiu % (antes)": (100 * s["taxa_dev"]).round(1), "Subiu % (teste)": (100 * s["taxa_te"]).round(1)}
    else:
        met = {"Variação média % (antes)": (100 * s["ret_dev"]).round(2),
               "Variação média % (teste)": (100 * s["ret_te"]).round(2)}
    k = list(met)
    out = pd.DataFrame({"Condição": s["Condição"], "Dias (antes)": s["n_dev"].astype(int), k[0]: met[k[0]],
                        "Dias (teste)": s["n_te"].astype(int), k[1]: met[k[1]],
                        "No teste": [_confirmou(a, b) for a, b in zip(s[e_dev], s[e_te])]})
    if agora is not None:
        out["Tokens nessa condição agora"] = [
            ", ".join(tok for tok, cods in agora.items()
                      if all(cods.get(c) == kk for c, kk in zip(cs, ks))) or "—"
            for cs, ks in zip(s["cols"], s["cods"])]
    return out.reset_index(drop=True)


# =========================================================================== #
# Modelos
# =========================================================================== #

TAREFAS = {
    "cls": "Compra × venda (classificação)",
    "reg": "Variação do preço em % (regressão)",
}

# grade = do mais simples ao mais complexo
FAMILIAS = {
    "gb": {
        "nome": "Gradient Boosting",
        "param": "níveis por árvore", "grade": (1, 2, 3),
        "desc": ("100 árvores de decisão pequenas, treinadas em sequência: cada nova árvore corrige o erro "
                 "das anteriores. Combina bem indicadores em regras do tipo 'RSI < 30 E Medo & Ganância "
                 "< 25' e lida sozinho com dados faltando."),
    },
    "rf": {
        "nome": "Random Forest",
        "param": "níveis por árvore", "grade": (2, 4, 6),
        "desc": ("100 árvores de decisão independentes, cada uma treinada com uma amostra diferente dos "
                 "dados; a resposta é a média delas. Mais estável e menos propensa a 'decorar' o passado."),
    },
    "lin": {
        "nome": {"cls": "Regressão logística", "reg": "Regressão linear (Ridge)"},
        "param": "liberdade dos pesos", "grade": (0.001, 0.01, 0.1),
        "desc": ("Soma ponderada dos indicadores: cada indicador ganha um peso (positivo = puxa para alta). "
                 "É o mais simples e fácil de interpretar, mas não capta combinações do tipo 'só funciona "
                 "quando os dois acontecem juntos'."),
    },
}


def nome_modelo(fam: str, tarefa: str) -> str:
    n = FAMILIAS[fam]["nome"]
    return n[tarefa] if isinstance(n, dict) else n


def descricao_complexidade(fam: str, valor) -> str:
    if fam in ("gb", "rf"):
        return "árvores de 1 nível" if int(valor) == 1 else f"árvores de {int(valor)} níveis"
    return {0.001: "pesos bem contidos", 0.01: "pesos moderados", 0.1: "pesos mais livres"}.get(valor, str(valor))


class Modelo(BaseEstimator):
    """Envelope único para os 3 modelos × 2 tarefas.
    - ignora colunas sem variação no treino (ex.: Medo & Ganância antes de 2018);
    - Random Forest e linear: preenche dados faltando com a mediana (o Gradient
      Boosting trata faltando nativamente);
    - regressão: limita o alvo aos percentis 1–99 do treino (evita que 2 ou 3
      dias de +80% dominem o modelo)."""

    def __init__(self, tarefa: str = "cls", fam: str = "gb", valor=1, n_treino: int = 1000):
        self.tarefa, self.fam, self.valor, self.n_treino = tarefa, fam, valor, n_treino

    def _estimador(self):
        cls = self.tarefa == "cls"
        if self.fam == "gb":
            Est = HistGradientBoostingClassifier if cls else HistGradientBoostingRegressor
            return Est(max_depth=self.valor, learning_rate=0.05, max_iter=100, l2_regularization=5.0,
                       min_samples_leaf=int(np.clip(self.n_treino // 40, 50, 500)), early_stopping=False,
                       random_state=0)
        if self.fam == "rf":
            Est = RandomForestClassifier if cls else RandomForestRegressor
            est = Est(n_estimators=100, max_depth=self.valor, max_features="sqrt", max_samples=0.3,
                      min_samples_leaf=int(np.clip(self.n_treino // 200, 20, 200)), n_jobs=-1, random_state=0)
        else:
            est = (LogisticRegression(C=self.valor, max_iter=2000) if cls
                   else Ridge(alpha=1.0 / self.valor))
            return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)
        return make_pipeline(SimpleImputer(strategy="median"), est)

    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = np.asarray(y, dtype=float)
        self.usar_ = [c for c in X.columns if X[c].nunique() >= 2]
        if self.tarefa == "cls":
            self.classes_ = np.array([0.0, 1.0])
        else:
            lo, hi = np.quantile(y, [0.01, 0.99])
            y = np.clip(y, lo, hi)
        self.media_ = float(np.mean(y))
        self.est_ = self._estimador().fit(X[self.usar_], y) if self.usar_ else None
        return self

    def saida(self, X) -> np.ndarray:
        """Probabilidade de alta (classificação) ou log-retorno previsto (regressão)."""
        if self.est_ is None:
            return np.full(len(X), self.media_)
        X = pd.DataFrame(X)[self.usar_]
        return self.est_.predict_proba(X)[:, 1] if self.tarefa == "cls" else self.est_.predict(X)

    def predict(self, X):
        return self.saida(X)


def nota(tarefa: str, y, p) -> float:
    """Classificação: AUC (0,5 = acaso). Regressão: correlação de postos (0 = acaso)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if tarefa == "cls":
        return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan
    if np.std(p) == 0 or len(y) < 3:
        return 0.0
    return float(spearmanr(p, y).statistic)


ACASO = {"cls": 0.5, "reg": 0.0}
METRICA = {"cls": "AUC", "reg": "Spearman"}     # métrica de acerto de cada tarefa


def _scorer(tarefa):
    return lambda est, X, y: nota(tarefa, y, est.saida(X))


def analisar(base: pd.DataFrame, cols: list[str], prazo: int, tarefa: str = "cls", familia: str = "auto",
             progresso=None) -> dict:
    """Seleciona a combinação de grupos, compara modelos, testa e faz a leitura de agora."""
    prog = progresso or (lambda *_: None)
    base = base.reset_index(drop=True)
    alvo = "y" if tarefa == "cls" else "ret"
    lab = base[alvo].notna().to_numpy()
    dobras, dev, te, corte = dividir(base["Date"], prazo)
    dev_l, te_l = dev & lab, te & lab
    sem_dado = [c for c in cols if base.loc[dev_l, c].nunique() < 2]
    cols = [c for c in cols if c not in sem_dado]
    if not cols:
        raise ValueError("Nenhum indicador com dados suficientes.")
    if not dobras:
        raise ValueError("Histórico curto demais para validar (escolha 'Todas juntas' ou um prazo menor).")
    todas = list(cols)
    y = base[alvo]
    acaso = ACASO[tarefa]

    def validar(cs: list[str], fam: str, valor) -> dict:
        tr_s, va_s = [], []
        for tr, va in dobras:
            tr, va = tr & lab, va & lab
            m_ = Modelo(tarefa, fam, valor, int(tr.sum())).fit(base.loc[tr, cs], y[tr])
            tr_s.append(nota(tarefa, y[tr], m_.saida(base.loc[tr, cs])))
            va_s.append(nota(tarefa, y[va], m_.saida(base.loc[va, cs])))
        va_a = np.array(va_s)
        return {"Treino": np.nanmean(tr_s), "Validação": np.nanmean(va_a),
                "± erro padrão": np.nanstd(va_a) / np.sqrt(max(len(va_a), 1)),
                "Pior janela": np.nanmin(va_a) if len(va_a) else np.nan}

    # --- 1) combinação de grupos (seleção para a frente, com Gradient Boosting simples)
    grupos_disp = [g for g in GRUPOS if any(GRUPO[c] == g for c in cols)]
    escolhidos, passos, melhor = [], [], acaso
    total = len(grupos_disp) * (len(grupos_disp) + 1) // 2
    feitos = 0
    while True:
        tentativas = []
        for g in [g for g in grupos_disp if g not in escolhidos]:
            cs = [c for c in cols if GRUPO[c] in escolhidos + [g]]
            tentativas.append((validar(cs, "gb", 1), g))
            feitos += 1
            prog(0.05 + 0.35 * feitos / total, f"Testando combinação com {g}")
        if not tentativas:
            break
        r, g = max(tentativas, key=lambda t: np.nan_to_num(t[0]["Validação"], nan=-1))
        if r["Validação"] < melhor + max(0.002, r["± erro padrão"]):
            break
        escolhidos.append(g)
        melhor = r["Validação"]
        passos.append({"Passo": len(escolhidos), "Adicionou": g, f"{METRICA[tarefa]} validação": r["Validação"],
                       "± erro padrão": r["± erro padrão"],
                       "Outras opções": " · ".join(f"{gg} {rr['Validação']:.3f}" for rr, gg in
                                                   sorted(tentativas, key=lambda t: -np.nan_to_num(t[0]["Validação"]))
                                                   if gg != g)})
    selecao = pd.DataFrame(passos)
    if escolhidos:
        cols = [c for c in cols if GRUPO[c] in escolhidos]
    X = base[cols]

    # --- 2) modelos × complexidade (regra de 1 erro padrão dentro de cada modelo)
    familias = list(FAMILIAS) if familia == "auto" else [familia]
    linhas_cmp, k_total, k = [], sum(len(FAMILIAS[f]["grade"]) for f in familias), 0
    melhores = {}
    for fam in familias:
        curva = []
        for v in FAMILIAS[fam]["grade"]:
            curva.append({"valor": v, **validar(cols, fam, v)})
            k += 1
            prog(0.4 + 0.3 * k / k_total, f"Testando {nome_modelo(fam, tarefa)} ({descricao_complexidade(fam, v)})")
        curva = pd.DataFrame(curva)
        b = int(curva["Validação"].idxmax())
        esc = int(curva.index[curva["Validação"] >= curva.loc[b, "Validação"] - curva.loc[b, "± erro padrão"]][0])
        melhores[fam] = curva.loc[esc].to_dict()
        for i, row in curva.iterrows():
            linhas_cmp.append({"Modelo": nome_modelo(fam, tarefa), "fam": fam,
                               "Complexidade": descricao_complexidade(fam, row["valor"]), "valor": row["valor"],
                               "Treino": row["Treino"], "Validação": row["Validação"],
                               "± erro padrão": row["± erro padrão"], "Pior janela": row["Pior janela"],
                               "melhor_da_familia": i == esc})
    comparacao = pd.DataFrame(linhas_cmp)
    fam = max(melhores, key=lambda f: np.nan_to_num(melhores[f]["Validação"], nan=-9))
    valor = melhores[fam]["valor"]
    valor = int(valor) if fam in ("gb", "rf") else float(valor)
    comparacao["Escolhido"] = np.where((comparacao["fam"] == fam) & comparacao["melhor_da_familia"], "⭐", "")

    # --- 3) teste final (uma vez)
    prog(0.72, "Teste final")
    m = Modelo(tarefa, fam, valor, int(dev_l.sum())).fit(X[dev_l], y[dev_l])
    p_dev = m.saida(X[dev_l])
    p_te = m.saida(X[te_l])
    y_te = y[te_l].to_numpy()
    r_te = np.expm1(base.loc[te_l, "ret"].to_numpy())
    sobe_te = base.loc[te_l, "y"].to_numpy()
    teste = {"nota": nota(tarefa, y_te, p_te), "nota_val": float(melhores[fam]["Validação"]),
             "n": int(te_l.sum()), "inicio": corte, "fim": base.loc[te_l, "Date"].max(),
             "subiu_todos": float(sobe_te.mean()), "ret_todos": float(r_te.mean())}
    if tarefa == "cls":
        # COMPRA / VENDA = quando a chance prevista está entre os 25% mais altos / mais baixos
        # do que o modelo costumava prever no período de desenvolvimento
        lim_c, lim_v = np.quantile(p_dev, 0.75), np.quantile(p_dev, 0.25)
        c, v = p_te >= lim_c, p_te <= lim_v
        teste.update(lim_compra=float(lim_c), lim_venda=float(lim_v), n_compra=int(c.sum()), n_venda=int(v.sum()),
                     acerto_compra=float(sobe_te[c].mean()) if c.any() else np.nan,
                     acerto_venda=float(1 - sobe_te[v].mean()) if v.any() else np.nan,
                     ret_compra=float(r_te[c].mean()) if c.any() else np.nan,
                     ret_venda=float(r_te[v].mean()) if v.any() else np.nan)
    else:
        prev = np.expm1(p_te)
        q_hi, q_lo = np.quantile(prev, 0.8), np.quantile(prev, 0.2)
        teste.update(acerto_direcao=float(((prev > 0) == (r_te > 0)).mean()),
                     ret_top=float(r_te[prev >= q_hi].mean()), ret_fundo=float(r_te[prev <= q_lo].mean()),
                     erro_tipico=float(np.median(np.abs(prev - r_te))),
                     erro_ingenuo=float(np.median(np.abs(np.expm1(np.median(y[dev_l])) - r_te))))

    # --- 4) o que mais pesa (permutação no teste)
    prog(0.8, "Medindo o peso de cada indicador")
    amostra = np.flatnonzero(te_l)
    if amostra.size > 5000:
        amostra = np.random.default_rng(0).choice(amostra, 5000, replace=False)
    imp = permutation_importance(m, X.iloc[amostra], y.iloc[amostra], scoring=_scorer(tarefa), n_repeats=3,
                                 random_state=0)
    importancia = pd.DataFrame({"col": cols, "Indicador": [NOME[c] for c in cols], "Grupo": [GRUPO[c] for c in cols],
                                "Peso": imp.importances_mean, "dp": imp.importances_std}).sort_values("Peso", ascending=False)

    # --- 5) efeito de cada indicador (dependência parcial)
    prog(0.86, "Calculando o efeito de cada indicador")
    rng = np.random.default_rng(1)
    idx_dev = np.flatnonzero(dev_l)
    amostra_dev = X.iloc[rng.choice(idx_dev, min(2000, idx_dev.size), replace=False)].copy()
    efeitos = {}
    for c in cols:
        if isinstance(INDICADORES[c][3], dict):
            grade = np.array(sorted(INDICADORES[c][3]), float)
        else:
            vals = X.loc[dev_l, c].dropna()
            if vals.empty:
                continue
            grade = np.unique(np.quantile(vals, np.linspace(0.05, 0.95, 12)))
        saida = []
        for g in grade:
            tmp = amostra_dev.copy()
            tmp[c] = g
            o = m.saida(tmp)
            saida.append(float(o.mean()) if tarefa == "cls" else float(np.expm1(o).mean()))
        efeitos[c] = (grade, np.array(saida))

    # --- 6) combinações históricas
    prog(0.92, "Procurando combinações")
    fx = faixas(base.loc[dev], todas)
    comb = combinacoes(base, todas, fx, dev, te, min_dev=max(150, int(0.004 * dev_l.sum())))

    # --- 7) leitura de agora: mesmo modelo, treinado com todo o histórico
    prog(0.96, "Leitura de agora")
    m_all = Modelo(tarefa, fam, valor, int(lab.sum())).fit(X[lab], y[lab])
    ult = base.groupby("token").tail(1)
    o = m_all.saida(ult[cols])
    agora = pd.DataFrame({"Token": ult["token"].to_numpy(), "Data": ult["Date"].dt.date.to_numpy()})
    if tarefa == "cls":
        p_all = m_all.saida(X[lab])
        lc, lv = np.quantile(p_all, 0.75), np.quantile(p_all, 0.25)
        agora["Sinal"] = np.where(o >= lc, "🟢 COMPRA", np.where(o <= lv, "🔴 VENDA", "⚪ neutro"))
        agora["Chance de alta %"] = (100 * o).round(1)
        agora = agora.sort_values("Chance de alta %", ascending=False)
        limites = (float(lc), float(lv))
    else:
        agora["Variação prevista %"] = (100 * np.expm1(o)).round(2)
        agora["Sinal"] = np.where(agora["Variação prevista %"] > 0, "🟢 alta", "🔴 queda")
        agora = agora.sort_values("Variação prevista %", ascending=False)
        limites = None
    cods_agora = {tok: {c: int(codigos([v_], fx[c])[0]) for c, v_ in linha.items() if c in fx}
                  for tok, linha in ult.set_index("token")[todas].iterrows()}

    return {"tarefa": tarefa, "fam": fam, "valor": valor, "modelo": nome_modelo(fam, tarefa),
            "comparacao": comparacao, "selecao": selecao, "grupos_escolhidos": escolhidos,
            "teste": teste, "importancia": importancia, "efeitos": efeitos, "faixas": fx, "comb": comb,
            "agora": agora.reset_index(drop=True), "limites_agora": limites, "cods_agora": cods_agora,
            "valores_agora": ult.set_index("token")[todas], "n_linhas": int(lab.sum()),
            "n_tokens": int(base["token"].nunique()), "cols": cols, "cols_todas": todas, "sem_dado": sem_dado}


def leitura_token(res: dict, base: pd.DataFrame, token: str) -> pd.DataFrame:
    """Valor de cada indicador agora, a faixa e o que aconteceu historicamente nela."""
    if token not in res["valores_agora"].index:
        return pd.DataFrame()
    v = res["valores_agora"].loc[token]
    y = base["y"].to_numpy()
    r = np.expm1(base["ret"].to_numpy())
    ok = ~np.isnan(y)
    linhas = []
    for c in res["cols_todas"]:
        fx = res["faixas"].get(c)
        if fx is None or pd.isna(v[c]):
            linhas.append((NOME[c], GRUPO[c], "—", "sem dado", np.nan, np.nan, 0))
            continue
        k = res["cods_agora"][token][c]
        m = ok & (codigos(base[c], fx) == k)
        val = fx[2][k] if fx[0] == "cat" else _fmt_num(float(v[c]), INDICADORES[c][2])
        linhas.append((NOME[c], GRUPO[c], val, fx[2][k], 100 * y[m].mean() if m.any() else np.nan,
                       100 * r[m].mean() if m.any() else np.nan, int(m.sum())))
    return pd.DataFrame(linhas, columns=["Indicador", "Grupo", "Agora", "Faixa", "Subiu % nessa faixa",
                                         "Variação média %", "Dias"])


def mapa_par(base: pd.DataFrame, res: dict, a: str, b: str, periodo: str = "todo") -> dict:
    """Matriz (faixa de A × faixa de B): dias em que subiu, dias em que caiu e variação média."""
    fa, fb = res["faixas"][a], res["faixas"][b]
    ca, cb = codigos(base[a], fa), codigos(base[b], fb)
    r = base["ret"].to_numpy()
    m = ~np.isnan(r) & (ca >= 0) & (cb >= 0)
    if periodo != "todo":
        _, dev, te, _ = dividir(base["Date"], 1)
        m &= dev if periodo == "antes" else te
    na, nb = len(fa[2]), len(fb[2])
    k = ca[m] * nb + cb[m]
    tam = na * nb
    sub = np.bincount(k, weights=(r[m] > 0).astype(float), minlength=tam).reshape(na, nb).astype(int)
    cai = np.bincount(k, weights=(r[m] < 0).astype(float), minlength=tam).reshape(na, nb).astype(int)
    cnt = np.bincount(k, minlength=tam).reshape(na, nb)
    soma = np.bincount(k, weights=np.expm1(r[m]), minlength=tam).reshape(na, nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        pct_sub = 100 * sub / cnt
        ret_med = 100 * soma / cnt
    base_pct = 100 * (r[m] > 0).mean() if m.any() else np.nan
    return {"sub": sub, "cai": cai, "cnt": cnt, "pct_sub": pct_sub, "ret_med": ret_med,
            "rot_a": fa[2], "rot_b": fb[2], "base_pct": base_pct}


# =========================================================================== #
# Gráficos
# =========================================================================== #

def fig_importancia(imp: pd.DataFrame, tarefa: str = "cls") -> go.Figure:
    met = METRICA[tarefa]
    d = imp.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=d["Peso"], y=d["Indicador"], orientation="h", marker_color=[COR_GRUPO[g] for g in d["Grupo"]],
        error_x=dict(type="data", array=d["dp"], color="#888"), customdata=d["Grupo"], showlegend=False,
        hovertemplate=f"%{{y}}<br>%{{customdata}}<br>queda no {met} ao embaralhar: %{{x:.4f}}<extra></extra>"))
    for g, cor in COR_GRUPO.items():
        if g in set(d["Grupo"]):
            fig.add_trace(go.Bar(x=[None], y=[None], marker_color=cor, name=g, showlegend=True))
    fig.update_layout(template="plotly_dark", height=28 * len(d) + 140, barmode="overlay",
                      xaxis_title=f"Quanto o {met} cai sem este indicador", margin=dict(l=10, r=20, t=40, b=40),
                      legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom"))
    return fig


def fig_efeitos(res: dict, top: int = 6, ref: float | None = None) -> go.Figure:
    cls = res["tarefa"] == "cls"
    cols = [c for c in res["importancia"]["col"] if c in res["efeitos"]][:top]
    ncol = min(3, max(len(cols), 1))
    nlin = int(np.ceil(len(cols) / ncol)) or 1
    fig = make_subplots(rows=nlin, cols=ncol, subplot_titles=[NOME[c] for c in cols],
                        vertical_spacing=0.2, horizontal_spacing=0.08)
    for i, c in enumerate(cols):
        g, p = res["efeitos"][c]
        yv = 100 * p
        r, k = i // ncol + 1, i % ncol + 1
        cor = COR_GRUPO[GRUPO[c]]
        if isinstance(INDICADORES[c][3], dict):
            fig.add_trace(go.Bar(x=[INDICADORES[c][3][int(v)] for v in g], y=yv, marker_color=cor, showlegend=False,
                                 hovertemplate="%{x}: %{y:.2f}%<extra></extra>"), row=r, col=k)
        else:
            fig.add_trace(go.Scatter(x=g, y=yv, mode="lines+markers", showlegend=False, line=dict(color=cor, width=2.5),
                                     hovertemplate="%{x:.2f}: %{y:.2f}%<extra></extra>"), row=r, col=k)
        if ref is not None:
            fig.add_hline(y=ref, line=dict(dash="dot", color="#888", width=1), row=r, col=k)
    fig.update_yaxes(title_text="chance de alta %" if cls else "variação prevista %", col=1)
    fig.update_layout(template="plotly_dark", height=260 * nlin + 60, margin=dict(l=50, r=20, t=50, b=30))
    return fig


def fig_mapa(mp: dict, nome_a: str, nome_b: str, token: str | None = None, marca: tuple | None = None,
             mostrar_ret: bool = False, min_dias: int = 30) -> go.Figure:
    """Célula: ▲ % subiu / ▼ % caiu e, embaixo, dias de alta / dias de queda."""
    na, nb = mp["cnt"].shape
    ref = mp["base_pct"]
    texto, hover = [], []
    for i in range(na):
        lt, lh = [], []
        for j in range(nb):
            n, s, c = mp["cnt"][i, j], mp["sub"][i, j], mp["cai"][i, j]
            if n == 0:
                lt.append(""); lh.append("sem dados")
                continue
            ps, pc = 100 * s / n, 100 * c / n
            extra = f"<br><sub>média {mp['ret_med'][i, j]:+.1f}%</sub>" if mostrar_ret else ""
            if n >= min_dias:
                lt.append(f"<b>▲{ps:.0f}% / ▼{pc:.0f}%</b><br><sub>{s} / {c} dias</sub>{extra}")
            else:
                lt.append(f"<sub>poucos dados<br>{s} / {c} dias</sub>")
            lh.append(f"subiu {ps:.1f}% ({s} dias) · caiu {pc:.1f}% ({c} dias)<br>"
                      f"variação média {mp['ret_med'][i, j]:+.2f}%")
        texto.append(lt)
        hover.append(lh)
    z = np.where(mp["cnt"] >= min_dias, mp["pct_sub"], np.nan)
    fig = go.Figure(go.Heatmap(
        z=z, x=mp["rot_b"], y=mp["rot_a"], text=texto, texttemplate="%{text}", customdata=hover,
        textfont=dict(size=13),
        colorscale="RdYlGn", zmid=ref, zmin=ref - 20, zmax=ref + 20, hoverongaps=False,
        colorbar=dict(title="▲ subiu %"),
        hovertemplate=f"{nome_a}: %{{y}}<br>{nome_b}: %{{x}}<br>%{{customdata}}<extra></extra>"))
    if marca is not None and marca[0] >= 0 and marca[1] >= 0:
        i, j = marca
        fig.add_shape(type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
                      line=dict(color="#00e5ff", width=6))
        fig.add_annotation(x=mp["rot_b"][j], y=mp["rot_a"][i], text=f"📍 {token or ''} AGORA", showarrow=False,
                           yshift=34, font=dict(size=13, color="#000"), bgcolor="#00e5ff",
                           bordercolor="#00e5ff", borderpad=3)
    fig.update_layout(template="plotly_dark", height=130 + 70 * na, xaxis_title=nome_b, yaxis_title=nome_a,
                      margin=dict(l=10, r=10, t=40, b=40))
    return fig


def fig_comparacao(cmp: pd.DataFrame, tarefa: str) -> go.Figure:
    """Treino × validação de cada modelo e complexidade (mostra under/overfitting)."""
    fig = go.Figure()
    rot = [f"{m}<br>{c}" for m, c in zip(cmp["Modelo"], cmp["Complexidade"])]
    fig.add_trace(go.Bar(x=rot, y=cmp["Treino"], name="Treino (dados que o modelo viu)", marker_color="#ffb74d"))
    fig.add_trace(go.Bar(x=rot, y=cmp["Validação"], name="Validação (dados que não viu)", marker_color="#26a69a",
                         error_y=dict(type="data", array=cmp["± erro padrão"])))
    fig.add_hline(y=ACASO[tarefa], line=dict(dash="dot", color="#bbb"), annotation_text="acaso")
    fig.update_layout(template="plotly_dark", height=360, barmode="group",
                      yaxis_title=METRICA[tarefa],
                      legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom"), margin=dict(l=50, r=20, t=40, b=60))
    return fig
