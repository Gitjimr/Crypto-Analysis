"""
linhas.py — Linhas de tendência, suportes e resistências automáticos
=====================================================================

Detecta LTA/LTB e zonas horizontais no PREÇO e no RSI, ranqueia por força,
gera sinais do candle atual e mede (backtest) se toques adicionais realmente
aumentam a chance de repique no SEU ativo.

Pipeline
--------
1. Pivôs (topos/fundos) no estilo ``ta.pivothigh/pivotlow`` do TradingView:
   o ponto i é topo se for o maior valor em [i-esq, i+dir]. Só fica
   CONFIRMADO dir barras depois -> sem repaint no backtest.
   (``scipy.signal.find_peaks`` com ``prominence``/``distance`` olha a série
   inteira, inclusive o futuro, e por isso "repinta"; aqui a proeminência é
   medida só dentro da janela do pivô.)
2. Candidatas: toda dupla de fundos (suporte) ou topos (resistência) dentro
   de ``span_max`` barras. A linha é descartada se algum FECHAMENTO entre as
   âncoras a atravessar além da tolerância (pavio pode furar, corpo não).
3. Extensão: a linha segue até o primeiro fechamento além da tolerância
   (rompimento). Toques = pivôs a até ``tol`` da linha.
3b. Depois do rompimento a linha CONTINUA (troca de papel: LTB rompida para cima
   vira suporte; LTA rompida para baixo vira resistência). O preço voltar até
   ela sem fechar do lado antigo é o RETESTE (compra no reteste da LTB rompida,
   venda no reteste da LTA perdida). Fechar de volta do lado antigo além da
   tolerância = rompimento falso.
4. Score 0-100 = toques + duração + recência + precisão + ativa (pesos em
   ``PESOS``). O componente "toques" depende de ``modo_forca``:
     - "toques":     mais toques = mais forte (retorno decrescente, log)
     - "virgem":     linha nunca testada (só as 2 âncoras) = mais forte
     - "calibrado":  usa a taxa de repique medida no backtest deste ativo
5. Supressão de duplicadas (NMS): mantém a de maior score entre linhas que
   ficam a menos de 2·tol uma da outra nas duas pontas; descarta linhas ativas
   longe demais do preço atual (``dist_max``).

Preço é tratado em escala log por padrão (linha reta no log = taxa de
crescimento constante, o que faz mais sentido em cripto) e a tolerância é
uma fração do ATR. No RSI a tolerância é em pontos.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as _janelas

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --------------------------------------------------------------------------- #
# Parâmetros
# --------------------------------------------------------------------------- #

PESOS = {"toques": 0.35, "duracao": 0.20, "recencia": 0.20, "precisao": 0.10, "ativa": 0.15}


@dataclass(frozen=True)
class ParamsLinhas:
    esquerda: int = 3          # barras à esquerda para validar um pivô
    direita: int = 3           # barras à direita (= atraso de confirmação)
    prom_min: float = 1.0      # proeminência mínima do pivô, em múltiplos de tol
    tol_atr: float = 0.5       # tolerância do preço, em ATR(14)
    tol_rsi: float = 2.0       # tolerância do RSI, em pontos
    gap_min: int = 6           # distância mínima entre as 2 âncoras (barras)
    span_max: int = 400        # distância máxima entre as 2 âncoras (barras)
    max_linhas: int = 3        # linhas ATIVAS exibidas por tipo (sup/res) e série
    max_rompidas: int = 2      # linhas rompidas recentemente exibidas por tipo
    janela_rompidas: int = 40  # barras: "rompida recentemente" para exibir
    janela_reteste: int = 120  # barras após o rompimento em que a linha segue no gráfico p/ reteste
    janela_sinal: int = 5      # barras: rompimento/divergência conta como sinal atual
    meia_vida: int = 90        # barras: meia-vida da recência do último toque
    modo_forca: str = "toques"  # "toques" | "virgem" | "calibrado"
    log: bool = True           # preço em escala log
    projecao: int = 30         # barras projetadas para o futuro no gráfico
    horizonte: int = 20        # backtest: barras para resolver repique x rompimento
    alvo: float = 2.0          # backtest: afastamento (em tol) que define repique/rompimento
    max_zonas: int = 4         # zonas horizontais exibidas
    dist_max: float = 6.0      # linha ativa só é exibida se estiver a até N·tol do fechamento atual


# --------------------------------------------------------------------------- #
# Indicadores base (Wilder, igual ao TradingView)
# --------------------------------------------------------------------------- #

def rsi_wilder(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    ganho = delta.clip(lower=0).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    perda = (-delta.clip(upper=0)).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    rs = ganho / perda.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(ganho.notna())


def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev).abs(), (low - prev).abs()))
    return tr.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()


# --------------------------------------------------------------------------- #
# Pivôs
# --------------------------------------------------------------------------- #

def pivos(x: np.ndarray, esq: int, dir_: int, tipo: str) -> tuple[np.ndarray, np.ndarray]:
    """Índices dos pivôs e sua proeminência local (mesma unidade de x).

    tipo="alto": topo se x[i] for o 1º máximo de x[i-esq : i+dir+1].
    tipo="baixo": fundo, análogo com o mínimo.
    """
    x = np.asarray(x, dtype=float)
    n, w = len(x), esq + dir_ + 1
    if n < w:
        return np.array([], dtype=int), np.array([])
    s = x if tipo == "alto" else -x
    win = _janelas(s, w)
    ok = np.isfinite(win).all(axis=1) & (win.argmax(axis=1) == esq)
    idx = np.flatnonzero(ok) + esq
    if idx.size == 0:
        return idx, np.array([])
    wv = win[ok]
    base_esq = wv[:, :esq].min(axis=1) if esq > 0 else wv[:, esq]
    base_dir = wv[:, esq + 1:].min(axis=1) if dir_ > 0 else wv[:, esq]
    prom = wv[:, esq] - np.maximum(base_esq, base_dir)
    return idx, prom


# --------------------------------------------------------------------------- #
# Séries de trabalho
# --------------------------------------------------------------------------- #

def _series(df: pd.DataFrame, serie: str, p: ParamsLinhas):
    """Retorna (alta, baixa, fechamento, tol, para_plot) no espaço de cálculo."""
    if serie == "preco":
        h, l, c = (df[k].astype(float) for k in ("High", "Low", "Close"))
        atr = atr_wilder(h, l, c).bfill()
        if p.log:
            H, L, C = np.log(h.to_numpy()), np.log(l.to_numpy()), np.log(c.to_numpy())
            tol = (p.tol_atr * atr / c).to_numpy()
            conv = np.exp
        else:
            H, L, C = h.to_numpy(), l.to_numpy(), c.to_numpy()
            tol = (p.tol_atr * atr).to_numpy()
            conv = lambda v: v  # noqa: E731
        return H, L, C, tol, conv
    r = df["RSI"] if "RSI" in df.columns else rsi_wilder(df["Close"])
    R = r.to_numpy(dtype=float)
    return R, R, R, np.full(len(R), p.tol_rsi), (lambda v: v)


# --------------------------------------------------------------------------- #
# Linhas candidatas
# --------------------------------------------------------------------------- #

def _candidatas(ext, C, tol, piv, val, tipo, p: ParamsLinhas) -> list[dict]:
    """Gera todas as linhas válidas ligando pares de pivôs do mesmo tipo."""
    n = len(C)
    sinal = 1.0 if tipo == "sup" else -1.0
    out = []
    for a in range(len(piv)):
        i = piv[a]
        t_rel = np.arange(n - i, dtype=float)          # t - i, para t em [i, n)
        Cs, tols = C[i:], tol[i:]
        for b in range(a + 1, len(piv)):
            j = piv[b]
            g = j - i
            if g < p.gap_min:
                continue
            if g > p.span_max:
                break
            m = (val[b] - val[a]) / g
            linha = val[a] + m * t_rel
            viol = sinal * (Cs - linha) < -tols           # fechamento além da tolerância
            if viol[: g + 1].any():
                continue
            pos = viol[g + 1:]
            brk = i + g + 1 + int(pos.argmax()) if pos.any() else -1
            fim = brk if brk >= 0 else n - 1
            # toques: pivôs em [i, fim] perto da linha
            lo, hi = np.searchsorted(piv, i), np.searchsorted(piv, fim, side="right")
            pk = piv[lo:hi]
            dist = np.abs(val[lo:hi] - (val[a] + m * (pk - i))) / tol[pk]
            perto = dist <= 1.0
            out.append({
                "tipo": tipo, "i": int(i), "j": int(j), "m": float(m), "b0": float(val[a]),
                "brk": int(brk), "fim": int(fim),
                "toques_idx": pk[perto].astype(int),
                "toques_dist": dist[perto],
            })
    return out


def _score(cands: list[dict], n: int, p: ParamsLinhas, calib: dict | None) -> None:
    """Preenche score (0-100) e componentes em cada candidata (in-place)."""
    for L in cands:
        k = len(L["toques_idx"])                    # inclui as 2 âncoras
        extra = max(k - 2, 0)
        if p.modo_forca == "virgem":
            c_toq = 1.0 if extra == 0 else 1.0 / (1 + extra)
        elif p.modo_forca == "calibrado" and calib:
            c_toq = calib.get(min(k, 5), 0.5)
        else:
            c_toq = min(1.0, np.log1p(extra) / np.log1p(4))
        dur = L["fim"] - L["i"]
        c_dur = min(1.0, np.log1p(dur) / np.log1p(max(n, 2)))
        ultimo = int(L["toques_idx"].max())
        c_rec = float(np.exp(-np.log(2) * (n - 1 - ultimo) / p.meia_vida))
        d_extra = np.sort(L["toques_dist"])[2:] if k > 2 else np.array([])
        c_prec = float(1 - d_extra.mean()) if d_extra.size else 0.5
        c_atv = 1.0 if L["brk"] < 0 else 0.0
        comp = {"toques": c_toq, "duracao": c_dur, "recencia": c_rec, "precisao": c_prec, "ativa": c_atv}
        L["score"] = 100 * sum(PESOS[c] * v for c, v in comp.items())
        L["n_toques"] = k
        L["ultimo_toque"] = ultimo
        L["comp"] = comp


def _valor(L: dict, t) -> float:
    return L["b0"] + L["m"] * (np.asarray(t) - L["i"])


def _selecionar(cands: list[dict], C, tol, serie: str, p: ParamsLinhas) -> list[dict]:
    """NMS: melhores linhas ativas + rompidas recentes, sem duplicatas.

    Linhas ativas muito longe do preço atual (> dist_max·tol) não ajudam a decidir
    nada agora e são ignoradas; no RSI, linhas projetadas fora de 0-100 também.
    """
    n = len(C)
    tol_ref = float(np.nanmedian(tol[-100:]))
    escolhidas = []
    for tipo in ("sup", "res"):
        grupo = sorted((c for c in cands if c["tipo"] == tipo), key=lambda c: -c["score"])
        ativas, rompidas = [], []
        for L in grupo:
            ativa = L["brk"] < 0
            v = float(_valor(L, n - 1))
            longe = abs(C[-1] - v) / tol[-1] > p.dist_max or (serie == "rsi" and not 0 <= v <= 100)
            if ativa and longe:
                continue
            if not ativa:
                pos = L.get("pos") or {}
                if L["brk"] < n - 1 - p.janela_reteste or not pos:
                    continue
                if pos["estado"] == "falhou" and pos["inval"] < n - 1 - 2 * p.janela_sinal:
                    continue                                       # falso rompimento antigo
                # rompidas continuam no gráfico mesmo longe do preço: é o nível do reteste
                if serie == "rsi" and (not 5 <= v <= 95 or abs(C[-1] - v) > 20):
                    continue
            alvo = ativas if ativa else rompidas
            limite = p.max_linhas if ativa else p.max_rompidas
            if len(alvo) >= limite:
                if len(ativas) >= p.max_linhas and len(rompidas) >= p.max_rompidas:
                    break
                continue
            dup = False
            for K in ativas + rompidas:
                t1, t2 = max(L["i"], K["i"]), min(L["fim"], K["fim"])
                if t2 < t1:
                    continue
                if (abs(_valor(L, t1) - _valor(K, t1)) <= 2 * tol_ref
                        and abs(_valor(L, t2) - _valor(K, t2)) <= 2 * tol_ref):
                    dup = True
                    break
            if not dup:
                alvo.append(L)
        escolhidas += ativas + rompidas
    return escolhidas


def _nome(tipo: str, m: float, serie: str) -> str:
    if tipo == "sup":
        nome = "LTA" if m > 0 else "Suporte descendente"
    else:
        nome = "LTB" if m < 0 else "Resistência ascendente"
    return f"{nome} (RSI)" if serie == "rsi" else nome


# --------------------------------------------------------------------------- #
# Backtest: toques prévios x probabilidade de repique
# --------------------------------------------------------------------------- #

def wilson(k: np.ndarray, n: np.ndarray, z: float = 1.96):
    n = np.maximum(n, 1)
    ph = k / n
    den = 1 + z ** 2 / n
    centro = (ph + z ** 2 / (2 * n)) / den
    meio = z * np.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2)) / den
    return centro - meio, centro + meio


def _taxa_base(C, tol, H, A):
    """P(fechamento andar +A·tol antes de -A·tol em até H barras), barra aleatória."""
    n = len(C)
    if n <= H + 1:
        return np.nan, np.nan
    fut = _janelas(C[1:], H)[: n - H]
    rel = (fut - C[: n - H, None]) / tol[: n - H, None]
    up, dn = rel >= A, rel <= -A
    fu = np.where(up.any(1), up.argmax(1), H)
    fd = np.where(dn.any(1), dn.argmax(1), H)
    res = (fu < H) | (fd < H)
    if not res.any():
        return np.nan, np.nan
    p_up = float((fu[res] < fd[res]).mean())
    return p_up, 1 - p_up


def estudo_toques(cands, ext_lo, ext_hi, C, tol, p: ParamsLinhas,
                  C_res=None, tol_res=None) -> tuple[pd.DataFrame, pd.DataFrame, tuple]:
    """Event study causal.

    Para cada linha, depois que a 2ª âncora é CONFIRMADA, cada nova entrada do
    preço na zona da linha (|dist| <= tol) é um "teste". Resultado em até
    ``horizonte`` barras: repique se, a partir do fechamento do teste, o preço
    anda ``alvo``·tol a favor da linha antes de andar ``alvo``·tol contra.
    Conta-se quantos repiques a linha já tinha ANTES do teste (2 = virgem).
    A "taxa base" é a mesma conta feita em todas as barras: se a linha não tiver
    poder preditivo, repique % ≈ base (edge ≈ 0). Em dados de passeio aleatório
    o edge fica perto de zero, que é a checagem de sanidade do método.

    ``C_res``/``tol_res``: série onde o resultado é medido. Para linhas do RSI
    passe o PREÇO: o que interessa é se o toque no RSI antecipa o preço (medir o
    próprio RSI dá "edge" falso, porque o RSI é limitado e reverte à média).

    Só conta como toque prévio o repique já RESOLVIDO antes do teste atual —
    senão janelas sobrepostas vazam o futuro e inflam o efeito dos toques.
    """
    n, H, A = len(C), p.horizonte, p.alvo
    Co = C if C_res is None else C_res
    to = tol if tol_res is None else tol_res
    ev = []
    for L in cands:
        sup = L["tipo"] == "sup"
        sinal = 1.0 if sup else -1.0
        ext = ext_lo if sup else ext_hi
        ini = L["j"] + p.direita + 1
        if ini >= n - 2:
            continue
        t = np.arange(ini, n)
        linha = _valor(L, t)
        d = sinal * (ext[ini:] - linha) / tol[ini:]
        zona = d <= 1.0
        entrada = zona & ~np.concatenate(([True], zona[:-1]))
        resolvidos = []                     # instantes em que repiques anteriores se confirmaram
        for e in np.flatnonzero(entrada):
            te = ini + e
            if L["brk"] >= 0 and te > L["brk"]:
                break
            # Resultado medido a partir do FECHAMENTO do candle do teste, com
            # alvos simétricos: assim é diretamente comparável à taxa base de
            # uma barra qualquer (medir a partir da linha favorece o repique,
            # porque o fechamento costuma ficar acima do pavio que tocou).
            w = sinal * (Co[te + 1: te + 1 + H] - Co[te]) / to[te]
            if w.size == 0:
                break
            up, dn = np.flatnonzero(w >= A), np.flatnonzero(w <= -A)
            fu = up[0] if up.size else H + 1
            fd = dn[0] if dn.size else H + 1
            if fu > H and fd > H:
                if w.size < H:
                    break
                continue
            repique = bool(fu < fd)
            previos = 2 + sum(r <= te for r in resolvidos)
            ev.append((L["tipo"], te, previos, te - L["j"], repique,
                       round(float(linha[e]) / float(tol[te]))))
            if repique:
                resolvidos.append(te + 1 + fu)
    eventos = pd.DataFrame(ev, columns=["tipo", "t", "toques_previos", "idade", "repique", "nivel"])
    base_up, base_dn = _taxa_base(Co, to, H, A)
    if eventos.empty:
        return eventos, pd.DataFrame(), (base_up, base_dn)
    # mesma barra + mesmo nível = mesmo evento visto por linhas quase iguais
    eventos = (eventos.sort_values("toques_previos", ascending=False)
               .drop_duplicates(["tipo", "t", "nivel"]).sort_values("t"))
    eventos["faixa"] = eventos["toques_previos"].clip(upper=5)
    g = eventos.groupby(["tipo", "faixa"])["repique"].agg(["sum", "count"]).reset_index()
    lo, hi = wilson(g["sum"].to_numpy(float), g["count"].to_numpy(float))
    g["taxa"] = g["sum"] / g["count"]
    g["ic_lo"], g["ic_hi"] = lo, hi
    g["base"] = np.where(g["tipo"] == "sup", base_up, base_dn)
    g["edge_pp"] = 100 * (g["taxa"] - g["base"])
    rotulo = {2: "2 (virgem)", 3: "3", 4: "4", 5: "5+"}
    tabela = pd.DataFrame({
        "Tipo": g["tipo"].map({"sup": "Suporte", "res": "Resistência"}),
        "Toques antes do teste": g["faixa"].map(rotulo),
        "Testes": g["count"].astype(int),
        "Repique %": (100 * g["taxa"]).round(1),
        "IC95% (min-máx)": [f"{100 * a:.0f}–{100 * b:.0f}" for a, b in zip(lo, hi)],
        "Base aleatória %": (100 * g["base"]).round(1),
        "Edge (p.p.)": g["edge_pp"].round(1),
    })
    return eventos, tabela, (base_up, base_dn)


def _calibracao(eventos: pd.DataFrame, base: tuple, min_eventos: int = 15) -> dict:
    """Converte a taxa de repique por nº de toques em componente 0-1 do score."""
    if eventos.empty:
        return {}
    base_up, base_dn = base
    e = eventos.assign(base=np.where(eventos["tipo"] == "sup", base_up, base_dn))
    e["edge"] = e["repique"].astype(float) - e["base"]
    g = e.groupby("faixa")["edge"].agg(["mean", "count"])
    return {int(k): float(np.clip(0.5 + 2.5 * r["mean"], 0, 1))
            for k, r in g.iterrows() if r["count"] >= min_eventos}


# --------------------------------------------------------------------------- #
# Depois do rompimento: continuação da linha e reteste
# --------------------------------------------------------------------------- #

def _pos_rompimento(L: dict, Hi, Lo, C, tol) -> dict:
    """Acompanha a linha depois do rompimento.

    - lado novo: acima da LTB/resistência rompida para cima; abaixo da LTA/suporte
      rompido para baixo;
    - RETESTE: a máxima/mínima volta até a linha (≤ 1 tol) sem fechar do lado antigo;
    - FALHOU: fechamento de volta do lado antigo além da tolerância (rompimento falso).
    Só usa candles posteriores ao rompimento, um a um (causal).
    """
    n, b = len(C), L["brk"]
    if b < 0:
        return {}
    s = 1.0 if L["tipo"] == "res" else -1.0
    t = np.arange(b + 1, n)
    if t.size == 0:
        return {"estado": "aguardando reteste", "retestes": [], "inval": -1, "lado": s}
    linha = _valor(L, t)
    ext = Lo[b + 1:] if s > 0 else Hi[b + 1:]
    d_ext = s * (ext - linha) / tol[b + 1:]
    d_fech = s * (C[b + 1:] - linha) / tol[b + 1:]
    inv = np.flatnonzero(d_fech < -1.0)
    fim = int(inv[0]) if inv.size else t.size
    zona = d_ext[:fim] <= 1.0
    entrada = zona & ~np.concatenate(([False], zona[:-1]))
    retestes = t[:fim][entrada]
    if inv.size:
        estado = "falhou"
    elif zona.size and zona[-1]:
        estado = "em reteste"
    elif retestes.size:
        estado = "retestada"
    else:
        estado = "aguardando reteste"
    return {"estado": estado, "retestes": retestes.astype(int).tolist(),
            "inval": int(t[inv[0]]) if inv.size else -1, "lado": s}


def estudo_retestes(cands, Hi, Lo, C, tol, p: ParamsLinhas, C_res=None, tol_res=None) -> pd.DataFrame:
    """Backtest: comprar no reteste da LTB rompida / vender no reteste da LTA perdida
    funciona melhor que entrar direto no rompimento? E melhor que uma barra qualquer?

    Para cada linha cujo rompimento aconteceu depois de ela já ser conhecida
    (2ª âncora confirmada), mede a partir do fechamento:
      - "No rompimento": candle do rompimento;
      - "No 1º reteste": primeiro reteste (até ``janela_reteste`` barras depois).
    Acerto = andar ``alvo``·tol a favor do rompimento antes de andar ``alvo``·tol contra,
    em até ``horizonte`` barras. Linhas do RSI são medidas no PREÇO.
    """
    H, A = p.horizonte, p.alvo
    Co = C if C_res is None else C_res
    to = tol if tol_res is None else tol_res
    n = len(C)
    ev = []
    for L in cands:
        b = L["brk"]
        if b < 0 or b <= L["j"] + p.direita or b >= n - 2:
            continue
        pos = _pos_rompimento(L, Hi, Lo, C, tol)
        s = pos["lado"]
        entradas = [("No rompimento", b)]
        ret = [r for r in pos["retestes"] if r - b <= p.janela_reteste]
        if ret:
            entradas.append(("No 1º reteste", ret[0]))
        for quando, te in entradas:
            w = s * (Co[te + 1: te + 1 + H] - Co[te]) / to[te]
            up, dn = np.flatnonzero(w >= A), np.flatnonzero(w <= -A)
            fu = up[0] if up.size else H + 1
            fd = dn[0] if dn.size else H + 1
            if fu > H and fd > H:
                continue
            ev.append((L["tipo"], quando, te, bool(fu < fd), round(float(_valor(L, te)) / float(tol[te]))))
    if not ev:
        return pd.DataFrame()
    e = pd.DataFrame(ev, columns=["tipo", "quando", "t", "acerto", "nivel"]).drop_duplicates(
        ["tipo", "quando", "t", "nivel"])
    base_up, base_dn = _taxa_base(Co, to, H, A)
    g = e.groupby(["tipo", "quando"])["acerto"].agg(["sum", "count"]).reset_index()
    lo, hi = wilson(g["sum"].to_numpy(float), g["count"].to_numpy(float))
    taxa = g["sum"] / g["count"]
    base = np.where(g["tipo"] == "res", base_up, base_dn)
    return pd.DataFrame({
        "Operação": g["tipo"].map({"res": "Compra · LTB/resistência rompida ↑",
                                   "sup": "Venda · LTA/suporte rompido ↓"}),
        "Entrada": g["quando"], "Casos": g["count"].astype(int),
        "Acerto %": (100 * taxa).round(1),
        "IC95% (min-máx)": [f"{100 * a:.0f}–{100 * b_:.0f}" for a, b_ in zip(lo, hi)],
        "Base aleatória %": (100 * base).round(1), "Edge (p.p.)": (100 * (taxa - base)).round(1),
    }).sort_values(["Operação", "Entrada"], ascending=[True, False], ignore_index=True)


# --------------------------------------------------------------------------- #
# Zonas horizontais e divergências
# --------------------------------------------------------------------------- #

def zonas_horizontais(piv_idx, piv_val, n, tol_ref, p: ParamsLinhas) -> list[dict]:
    """Agrupa topos e fundos em níveis horizontais (largura máx. 2·tol)."""
    if len(piv_idx) < 2:
        return []
    ordem = np.argsort(piv_val)
    v, ix = piv_val[ordem], piv_idx[ordem]
    # agrupamento guloso: um nível novo começa quando o valor passa 2·tol do 1º do grupo
    # (evita o "efeito corrente", em que pivôs vizinhos encadeiam uma zona larga demais)
    cortes, inicio = [], v[0]
    for k in range(1, len(v)):
        if v[k] - inicio > 2 * tol_ref:
            cortes.append(k)
            inicio = v[k]
    zonas = []
    for gv, gi in zip(np.split(v, cortes), np.split(ix, cortes)):
        if len(gv) < 2:
            continue
        rec = float(np.exp(-np.log(2) * (n - 1 - gi.max()) / p.meia_vida))
        forca = min(1.0, np.log1p(len(gv) - 1) / np.log1p(5))
        zonas.append({"nivel": float(np.median(gv)), "toques": int(len(gv)),
                      "ultimo": int(gi.max()), "score": 100 * (0.6 * forca + 0.4 * rec)})
    return sorted(zonas, key=lambda z: -z["score"])[: p.max_zonas]


def divergencias(close: np.ndarray, low: np.ndarray, high: np.ndarray, rsi: np.ndarray,
                 esq: int = 5, dir_: int = 5, faixa: tuple[int, int] = (5, 60)) -> pd.DataFrame:
    """Divergências regulares como o indicador nativo do TradingView.

    Pivôs no RSI; compara com o preço nas mesmas barras. Sinal fica disponível
    em ``confirmacao`` = 2º pivô + dir barras.
    """
    rows = []
    for tipo, piv_t, preco, cond in (
        ("Bull", "baixo", low, lambda r0, r1, p0, p1: r1 > r0 and p1 < p0),
        ("Bear", "alto", high, lambda r0, r1, p0, p1: r1 < r0 and p1 > p0),
    ):
        idx, _ = pivos(rsi, esq, dir_, piv_t)
        for a, b in zip(idx[:-1], idx[1:]):
            if faixa[0] <= b - a <= faixa[1] and cond(rsi[a], rsi[b], preco[a], preco[b]):
                rows.append((tipo, int(a), int(b), int(b + dir_)))
    return pd.DataFrame(rows, columns=["tipo", "i0", "i1", "confirmacao"])


# --------------------------------------------------------------------------- #
# Análise completa de uma série
# --------------------------------------------------------------------------- #

def _analisar_serie(df: pd.DataFrame, serie: str, p: ParamsLinhas, backtest: bool) -> dict:
    H, L, C, tol, conv = _series(df, serie, p)
    _, _, C_preco, tol_preco, _ = _series(df, "preco", p) if serie == "rsi" else (None,) * 5
    n = len(C)
    tol_ref = float(np.nanmedian(tol[-100:]))
    piv = {}
    for tipo, x, k in (("res", H, "alto"), ("sup", L, "baixo")):
        idx, prom = pivos(x, p.esquerda, p.direita, k)
        keep = prom >= p.prom_min * tol[idx] if idx.size else np.array([], bool)
        piv[tipo] = (idx[keep], x[idx[keep]])

    cands = []
    for tipo, ext in (("sup", L), ("res", H)):
        cands += _candidatas(ext, C, tol, piv[tipo][0], piv[tipo][1], tipo, p)

    eventos, tabela, calib, tab_reteste = pd.DataFrame(), pd.DataFrame(), None, pd.DataFrame()
    if backtest or p.modo_forca == "calibrado":
        eventos, tabela, base = estudo_toques(cands, L, H, C, tol, p, C_preco, tol_preco)
        calib = _calibracao(eventos, base) if not eventos.empty else None
    if backtest:
        tab_reteste = estudo_retestes(cands, H, L, C, tol, p, C_preco, tol_preco)

    _score(cands, n, p, calib)
    for L_ in cands:                      # continuação/reteste das rompidas recentes
        if L_["brk"] >= n - 1 - p.janela_reteste:
            L_["pos"] = _pos_rompimento(L_, H, L, C, tol)
    sel = _selecionar(cands, C, tol, serie, p)

    linhas = []
    for L_ in sel:
        ativa = L_["brk"] < 0
        pos = L_.get("pos") or {}
        estado = "ativa" if ativa else pos.get("estado", "aguardando reteste")
        if ativa or estado != "falhou":
            t_fim = n - 1 + p.projecao      # rompidas também seguem (continuação p/ reteste)
        else:
            t_fim = pos["inval"]
        v_now = float(_valor(L_, n - 1))
        # sinal: lado "certo" da linha (acima do suporte / abaixo da resistência);
        # para a rompida, o lado certo passa a ser o NOVO lado
        sinal = 1.0 if L_["tipo"] == "sup" else -1.0
        if not ativa:
            sinal = -sinal
        ext_now = (L[-1] if sinal > 0 else H[-1])
        ext_arr = L if L_["tipo"] == "sup" else H
        ext_novo = L if L_["tipo"] == "res" else H
        rt = pos.get("retestes", [])
        linhas.append({
            "serie": serie, "tipo": L_["tipo"], "nome": _nome(L_["tipo"], L_["m"], serie),
            "score": round(L_["score"], 1), "n_toques": L_["n_toques"],
            "status": "ativa" if ativa else f"rompida · {estado}", "estado": estado,
            "t_brk": None if ativa else int(L_["brk"]),
            "y_brk": None if ativa else float(conv(_valor(L_, L_["brk"]))),
            "retestes_idx": rt, "retestes_y": [float(conv(v)) for v in ext_novo[rt]] if rt else [],
            "barras_desde_reteste": int(n - 1 - rt[-1]) if rt else None,
            "barras_desde_falha": int(n - 1 - pos["inval"]) if estado == "falhou" else None,
            "barras_desde_rompimento": None if ativa else int(n - 1 - L_["brk"]),
            "barras_desde_ultimo_toque": int(n - 1 - L_["ultimo_toque"]),
            "inclinacao": (np.expm1(L_["m"]) * 100 if (serie == "preco" and p.log) else L_["m"]),
            "valor_atual": float(conv(v_now)),
            "dist_fech_tol": float(sinal * (C[-1] - v_now) / tol[-1]),
            "dist_ext_tol": float(sinal * (ext_now - v_now) / tol[-1]),
            "t_ini": L_["i"], "t_fim": int(t_fim),
            "y_ini": float(conv(L_["b0"])), "y_fim": float(conv(_valor(L_, t_fim))),
            "toques_idx": L_["toques_idx"].tolist(),
            "toques_y": [float(conv(v)) for v in ext_arr[L_["toques_idx"]]],
            **{f"c_{k}": round(v, 3) for k, v in L_["comp"].items()},
        })

    zonas = []
    if serie == "preco":
        allp = np.concatenate([piv["sup"][0], piv["res"][0]])
        allv = np.concatenate([piv["sup"][1], piv["res"][1]])
        for z in zonas_horizontais(allp, allv, n, tol_ref, p):
            z.update(y=float(conv(z["nivel"])), y0=float(conv(z["nivel"] - tol_ref)),
                     y1=float(conv(z["nivel"] + tol_ref)),
                     dist_tol=float((C[-1] - z["nivel"]) / tol[-1]))
            zonas.append(z)

    return {"linhas": pd.DataFrame(linhas), "zonas": zonas, "eventos": eventos,
            "tabela": tabela, "tabela_reteste": tab_reteste, "n_candidatas": len(cands),
            "n_pivos": {k: int(len(v[0])) for k, v in piv.items()}}


def sinais_linhas(res: dict, n: int, p: ParamsLinhas) -> pd.DataFrame:
    """Sinais do candle atual derivados das linhas/zonas/divergências."""
    s = []
    for serie in ("preco", "rsi"):
        lin = res[serie]["linhas"]
        rot = "RSI" if serie == "rsi" else "Preço"
        for r in (lin.itertuples() if not lin.empty else []):
            forca = f"score {r.score:.0f}, {r.n_toques} toques"
            if r.estado == "ativa":
                if r.dist_ext_tol <= 1.0 and r.dist_fech_tol >= -1.0:
                    lado = "Compra" if r.tipo == "sup" else "Venda"
                    s.append((lado, "🟢" if lado == "Compra" else "🔴",
                              f"{rot} testando {r.nome} ({forca})", r.score))
                continue
            # rompidas: lado do rompimento (LTB ↑ = compra; LTA ↓ = venda)
            lado = "Compra" if r.tipo == "res" else "Venda"
            contra = "Venda" if lado == "Compra" else "Compra"
            k = int(r.barras_desde_rompimento)
            if r.estado == "em reteste":
                s.append((lado, "🎯", f"{rot} RETESTANDO {r.nome} rompida há {k} barra(s): "
                                      f"{lado.lower()} no reteste ({forca})", min(100.0, r.score + 15)))
            elif r.estado == "retestada" and r.barras_desde_reteste <= p.janela_sinal:
                s.append((lado, "✅", f"{rot}: reteste de {r.nome} rompida segurou há "
                                      f"{int(r.barras_desde_reteste)} barra(s) ({forca})", r.score))
            elif r.estado == "falhou" and r.barras_desde_falha <= 2 * p.janela_sinal:
                s.append((contra, "⚠️", f"{rot}: rompimento FALSO de {r.nome} (voltou há "
                                        f"{int(r.barras_desde_falha)} barra(s)) ({forca})", r.score))
            if k <= p.janela_sinal:
                s.append((lado, "🚀" if lado == "Compra" else "⚠️",
                          f"{rot} rompeu {r.nome} há {k} barra(s) ({forca})", r.score))
    for z in res["preco"]["zonas"]:
        if abs(z["dist_tol"]) <= 1.0:
            s.append(("Neutro", "📏", f"Preço dentro da zona horizontal {fmt(z['y'])} "
                      f"({z['toques']} toques)", z["score"]))
    div = res["divergencias"]
    if not div.empty:
        rec = div[(div["confirmacao"] <= n - 1) & (div["confirmacao"] >= n - 1 - p.janela_sinal)]
        for r in rec.itertuples():
            lado = "Compra" if r.tipo == "Bull" else "Venda"
            s.append((lado, "🟢" if lado == "Compra" else "🔴",
                      f"Divergência {'de alta' if r.tipo == 'Bull' else 'de baixa'} RSI x preço "
                      f"(confirmada há {n - 1 - r.confirmacao} barra(s))", 60.0))
    out = pd.DataFrame(s, columns=["Lado", " ", "Sinal", "Força"])
    # linhas quase paralelas geram o mesmo sinal várias vezes: fica só a mais forte
    out["_tipo"] = out["Sinal"].str.replace(r"\s*\(score.*$", "", regex=True)
    out = out.sort_values("Força", ascending=False).drop_duplicates("_tipo").drop(columns="_tipo")
    return out.reset_index(drop=True)


def analisar(df: pd.DataFrame, p: ParamsLinhas = ParamsLinhas(), backtest: bool = False) -> dict:
    """Ponto de entrada. ``df`` com índice de datas e colunas Open/High/Low/Close (+RSI opcional)."""
    df = df.dropna(subset=["Close"])
    if "RSI" not in df.columns:
        df = df.assign(RSI=rsi_wilder(df["Close"]))
    res = {"preco": _analisar_serie(df, "preco", p, backtest),
           "rsi": _analisar_serie(df, "rsi", p, backtest),
           "params": asdict(p), "n": len(df)}
    res["divergencias"] = divergencias(df["Close"].to_numpy(float), df["Low"].to_numpy(float),
                                       df["High"].to_numpy(float), df["RSI"].to_numpy(float),
                                       p.esquerda, p.direita)
    res["sinais"] = sinais_linhas(res, len(df), p)
    return res


# --------------------------------------------------------------------------- #
# Fibonacci (retração + extensões) no período visível
# --------------------------------------------------------------------------- #

NIVEIS_FIBO = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)
EXT_FIBO = (1.272, 1.618, 2.0, 2.618)
CORES_FIBO = ("#787b86", "#f23645", "#ff9800", "#4caf50", "#089981", "#00bcd4", "#787b86",
              "#2962ff", "#9c27b0", "#e91e63", "#f23645")


def fibonacci(df: pd.DataFrame, direcao: str = "auto", extensoes: bool = False) -> dict | None:
    """Retração de Fibonacci entre a MÍNIMA e a MÁXIMA do período recebido.

    Convenção (a mesma do seu gráfico no TradingView):
      - fundo → topo (tendência de alta): 0 = fundo, 1 = topo, extensões ACIMA do topo;
      - topo → fundo (tendência de baixa): 0 = topo, 1 = fundo, extensões ABAIXO do fundo.
    ``direcao="auto"``: usa a ordem em que fundo e topo aconteceram no período.
    """
    if df.empty:
        return None
    i_alto, i_baixo = df["High"].idxmax(), df["Low"].idxmin()
    alto, baixo = float(df.loc[i_alto, "High"]), float(df.loc[i_baixo, "Low"])
    if alto <= baixo:
        return None
    if direcao == "auto":
        direcao = "fundo_topo" if i_baixo < i_alto else "topo_fundo"
    rng = alto - baixo
    razoes = NIVEIS_FIBO + (EXT_FIBO if extensoes else ())
    niveis = []
    for r in razoes:
        preco = baixo + r * rng if direcao == "fundo_topo" else alto - r * rng
        if preco > 0:
            niveis.append((r, preco))
    inicio = min(i_alto, i_baixo)
    c = float(df["Close"].iloc[-1])
    ordenados = sorted(niveis, key=lambda t: t[1])
    abaixo = [t for t in ordenados if t[1] <= c]
    acima = [t for t in ordenados if t[1] > c]
    return {"direcao": direcao, "niveis": niveis, "inicio": inicio, "i_alto": i_alto, "i_baixo": i_baixo,
            "alto": alto, "baixo": baixo, "entre": (abaixo[-1] if abaixo else None, acima[0] if acima else None)}


# --------------------------------------------------------------------------- #
# Plotly
# --------------------------------------------------------------------------- #

def fmt(v: float) -> str:
    """Formata preço: 83.340 / 1,2345 / 0,0000123 (estilo legível em qualquer escala)."""
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 1:
        return f"{v:,.2f}"
    return f"{v:.4g}"


COR = {"sup": "#26a69a", "res": "#ef5350", "rompida": "#8a8f98",
       "zona": "#ffd54f", "bull": "#2e7d32", "bear": "#c62828"}


def figura_linhas(df: pd.DataFrame, res: dict, token: str = "BTC", freq: str = "1D",
                  barras: int = 365, mostrar_zonas: bool = True,
                  mostrar_toques: bool = True, altura: int = 820, fibo: dict | None = None) -> go.Figure:
    p = ParamsLinhas(**res["params"])
    df = df.dropna(subset=["Close"])
    n = len(df)
    datas = pd.DatetimeIndex(df.index)
    passo = pd.Series(datas[-30:]).diff().median() if n > 2 else pd.Timedelta(days=1)
    futuro = pd.DatetimeIndex([datas[-1] + passo * k for k in range(1, p.projecao + 1)])
    eixo_x = datas.append(futuro)

    def x(t):
        return eixo_x[min(max(int(t), 0), len(eixo_x) - 1)]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32],
                        vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=datas, open=df["Open"], high=df["High"], low=df["Low"],
                                 close=df["Close"], name=token, showlegend=False,
                                 increasing_line_color="#26a69a", decreasing_line_color="#ef5350"),
                  row=1, col=1)
    rsi = df["RSI"] if "RSI" in df.columns else rsi_wilder(df["Close"])
    fig.add_trace(go.Scatter(x=datas, y=rsi, name="RSI 14", line=dict(color="#9575cd", width=1.4),
                             showlegend=False), row=2, col=1)
    for lvl, dash in ((30, "dot"), (50, "dash"), (70, "dot")):
        fig.add_hline(y=lvl, line=dict(width=1, dash=dash, color="#555"), row=2, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="#7e57c2", opacity=0.06, line_width=0, row=2, col=1)

    legendas = set()
    for serie, row in (("preco", 1), ("rsi", 2)):
        lin = res[serie]["linhas"]
        for r in (lin.itertuples() if not lin.empty else []):
            rompida = r.estado != "ativa"
            cor = COR["rompida"] if rompida else COR[r.tipo]
            peso = r.score / 100
            grupo = f"{'Rompidas' if rompida else ('Suportes / LTA' if r.tipo == 'sup' else 'Resistências / LTB')}"
            hover = (f"<b>{r.nome}</b><br>score {r.score:.0f} · {r.n_toques} toques"
                     f"<br>{r.status}<br>agora: {fmt(r.valor_atual)}<extra></extra>")
            fim1 = r.t_brk if rompida else r.t_fim
            y_fim1 = r.y_brk if rompida else r.y_fim
            fig.add_trace(go.Scatter(
                x=[x(r.t_ini), x(fim1)], y=[r.y_ini, y_fim1], mode="lines",
                line=dict(color=cor, width=1 + 2.5 * peso, dash="dash" if rompida else "solid"),
                opacity=0.45 + 0.55 * peso, name=grupo, legendgroup=grupo,
                showlegend=grupo not in legendas, hovertemplate=hover,
            ), row=row, col=1)
            legendas.add(grupo)
            if rompida:
                # continuação depois do rompimento, já com o papel trocado:
                # LTB rompida vira suporte (verde); LTA perdida vira resistência (vermelho)
                cor2 = COR["rompida"] if r.estado == "falhou" else COR["sup" if r.tipo == "res" else "res"]
                g2 = "Continuação p/ reteste"
                fig.add_trace(go.Scatter(
                    x=[x(r.t_brk), x(r.t_fim)], y=[r.y_brk, r.y_fim], mode="lines",
                    line=dict(color=cor2, width=1 + 2 * peso, dash="dot"), opacity=0.9,
                    name=g2, legendgroup=g2, showlegend=g2 not in legendas, hovertemplate=hover,
                ), row=row, col=1)
                legendas.add(g2)
                fig.add_trace(go.Scatter(
                    x=[x(r.t_brk)], y=[r.y_brk], mode="markers", showlegend=False, legendgroup=g2,
                    hovertemplate=f"rompimento de {r.nome}<extra></extra>",
                    marker=dict(symbol="x-thin-open", size=11, color=cor2, line=dict(width=2)),
                ), row=row, col=1)
                if r.retestes_idx:
                    fig.add_trace(go.Scatter(
                        x=datas[r.retestes_idx], y=r.retestes_y, mode="markers+text", text=["R"] * len(r.retestes_idx),
                        textposition="bottom center" if r.tipo == "res" else "top center",
                        textfont=dict(size=9, color=cor2), showlegend=False, legendgroup=g2,
                        hovertemplate=f"reteste de {r.nome}<extra></extra>",
                        marker=dict(symbol="diamond", size=9, color=cor2),
                    ), row=row, col=1)
            if mostrar_toques and r.toques_idx:
                fig.add_trace(go.Scatter(
                    x=datas[r.toques_idx], y=r.toques_y, mode="markers", showlegend=False,
                    legendgroup=grupo, hoverinfo="skip",
                    marker=dict(symbol="circle-open", size=8, color=cor, line=dict(width=1.5)),
                ), row=row, col=1)

    if fibo:
        # faixas coloridas entre níveis (como no TradingView) + linha e rótulo em cada nível
        niv = sorted(fibo["niveis"], key=lambda t: t[1])
        x0, x1 = fibo["inicio"], eixo_x[-1]
        for k in range(len(niv) - 1):
            cor = CORES_FIBO[(NIVEIS_FIBO + EXT_FIBO).index(niv[k][0]) % len(CORES_FIBO)]
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=niv[k][1], y1=niv[k + 1][1], fillcolor=cor,
                          opacity=0.08, line_width=0, layer="below", row=1, col=1)
        for r, preco in fibo["niveis"]:
            cor = CORES_FIBO[(NIVEIS_FIBO + EXT_FIBO).index(r) % len(CORES_FIBO)]
            fig.add_shape(type="line", x0=x0, x1=x1, y0=preco, y1=preco, line=dict(color=cor, width=1.3),
                          row=1, col=1)
            fig.add_annotation(x=x0, y=np.log10(preco) if p.log else preco, text=f"{r:g} ({fmt(preco)})",
                               showarrow=False, xanchor="left", yanchor="bottom", xshift=4,
                               font=dict(size=10, color=cor), row=1, col=1)
        a_, b_ = ((fibo["i_baixo"], fibo["baixo"]), (fibo["i_alto"], fibo["alto"]))
        ini_, fim_ = (a_, b_) if fibo["direcao"] == "fundo_topo" else (b_, a_)
        fig.add_trace(go.Scatter(x=[ini_[0], fim_[0]], y=[ini_[1], fim_[1]], mode="lines+markers",
                                 line=dict(color="#bbbbbb", width=1, dash="dot"), marker=dict(size=6),
                                 name="Fibonacci", hoverinfo="skip"), row=1, col=1)

    if mostrar_zonas:
        for z in res["preco"]["zonas"]:
            fig.add_hrect(y0=z["y0"], y1=z["y1"], fillcolor=COR["zona"], opacity=0.10,
                          line_width=0, row=1, col=1)
            fig.add_annotation(x=eixo_x[-1], y=np.log10(z["y"]) if p.log else z["y"],
                               text=f"{fmt(z['y'])} · {z['toques']}x", showarrow=False,
                               xanchor="right", font=dict(size=10, color=COR["zona"]), row=1, col=1)

    div = res["divergencias"]
    for r in (div.itertuples() if not div.empty else []):
        if r.confirmacao > n - 1:
            continue
        cor = COR["bull"] if r.tipo == "Bull" else COR["bear"]
        fig.add_trace(go.Scatter(x=[datas[r.i0], datas[r.i1]], y=[rsi.iloc[r.i0], rsi.iloc[r.i1]],
                                 mode="lines", line=dict(color=cor, width=2), showlegend=False,
                                 hoverinfo="skip"), row=2, col=1)
        fig.add_annotation(x=datas[r.i1], y=rsi.iloc[r.i1], text=r.tipo, showarrow=False,
                           yshift=-14 if r.tipo == "Bull" else 14, bgcolor=cor,
                           font=dict(size=10, color="white"), row=2, col=1)

    ini = max(0, n - barras)
    vis = df.iloc[ini:]
    lo, hi = float(vis["Low"].min()), float(vis["High"].max())
    if fibo:                       # mostra os níveis de Fibonacci até a extensão 1,618
        vals = [v for r, v in fibo["niveis"] if r <= 1.618]
        lo, hi = min(lo, *vals), max(hi, *vals)
    if p.log:
        yr = [np.log10(lo) - 0.03, np.log10(hi) + 0.03]
        fig.update_yaxes(type="log", range=yr, row=1, col=1)
    else:
        pad = 0.05 * (hi - lo)
        fig.update_yaxes(range=[lo - pad, hi + pad], row=1, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1, title_text="RSI")
    fig.update_xaxes(range=[datas[ini], eixo_x[-1]])
    fig.update_layout(
        template="plotly_dark", height=altura, hovermode="closest",
        title=f"{token} · {freq} · linhas automáticas (preço {'log' if p.log else 'linear'})",
        xaxis_rangeslider_visible=False, margin=dict(l=50, r=30, t=70, b=30),
        legend=dict(orientation="h", y=1.01, x=1, xanchor="right", yanchor="bottom"),
    )
    return fig


def tabela_linhas(res: dict) -> pd.DataFrame:
    """Tabela amigável das linhas selecionadas (preço + RSI)."""
    partes = [res[s]["linhas"] for s in ("preco", "rsi") if not res[s]["linhas"].empty]
    if not partes:
        return pd.DataFrame()
    t = pd.concat(partes, ignore_index=True).sort_values(["status", "score"], ascending=[True, False])
    return pd.DataFrame({
        "Linha": t["nome"], "Status": t["status"], "Score": t["score"], "Toques": t["n_toques"],
        "Valor agora": t["valor_atual"], "Distância (tol)": t["dist_fech_tol"].round(2),
        "Retestes": t["retestes_idx"].map(len),
        "Inclinação/barra": t["inclinacao"].round(3),
        "Barras desde último toque": t["barras_desde_ultimo_toque"],
        "Rompida há (barras)": t["barras_desde_rompimento"],
    })
