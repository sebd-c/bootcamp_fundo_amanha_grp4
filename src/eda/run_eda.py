"""Análise exploratória de dados (EDA) para o problema de index tracking do S&P 100.

Lê as saídas do data_mining.py, grava os gráficos em <output_folder>/figures/ e um resumo em
markdown em <output_folder>/eda_summary.md. As seções seguem o "Guia de Dados & EDA" do bootcamp:

  1. Qualidade dos dados      dados faltantes, critério de remoção, outliers
  2. O índice                 estatísticas, normalidade, drawdown, retorno mensal e anual
  3. Retornos das ações       performance normalizada (base 100), estatísticas por ação, risco x retorno
  4. Correlação com o índice  ranking, beta, análise por setor
  5. Correlação entre ações   matriz por setor, top 30, pares redundantes, correlação móvel, PCA
  6. Volatilidade e crises    linha do tempo de eventos, volatilidade móvel
  7. Baselines de tracking    tracking error de carteiras simples, dentro e fora da amostra

Uso:
  python3 -m src.eda.run_eda -i data -o reports                                 # dados em data/, relatório em reports/
  python3 -m src.eda.run_eda -i data/2025 -o reports/2025                       # outra pasta de dados/relatório
  python3 -m src.eda.run_eda -i data -o reports/total -sd 2023-01-01 -rt total  # data de corte e retorno total (Adj Close)
"""
#################################################################################################
# imports
from __future__ import annotations

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path

import matplotlib

# Backend "Agg": desenha os gráficos direto em arquivo, sem abrir janelas (funciona em servidor/terminal).
# Precisa ser chamado ANTES de importar o pyplot.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # interface de gráficos
import numpy as np  # álgebra linear e estatística
import pandas as pd  # tabelas e séries temporais
from matplotlib.colors import LinearSegmentedColormap  # cria escalas de cor contínuas personalizadas
from matplotlib.ticker import PercentFormatter  # formata eixos em %
from scipy import stats  # testes estatísticos (Jarque-Bera, qui-quadrado, normal)

from src._execution_formatting import print_execution_parameters  # imprime os parâmetros usados

#################################################################################################
# constantes do módulo

INDEX_TICKER = "^OEX"  # nome da coluna do índice nas tabelas
TRADING_DAYS = 252  # número médio de pregões por ano, usado para anualizar métricas diárias

# Valores padrão dos parâmetros do terminal.
DEFAULT_TRAIN_FRACTION = 0.7  # sem data de corte, os primeiros 70% dos pregões viram treino
DEFAULT_OUTLIER_THRESHOLD = 0.15  # retorno diário acima de ±15% é marcado como extremo (valor do guia)
DEFAULT_HIGHLIGHTS = ["AAPL", "MSFT", "NVDA", "TSLA"]  # ações destacadas no gráfico de base 100 (as do guia)

# Linha do tempo dos principais eventos de mercado (seção 7.1 do guia).
EVENTS = [
    ("2018-10-01", "Guerra comercial EUA x China"),
    ("2020-03-23", "COVID-19 (mínimo do mercado)"),
    ("2020-11-09", "Anúncio das vacinas"),
    ("2022-01-03", "Início da alta de juros (Fed)"),
    ("2022-11-11", "Crise FTX (cripto)"),
    ("2023-10-09", "Conflito Israel-Hamas"),
    ("2024-01-02", "Rally de IA (NVDA, MSFT)"),
]

# Paleta de cores validada para daltonismo e contraste (modo claro).
SURFACE = "#fcfcfb"  # fundo dos gráficos
TEXT = "#0b0b0b"  # texto principal (títulos)
TEXT_2 = "#52514e"  # texto secundário (eixos, rótulos)
MUTED = "#c3c2b7"  # linhas de fundo (ações individuais no gráfico de base 100)
GRID = "#e4e3df"  # linhas de grade, bem discretas
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"  # séries, nesta ordem
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]  # ordem fixa das cores categóricas
NAVY = "#0d366b"  # destaque do índice
RED = "#e34948"  # perdas (drawdown, meses negativos)
# Escala divergente azul -> cinza -> vermelho para correlações (-1 a +1), com cinza neutro no zero.
DIVERGING = LinearSegmentedColormap.from_list("div", ["#104281", "#6da7ec", "#f0efec", "#ec8a89", "#a82626"])

# Estilo padrão aplicado a todos os gráficos.
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,  # fundos
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,  # cores de eixos e títulos
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",  # título em negrito, à esquerda
    "axes.spines.top": False, "axes.spines.right": False,  # remove as bordas de cima e da direita
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,  # grade fina e clara
    "xtick.color": TEXT_2, "ytick.color": TEXT_2, "text.color": TEXT,  # cores dos números nos eixos
    "lines.linewidth": 1.6, "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 150,  # linhas, legenda, resolução
})

# Lista de trechos de markdown que, no final, é juntada no arquivo eda_summary.md.
report: list[str] = []

#################################################################################################
# funções auxiliares de relatório e métricas


def section(title: str) -> None:
    """Adiciona um título de seção (## ...) ao relatório."""
    report.append(f"\n## {title}\n")


def note(text: str) -> None:
    """Adiciona um parágrafo de texto ao relatório."""
    report.append(text + "\n")


def table(df: pd.DataFrame | pd.Series, floatfmt: str = ".4f") -> None:
    """Adiciona uma tabela ao relatório (to_markdown usa o pacote tabulate)."""
    report.append(df.to_markdown(floatfmt=floatfmt) + "\n")


def save(fig: plt.Figure, name: str, fig_dir: Path) -> None:
    """Salva a figura em PNG, libera a memória e insere a imagem no relatório."""
    fig.tight_layout()  # ajusta os espaçamentos para nada ficar cortado
    fig.savefig(fig_dir / f"{name}.png", bbox_inches="tight")  # grava o arquivo
    plt.close(fig)  # fecha a figura (sem isso, a memória cresce a cada gráfico)
    report.append(f"![{name}](figures/{name}.png)\n")  # link relativo da imagem no markdown


def mark_events(ax: plt.Axes, index: pd.DatetimeIndex) -> None:
    """Desenha linhas verticais pontilhadas nos eventos que caem dentro do período do gráfico."""
    for i, (date, _) in enumerate(EVENTS, start=1):
        d = pd.Timestamp(date)
        # só marca eventos dentro do intervalo de datas dos dados
        if index.min() <= d <= index.max():
            ax.axvline(d, color=TEXT_2, linestyle=":", linewidth=1)
            # número do evento no topo do gráfico (a legenda completa fica na tabela da seção 6)
            ax.annotate(str(i), (d, 1), xycoords=("data", "axes fraction"), xytext=(2, -10),
                        textcoords="offset points", fontsize=8, color=TEXT_2)


def annualized_stats(r: pd.Series) -> pd.Series:
    """Estatísticas descritivas de uma série de retornos diários."""
    return pd.Series({
        # Retorno anual composto (CAGR): produto de (1 + r) elevado a (252 / nº de dias), menos 1.
        "retorno_anual_composto": (1 + r).prod() ** (TRADING_DAYS / len(r)) - 1,
        # Volatilidade anual: desvio-padrão diário x raiz de 252 (a variância cresce linearmente com o tempo).
        "volatilidade_anual": r.std() * np.sqrt(TRADING_DAYS),
        # Índice de Sharpe com taxa livre de risco = 0: retorno médio / risco, anualizado.
        "sharpe_rf0": r.mean() / r.std() * np.sqrt(TRADING_DAYS),
        # Assimetria: negativa = quedas grandes são mais frequentes que altas grandes.
        "assimetria": r.skew(),
        # Curtose em excesso: > 0 = caudas mais pesadas que a normal (eventos extremos mais comuns).
        "curtose_excesso": r.kurt(),
        "pior_dia": r.min(),  # pior dia
        "melhor_dia": r.max(),  # melhor dia
    })


def tracking_error(portfolio: pd.Series, index: pd.Series) -> float:
    """Tracking error anualizado: desvio-padrão da diferença diária de retorno (carteira - índice) x raiz de 252."""
    return (portfolio - index).std() * np.sqrt(TRADING_DAYS)


def load(input_folder: Path, return_type: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega os arquivos gerados pelo data_mining.py."""
    raw_dir = input_folder / "raw"  # dados brutos
    processed_dir = input_folder / "processed"  # retornos limpos
    prices = pd.read_parquet(raw_dir / "prices.parquet")  # preços (data x ticker), com ^OEX
    volume = pd.read_parquet(raw_dir / "volume.parquet")  # volumes (data x ticker)
    # "price" = só preço (mesma base do ^OEX); "total" = com dividendos (Adj Close, como no guia)
    returns_file = "returns.parquet" if return_type == "price" else "returns_total.parquet"
    returns = pd.read_parquet(processed_dir / returns_file)
    # Composição do índice, indexada pelo ticker do Yahoo para casar com as colunas das outras tabelas.
    constituents = pd.read_csv(raw_dir / "constituents.csv").set_index("yahoo_ticker")
    return prices, volume, returns, constituents

#################################################################################################
# 1. Qualidade dos dados


def data_quality(prices: pd.DataFrame, volume: pd.DataFrame, returns: pd.DataFrame,
                 outlier_threshold: float, fig_dir: Path) -> None:
    section("1. Qualidade dos dados")
    stocks = prices.drop(columns=INDEX_TICKER)  # só as ações, sem o índice
    # Considera só os dias de pregão do índice (o calendário usado nos retornos).
    stocks = stocks[prices[INDEX_TICKER].notna()]
    # Percentual de dias sem preço para cada ação.
    pct_missing = stocks.isna().mean().sort_values(ascending=False)
    # Primeira data com preço válido de cada ação (mostra quem abriu capital depois do início).
    first_valid = stocks.apply(lambda s: s.first_valid_index())
    # Ações que sobreviveram à limpeza (as colunas do painel de retornos, menos o índice).
    kept = [c for c in returns.columns if c != INDEX_TICKER]
    dropped = sorted(set(stocks.columns) - set(kept))  # ações removidas na limpeza

    note(
        f"- Painel bruto: **{len(stocks)} pregões x {stocks.shape[1]} ações** "
        f"({stocks.index.min().date()} a {stocks.index.max().date()}).\n"
        f"- Ações sem nenhum dado faltante: **{(pct_missing == 0).sum()}**; com algum dado faltante: "
        f"**{(pct_missing > 0).sum()}**.\n"
        f"- Painel limpo de retornos: **{returns.shape[0]} pregões x {len(kept)} ações**. "
        f"Removidas por histórico insuficiente: {', '.join(dropped) if dropped else 'nenhuma'}.\n"
        "- Critério: remover ações com menos de 95% de cobertura (parâmetro `-mc` do data_mining) e "
        "preencher buracos de até 5 pregões com o último preço (forward fill). O critério é mais "
        "rígido que os 20% do guia de propósito: o painel precisa estar completo (sem NaN) para os "
        "modelos, então uma ação com 15% de dados faltantes cortaria 15% do histórico de todas as outras.\n"
        f"- Dias sem dado do índice (^OEX): {prices[INDEX_TICKER].isna().sum()}.\n"
        "- **Viés de sobrevivência:** o universo é a composição *de hoje* aplicada a todo o histórico. "
        "Ações que saíram do índice no período não aparecem, e ações que entraram depois aparecem antes "
        "de serem membros. Os backtests tendem a ser otimistas."
    )

    # Tabela com as ações que têm algum dado faltante, a data do primeiro preço e o motivo provável.
    missing = pct_missing[pct_missing > 0].to_frame("pct_faltante")
    missing["primeiro_preco"] = first_valid.reindex(missing.index).dt.date
    missing["situacao"] = np.where(missing.index.isin(dropped), "removida", "mantida (ffill)")
    if len(missing):
        note("Ações com dados faltantes (em geral, IPO ou spin-off depois do início do período):")
        table(missing, floatfmt=".3f")

    # Gráfico: completude de cada ação com dado faltante, com a linha do critério de 95%.
    if len(missing):
        fig, ax = plt.subplots(figsize=(8, max(2.5, 0.3 * len(missing) + 1)))
        completeness = (1 - missing["pct_faltante"]).sort_values()  # fração de dados presentes
        ax.barh(completeness.index, completeness.values, color=BLUE, height=0.6)
        ax.axvline(0.95, color=TEXT_2, linestyle="--", linewidth=1)  # linha do critério
        ax.annotate("critério: 95%", (0.95, 1), xycoords=("data", "axes fraction"), xytext=(4, -12),
                    textcoords="offset points", fontsize=8, color=TEXT_2)
        ax.set_title("Completude dos dados por ação (só ações com dados faltantes)")
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlim(0, 1)
        ax.grid(axis="y", visible=False)
        save(fig, "01_completude", fig_dir)

    # Conta dias com volume zero por ação: um possível sinal de dado ruim (preço "parado").
    zero_vol = (volume.drop(columns=INDEX_TICKER, errors="ignore") == 0).sum()
    zero_vol = zero_vol[zero_vol > 0]  # mantém só quem teve pelo menos um dia assim
    note(f"Ações com dias de volume zero: {zero_vol.to_dict() if len(zero_vol) else 'nenhuma'}.")

    # stack() transforma a tabela larga (data x ticker) em uma lista longa de pares (data, ticker) -> retorno.
    extreme = returns[kept].stack()
    # Filtra os movimentos acima do limite em módulo e transforma o índice (data, ticker) em colunas.
    extreme = extreme[extreme.abs() > outlier_threshold].rename("retorno").reset_index()
    extreme.columns = ["data", "ticker", "retorno"]  # nomes das colunas
    extreme["data"] = extreme["data"].dt.date  # tira o horário, para a tabela ficar legível
    note(f"**Outliers:** {len(extreme)} retornos diários acima de ±{outlier_threshold:.0%} "
         f"em {extreme['ticker'].nunique()} ações. As verificações pontuais batem com eventos reais "
         "(resultados trimestrais, março de 2020), então são mantidos. Mas eles pesam muito em perdas "
         "quadráticas; vale considerar funções de perda robustas no modelo.")
    # Número de outliers por ação (as 15 com mais ocorrências).
    table(extreme["ticker"].value_counts().head(15).rename("n_outliers").to_frame())
    note("Os 10 maiores movimentos em módulo:")
    top_moves = extreme.reindex(extreme["retorno"].abs().sort_values(ascending=False).index).head(10)
    table(top_moves.set_index(["data", "ticker"]), floatfmt=".3f")

#################################################################################################
# 2. O índice


def index_analysis(prices: pd.DataFrame, returns: pd.DataFrame, fig_dir: Path) -> None:
    section("2. O índice (^OEX, índice de preço)")
    idx = prices[INDEX_TICKER].dropna()  # nível (pontos) do índice
    r = returns[INDEX_TICKER]  # retornos diários do índice
    table(annualized_stats(r).to_frame("^OEX"))  # tabela de estatísticas descritivas

    # Jarque-Bera: testa se os retornos seguem uma distribuição normal (H0 = normal).
    jb = stats.jarque_bera(r)
    # Ljung-Box nos retornos: existe autocorrelação (os retornos passados preveem os futuros)?
    lb_r = _ljung_box(r, 10)
    # Ljung-Box nos retornos AO QUADRADO: a volatilidade de hoje depende da de ontem (agrupamento)?
    lb_r2 = _ljung_box(r**2, 10)
    note(
        f"- Jarque-Bera p-valor = {jb.pvalue:.2e}: os retornos **não são normais** (caudas pesadas, "
        f"curtose em excesso de {r.kurt():.1f}; a normal tem 0).\n"
        # autocorr(1) = correlação entre o retorno de hoje e o de ontem; recalculada sem 2020 para comparar.
        f"- Ljung-Box (10 defasagens) nos retornos: p = {lb_r:.3g} (autocorrelação de 1 dia "
        f"{r.autocorr(1):.3f}; {r[r.index.year != 2020].autocorr(1):.3f} sem 2020). A dependência "
        "serial é fraca e concentrada nas crises.\n"
        f"- Nos retornos ao quadrado: p = {lb_r2:.3g}, **forte agrupamento de volatilidade**. "
        "Pesos estimados em períodos calmos podem se comportar mal em períodos de estresse."
    )

    # Drawdown: queda percentual em relação ao maior valor já atingido até cada data.
    drawdown = idx / idx.cummax() - 1
    # Duas figuras empilhadas com o mesmo eixo x: nível em cima (2/3 da altura) e drawdown embaixo (1/3).
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, height_ratios=[2, 1])
    a1.plot(idx.index, idx.values, color=BLUE)
    a1.set_title("S&P 100 (^OEX): nível do índice")
    # Área preenchida entre o drawdown e zero, mais a linha por cima.
    a2.fill_between(drawdown.index, drawdown.values, 0, color=RED, alpha=0.35, linewidth=0)
    a2.plot(drawdown.index, drawdown.values, color=RED, linewidth=1)
    a2.set_title("Drawdown (queda desde o pico anterior)")
    a2.yaxis.set_major_formatter(PercentFormatter(1))  # eixo em %, onde 1.0 = 100%
    save(fig, "02_indice_nivel_drawdown", fig_dir)
    note(f"Drawdown máximo: **{drawdown.min():.1%}** em {drawdown.idxmin().date()}.")

    # Histograma dos retornos com a curva normal de mesma média e desvio (evidencia as caudas pesadas).
    fig, ax = plt.subplots(figsize=(9, 3.6))
    # Histograma normalizado (density=True) para ser comparável à curva de densidade da normal.
    ax.hist(r, bins=120, density=True, color=BLUE, alpha=0.8, edgecolor=SURFACE, linewidth=0.3,
            label="distribuição real")
    x = np.linspace(r.min(), r.max(), 400)  # 400 pontos entre o menor e o maior retorno
    ax.plot(x, stats.norm.pdf(x, r.mean(), r.std()), color=ORANGE, label="normal teórica")
    ax.set_title("Distribuição dos retornos diários do S&P 100")
    ax.set_xlabel("retorno diário")
    ax.legend()
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    save(fig, "03_distribuicao_retornos", fig_dir)

    # Retorno mensal composto: produto de (1 + r) dentro de cada mês, menos 1 ("ME" = fim do mês).
    monthly = (1 + r).resample("ME").prod() - 1
    fig, ax = plt.subplots(figsize=(11, 3.8))
    # Azul para meses positivos e vermelho para negativos (os dois polos da escala divergente).
    ax.bar(monthly.index, monthly.values, width=20, color=np.where(monthly.values >= 0, BLUE, RED))
    ax.axhline(0, color=TEXT_2, linewidth=0.8)  # linha do zero
    worst = monthly.idxmin()  # pior mês
    ax.annotate(f"pior mês: {monthly.min():.1%}", (worst, monthly.min()), xytext=(10, -2),
                textcoords="offset points", fontsize=8, color=TEXT_2, va="center")
    ax.set_title("Retorno mensal do S&P 100 (azul = alta, vermelho = queda)")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="x", visible=False)
    save(fig, "04_retorno_mensal", fig_dir)

    # Retorno acumulado em cada ano ("YE" = fim do ano); o primeiro e o último ano podem estar incompletos.
    annual = (1 + r).resample("YE").prod() - 1
    annual.index = annual.index.year  # mostra só o ano
    days = r.groupby(r.index.year).size()  # pregões de cada ano na amostra
    note(f"Pior mês: **{monthly.min():.1%}** ({worst:%m/%Y}); melhor mês: **{monthly.max():.1%}** "
         f"({monthly.idxmax():%m/%Y}). Retorno por ano (anos com menos de 250 pregões estão incompletos):")
    table(pd.DataFrame({"retorno": annual.map(lambda v: f"{v:.1%}"), "pregoes": days}))


def _ljung_box(x: pd.Series, lags: int) -> float:
    """p-valor do teste Q de Ljung-Box (implementado aqui para não depender do statsmodels)."""
    x = x - x.mean()  # centraliza a série na média
    n = len(x)  # número de observações
    denom = (x**2).sum()  # soma dos quadrados (denominador da autocorrelação)
    # Autocorrelação para cada defasagem k = 1..lags: soma de x_t * x_{t-k}, dividida pelo denominador.
    acf = np.array([(x[k:].values * x[:-k].values).sum() / denom for k in range(1, lags + 1)])
    # Estatística Q = n(n+2) * soma(acf_k^2 / (n-k)); sob H0 (sem autocorrelação) segue qui-quadrado(lags).
    q = n * (n + 2) * np.sum(acf**2 / (n - np.arange(1, lags + 1)))
    # sf = 1 - CDF: probabilidade de um Q tão grande quanto o observado se H0 fosse verdadeira.
    return float(stats.chi2.sf(q, lags))

#################################################################################################
# 3. Retornos das ações


def stock_returns_analysis(returns: pd.DataFrame, highlights: list[str], fig_dir: Path) -> pd.DataFrame:
    section("3. Retornos das ações")
    stocks = returns.drop(columns=INDEX_TICKER)
    # Performance normalizada: cada série começa em 100 e cresce com seus retornos acumulados.
    norm = (1 + returns).cumprod() * 100
    # Acrescenta o ponto inicial (100) no dia anterior ao primeiro retorno.
    norm.loc[returns.index[0] - pd.Timedelta(days=1)] = 100
    norm = norm.sort_index()

    fig, ax = plt.subplots(figsize=(11, 6))
    # Todas as ações em cinza claro, ao fundo (mostram a dispersão do universo).
    ax.plot(norm.index, norm[stocks.columns].values, color=MUTED, linewidth=0.5, alpha=0.6)
    # Ações destacadas, com as cores categóricas na ordem fixa (só as que existem nos dados).
    shown = [t for t in highlights if t in norm.columns][:len(SERIES) - 1]
    for t, color in zip(shown, SERIES):
        ax.plot(norm.index, norm[t], color=color, linewidth=1.5, label=t)
    # O índice por cima de tudo, mais grosso.
    ax.plot(norm.index, norm[INDEX_TICKER], color=NAVY, linewidth=2.5, label="S&P 100 (índice)")
    mark_events(ax, norm.index)  # linhas dos eventos numerados
    ax.set_yscale("log")  # escala log: a mesma distância vertical = a mesma variação percentual
    ax.set_title(f"Performance normalizada (base 100 em {norm.index[0]:%m/%Y}, escala log)")
    ax.set_ylabel("valor (base 100)")
    ax.legend(loc="upper left")
    save(fig, "05_performance_normalizada", fig_dir)
    note("Escala logarítmica: sem ela, ações que multiplicaram de valor (como a NVDA) achatariam todas "
         "as outras. Os números no topo são os eventos da tabela da seção 6.")

    # Estatísticas descritivas por ação (seção 5.2 do guia).
    stock_stats = pd.DataFrame({
        "retorno_medio_diario": stocks.mean(),
        "retorno_medio_anual": stocks.mean() * TRADING_DAYS,  # média aritmética anualizada, como no guia
        "volatilidade_anual": stocks.std() * np.sqrt(TRADING_DAYS),
        "sharpe_rf0": stocks.mean() / stocks.std() * np.sqrt(TRADING_DAYS),
        "pior_dia": stocks.min(),
        "melhor_dia": stocks.max(),
        "assimetria": stocks.skew(),
        "curtose_excesso": stocks.kurt(),
        "pregoes": stocks.count(),
    })
    note("Maior retorno médio anualizado:")
    table(stock_stats.nlargest(10, "retorno_medio_anual").iloc[:, 1:4], floatfmt=".3f")
    note("Menor volatilidade (mais estáveis):")
    table(stock_stats.nsmallest(10, "volatilidade_anual").iloc[:, 1:4], floatfmt=".3f")
    note("Melhor Sharpe (retorno ajustado ao risco):")
    table(stock_stats.nlargest(10, "sharpe_rf0").iloc[:, 1:4], floatfmt=".3f")

    # Gráfico risco x retorno: cada ponto é uma ação; o índice aparece em destaque.
    idx = returns[INDEX_TICKER]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(stock_stats["volatilidade_anual"], stock_stats["retorno_medio_anual"], s=40, color=BLUE,
               alpha=0.6, edgecolor=SURFACE, linewidth=1, label="ações")
    ax.scatter(idx.std() * np.sqrt(TRADING_DAYS), idx.mean() * TRADING_DAYS, s=120, color=NAVY, marker="D",
               edgecolor=SURFACE, linewidth=1.5, label="S&P 100 (índice)")
    # Rótulos das 5 ações com melhor Sharpe.
    for t in stock_stats.nlargest(5, "sharpe_rf0").index:
        ax.annotate(t, (stock_stats.at[t, "volatilidade_anual"], stock_stats.at[t, "retorno_medio_anual"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8, color=TEXT_2)
    ax.axhline(0, color=TEXT_2, linestyle="--", linewidth=0.8)  # linha do retorno zero
    ax.set_title("Risco x retorno das ações do S&P 100")
    ax.set_xlabel("volatilidade anualizada")
    ax.set_ylabel("retorno médio anualizado")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="upper left")
    save(fig, "06_risco_retorno", fig_dir)
    note(f"O índice tem volatilidade de {idx.std() * np.sqrt(TRADING_DAYS):.1%}, contra a mediana de "
         f"{stock_stats['volatilidade_anual'].median():.1%} das ações: a diversificação elimina boa parte do "
         "risco específico de cada empresa. A carteira de tracking precisa de ações suficientes para isso.")
    return stock_stats

#################################################################################################
# 4. Correlação com o índice


def index_correlation_analysis(returns: pd.DataFrame, constituents: pd.DataFrame, fig_dir: Path) -> pd.DataFrame:
    section("4. Correlação com o índice (a análise central do projeto)")
    idx = returns[INDEX_TICKER]  # retornos do índice
    stocks = returns.drop(columns=INDEX_TICKER)  # retornos das ações
    # Beta de cada ação: cov(ação, índice) / var(índice) = sensibilidade da ação aos movimentos do índice.
    beta = stocks.apply(lambda s: np.cov(s, idx)[0, 1] / idx.var())
    # Tabela com uma linha por ação e suas métricas em relação ao índice.
    per_stock = pd.DataFrame({
        "setor": constituents["sector"].reindex(stocks.columns),  # setor GICS
        "peso_aprox": constituents["approx_weight"].reindex(stocks.columns),  # peso aproximado no índice
        "volatilidade_anual": stocks.std() * np.sqrt(TRADING_DAYS),  # volatilidade anualizada
        "corr_indice": stocks.corrwith(idx),  # correlação de Pearson com o índice
        "beta": beta,
    }).sort_values("corr_indice", ascending=False)

    pct_07 = (per_stock["corr_indice"] > 0.7).mean()  # fração de ações com correlação > 0,7
    note(
        f"- Correlação com o índice: mediana **{per_stock['corr_indice'].median():.2f}**, de "
        f"{per_stock['corr_indice'].min():.2f} a {per_stock['corr_indice'].max():.2f}. "
        f"**{pct_07:.0%}** das ações têm correlação > 0,7 e {(per_stock['corr_indice'] > 0.5).mean():.0%} "
        "têm > 0,5.\n"
        "- Nenhuma ação sozinha replica o índice; é a combinação delas que reduz o risco específico.\n"
        f"- Concentração (pesos aproximados pelo valor de mercado na última data dos dados): as 10 maiores somam "
        f"**{constituents['approx_weight'].nlargest(10).sum():.1%}** do índice."
    )
    note("As 15 ações mais correlacionadas com o índice (melhores candidatas para o tracking):")
    table(per_stock.head(15), floatfmt=".3f")
    note("As 10 menos correlacionadas:")
    table(per_stock.tail(10), floatfmt=".3f")

    # Gráfico de barras ranqueado, com as linhas de referência 0,7 e 0,5 do guia.
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.bar(range(len(per_stock)), per_stock["corr_indice"], color=BLUE, width=0.75)
    for level in (0.7, 0.5):
        ax.axhline(level, color=TEXT_2, linestyle="--", linewidth=1)
        ax.annotate(f"{level:.1f}", (1, level), xycoords=("axes fraction", "data"), xytext=(4, 0),
                    textcoords="offset points", fontsize=8, color=TEXT_2, va="center")
    ax.set_xticks(range(len(per_stock)), per_stock.index, rotation=90, fontsize=6)
    ax.set_xlim(-0.8, len(per_stock) - 0.2)
    ax.set_title("Correlação de cada ação com o S&P 100")
    ax.set_ylabel("correlação de Pearson")
    ax.grid(axis="x", visible=False)
    save(fig, "07_correlacao_com_indice", fig_dir)

    # Correlação com o índice por setor: média, desvio e número de ações.
    by_sector = per_stock.groupby("setor")["corr_indice"].agg(["mean", "std", "count"])
    by_sector.columns = ["corr_media", "desvio", "n_acoes"]
    by_sector = by_sector.sort_values("corr_media", ascending=False)
    note("Correlação com o índice por setor (setores com correlação alta são mais fáceis de replicar):")
    table(by_sector, floatfmt=".3f")

    # Boxplot horizontal da correlação por setor, ordenado pela mediana.
    order = per_stock.groupby("setor")["corr_indice"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.boxplot([per_stock.loc[per_stock["setor"] == s, "corr_indice"] for s in order], vert=False,
               tick_labels=order, widths=0.6, patch_artist=True,
               boxprops={"facecolor": "#cde2fb", "edgecolor": BLUE}, medianprops={"color": NAVY, "linewidth": 2},
               whiskerprops={"color": BLUE}, capprops={"color": BLUE},
               flierprops={"markeredgecolor": BLUE, "markersize": 4})
    ax.set_title("Correlação com o S&P 100 por setor")
    ax.set_xlabel("correlação com o índice")
    ax.grid(axis="y", visible=False)
    save(fig, "08_correlacao_por_setor", fig_dir)

    # Peso total e número de ações por setor.
    sector_w = constituents.groupby("sector")["approx_weight"].sum().sort_values()
    sector_n = constituents.groupby("sector").size().reindex(sector_w.index)  # mesma ordem dos pesos
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.barh(sector_w.index, sector_w.values, color=BLUE, height=0.7)
    # Escreve "peso% (n ações)" logo à direita de cada barra.
    for bar, w, n in zip(bars, sector_w.values, sector_n.values):
        ax.text(bar.get_width() + 0.004, bar.get_y() + bar.get_height() / 2, f"{w:.1%} ({n} ações)",
                va="center", fontsize=8, color=TEXT_2)
    ax.set_title("Peso aproximado de cada setor no índice")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", visible=False)  # remove a grade horizontal (desnecessária em barras horizontais)
    ax.set_xlim(0, sector_w.max() * 1.3)  # 30% de folga à direita para caber o texto
    save(fig, "09_peso_por_setor", fig_dir)

    # Histograma dos betas, com uma linha tracejada em beta = 1 (move-se junto com o índice).
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.hist(per_stock["beta"], bins=25, color=BLUE, edgecolor=SURFACE, linewidth=1)
    ax.axvline(1, color=TEXT_2, linestyle="--", linewidth=1)
    ax.set_title("Beta das ações em relação ao índice")
    ax.set_xlabel("beta (1 = acompanha o índice na mesma intensidade)")
    save(fig, "10_beta", fig_dir)
    return per_stock  # devolvida porque as seções 5 e 7 também usam essa tabela

#################################################################################################
# 5. Correlação entre ações


def stock_correlation_analysis(returns: pd.DataFrame, per_stock: pd.DataFrame, fig_dir: Path) -> None:
    section("5. Correlação entre ações")
    stocks = returns.drop(columns=INDEX_TICKER)
    corr = stocks.corr()  # matriz de correlação ação x ação

    # Mapa de calor completo, com as ações agrupadas por setor (mostra os blocos setoriais).
    order = per_stock.sort_values(["setor", "corr_indice"]).index
    corr_sorted = corr.loc[order, order]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr_sorted.values, cmap=DIVERGING, vmin=-1, vmax=1, interpolation="nearest")
    sectors = per_stock.loc[order, "setor"].values  # setor de cada linha/coluna, na ordem do gráfico
    # Posições onde o setor muda: ali desenhamos linhas separando os blocos.
    bounds = [i for i in range(1, len(sectors)) if sectors[i] != sectors[i - 1]]
    for b in bounds:
        ax.axhline(b - 0.5, color=SURFACE, linewidth=1.5)  # separador horizontal
        ax.axvline(b - 0.5, color=SURFACE, linewidth=1.5)  # separador vertical
    # Posição do meio de cada bloco, onde fica o nome do setor no eixo y.
    mids = [(a + b) / 2 for a, b in zip([0] + bounds, bounds + [len(sectors)])]
    ax.set_yticks(mids, [sectors[int(m)] for m in mids], fontsize=7)
    ax.set_xticks([])  # sem rótulos no eixo x (repetiriam os do eixo y)
    ax.grid(False)
    ax.set_title("Correlação entre as ações, agrupadas por setor")
    fig.colorbar(im, ax=ax, shrink=0.7, label="correlação")  # barra de cores com a escala
    save(fig, "11_matriz_correlacao_setores", fig_dir)

    # Heatmap das 30 ações mais correlacionadas com o índice, com os valores escritos (seção 6.1 do guia).
    top30 = per_stock.index[:30]
    c30 = corr.loc[top30, top30].values
    # Máscara do triângulo superior (inclui a diagonal): mostra cada par uma única vez.
    masked = np.ma.masked_where(np.triu(np.ones_like(c30, dtype=bool)), c30)
    fig, ax = plt.subplots(figsize=(12, 10.5))
    im = ax.imshow(masked, cmap=DIVERGING, vmin=-1, vmax=1)
    for i in range(len(top30)):
        for j in range(i):  # só abaixo da diagonal
            # texto branco nas células escuras e preto nas claras, para manter a leitura
            ax.text(j, i, f"{c30[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                    color=SURFACE if abs(c30[i, j]) > 0.6 else TEXT)
    ax.set_xticks(range(len(top30)), top30, rotation=90, fontsize=7)
    ax.set_yticks(range(len(top30)), top30, fontsize=7)
    ax.grid(False)
    ax.spines[:].set_visible(False)
    ax.set_title("Correlação entre as 30 ações mais correlacionadas com o S&P 100")
    fig.colorbar(im, ax=ax, shrink=0.7, label="correlação")
    save(fig, "12_heatmap_top30", fig_dir)

    # Triângulo superior da matriz, sem a diagonal (k=1): cada par de ações aparece uma única vez.
    iu = np.triu_indices_from(corr, k=1)
    pairs = pd.DataFrame({"acao_1": corr.index[iu[0]], "acao_2": corr.columns[iu[1]],
                          "correlacao": corr.values[iu]}).sort_values("correlacao", ascending=False)
    n90 = (pairs["correlacao"] > 0.9).sum()  # pares potencialmente redundantes
    note(f"- Correlação média entre pares: **{pairs['correlacao'].mean():.2f}** "
         f"(mediana {pairs['correlacao'].median():.2f}). Há blocos setoriais claros: exigir equilíbrio "
         "entre setores é uma restrição natural para a carteira de tracking.\n"
         f"- Pares com correlação > 0,90 (redundantes, colocar os dois não diversifica): **{n90}**. "
         "GOOG/GOOGL são duas classes de ação da mesma empresa (Alphabet). Os 15 pares mais correlacionados:")
    table(pairs.head(15).set_index(["acao_1", "acao_2"]), floatfmt=".3f")

    # Correlação média entre pares ao longo do tempo, em janelas de 63 pregões (~1 trimestre).
    window = 63
    avg_corr = _rolling_avg_corr(stocks, window)
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(avg_corr.index, avg_corr.values, color=BLUE)
    mark_events(ax, avg_corr.index)
    ax.set_title(f"Correlação média entre as ações, janela móvel de {window} pregões")
    save(fig, "13_correlacao_media_movel", fig_dir)
    note(f"- A correlação média móvel vai de {avg_corr.min():.2f} a {avg_corr.max():.2f}, com pico em "
         f"{avg_corr.idxmax().date()}: nas crises, tudo cai junto. A correlação **não é estacionária**, "
         "então os pesos devem ser reestimados periodicamente (janelas móveis / rebalanceamento).")

    # PCA feita "à mão": padroniza cada ação (média 0, desvio 1) para que as mais voláteis não dominem.
    standardized = (stocks - stocks.mean()) / stocks.std()
    # Autovalores/autovetores da matriz de covariância; eigh serve para matrizes simétricas.
    eigvals, eigvecs = np.linalg.eigh(np.cov(standardized.T))
    # eigh devolve em ordem crescente; invertemos para o 1º componente ser o de maior variância.
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]
    explained = eigvals / eigvals.sum()  # fração da variância total explicada por cada componente
    pc1 = standardized.values @ eigvecs[:, 0]  # série temporal do 1º componente (o "fator mercado")
    # Correlação do PC1 com o índice, em módulo (o sinal de um autovetor é arbitrário).
    pc1_corr = abs(np.corrcoef(pc1, returns[INDEX_TICKER])[0, 1])
    # Número de componentes necessários para explicar 90% da variância.
    n90_pc = int(np.searchsorted(np.cumsum(explained), 0.9) + 1)

    # Gráfico: barras = variância de cada componente; linha = variância acumulada (20 primeiros).
    fig, ax = plt.subplots(figsize=(9, 3.4))
    k = 20
    ax.bar(range(1, k + 1), explained[:k], color=BLUE, width=0.75, label="por componente")
    ax.plot(range(1, k + 1), np.cumsum(explained[:k]), color=ORANGE, marker="o", markersize=4, label="acumulada")
    ax.set_title("PCA dos retornos padronizados: variância explicada")
    ax.set_xlabel("componente principal")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend()
    save(fig, "14_pca_variancia_explicada", fig_dir)
    note(f"- O 1º componente (o fator mercado) explica **{explained[0]:.1%}** da variância e tem correlação "
         f"{pc1_corr:.3f} com o índice. São necessários {n90_pc} componentes para 90%: um fator dominante e "
         "uma cauda longa de fatores setoriais e específicos.")


def _rolling_avg_corr(stocks: pd.DataFrame, window: int, step: int = 5) -> pd.Series:
    """Correlação média entre todos os pares de ações, em janelas móveis (avançando `step` dias por vez)."""
    values, dates = [], []
    arr = stocks.values  # NumPy puro é bem mais rápido que o pandas neste laço
    n = arr.shape[1]  # número de ações
    # Cada iteração olha os `window` dias que terminam em `end`; pular de 5 em 5 dias deixa o cálculo ~5x mais rápido.
    for end in range(window, len(arr) + 1, step):
        c = np.corrcoef(arr[end - window:end].T)  # matriz de correlação da janela
        # Média fora da diagonal: soma total menos os n "1"s da diagonal, dividida pelo nº de pares n(n-1).
        values.append((c.sum() - n) / (n * (n - 1)))
        dates.append(stocks.index[end - 1])  # data do último dia da janela
    return pd.Series(values, index=dates)

#################################################################################################
# 6. Volatilidade e períodos de crise


def volatility_analysis(returns: pd.DataFrame, fig_dir: Path) -> None:
    section("6. Volatilidade e períodos de crise")
    r = returns[INDEX_TICKER]
    # Tabela de eventos: só os que caem dentro do período dos dados.
    in_range = [(i, d, e) for i, (d, e) in enumerate(EVENTS, start=1)
                if r.index.min() <= pd.Timestamp(d) <= r.index.max()]
    note("Linha do tempo dos principais eventos (numerados nos gráficos):")
    table(pd.DataFrame(in_range, columns=["n", "data", "evento"]).set_index("n"))

    # Volatilidade em janela móvel de 21 pregões (~1 mês), anualizada.
    vol = r.rolling(21).std() * np.sqrt(TRADING_DAYS)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(vol.index, vol.values, 0, color=BLUE, alpha=0.25, linewidth=0)  # área sob a curva
    ax.plot(vol.index, vol.values, color=BLUE, linewidth=1.2)
    ax.axhline(vol.mean(), color=TEXT_2, linestyle="--", linewidth=1)  # média histórica
    ax.annotate(f"média: {vol.mean():.1%}", (0, vol.mean()), xycoords=("axes fraction", "data"),
                xytext=(4, 4), textcoords="offset points", fontsize=8, color=TEXT_2)
    mark_events(ax, vol.index)
    ax.set_title("Volatilidade móvel do S&P 100 (21 pregões, anualizada)")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylim(bottom=0)
    save(fig, "15_volatilidade_movel", fig_dir)
    note(f"- Volatilidade média: **{vol.mean():.1%}** ao ano; máxima: **{vol.max():.1%}** em "
         f"{vol.idxmax().date()}.\n"
         "- Em crises, a volatilidade e a correlação entre as ações sobem juntas (ver seção 5). Isso "
         "torna o índice *mais fácil* de replicar nesses dias (tudo anda junto), mas os erros de "
         "tracking em valor absoluto ficam maiores.")

#################################################################################################
# 7. Baselines de tracking


def tracking_baselines(returns: pd.DataFrame, per_stock: pd.DataFrame, split_date: pd.Timestamp,
                       return_type: str, fig_dir: Path) -> pd.DataFrame:
    section("7. Baselines de tracking")
    idx = returns[INDEX_TICKER]
    stocks = returns.drop(columns=INDEX_TICKER)
    # Divisão temporal (nunca aleatória em séries temporais): treino antes da data de corte, teste depois.
    train, test = stocks[stocks.index < split_date], stocks[stocks.index >= split_date]
    idx_train, idx_test = idx[train.index], idx[test.index]  # o índice nas mesmas datas
    note(f"Treino: {train.index.min().date()} a {train.index.max().date()} ({len(train)} pregões); "
         f"teste: {test.index.min().date()} a {test.index.max().date()} ({len(test)} pregões). "
         "Tracking error (TE) = desvio-padrão anualizado da diferença diária de retorno (carteira - índice). "
         "Os pesos ficam constantes (rebalanceamento diário), uma simplificação.")
    if return_type == "total":
        note("**Atenção:** com `-rt total`, os retornos das ações incluem dividendos, mas o ^OEX não. "
             "A carteira tende a ficar à frente do índice por aproximadamente o dividend yield (~1,5% ao ano).")
    # Com poucos dias de treino, a regressão tem quase tantos parâmetros quanto observações.
    if len(train) < 2 * stocks.shape[1]:
        note(f"**Atenção:** só {len(train)} pregões de treino para até {stocks.shape[1]} pesos. "
             "Os pesos por OLS com muitas ações ficam instáveis (sobreajuste).")

    weights = per_stock["peso_aprox"].fillna(0)  # pesos aproximados (ausentes viram 0)
    rows = {}  # linhas da tabela de resultados: nome -> (TE treino, TE teste)
    # Baseline 1: peso igual (1/N) em todas as ações.
    rows["Peso igual, todas as ações"] = _te_pair(pd.Series(1 / stocks.shape[1], stocks.columns), train, test,
                                                  idx_train, idx_test)
    # Baseline 2: pesos pelo valor de mercado na última data (tem viés de look-ahead: usa informação do futuro).
    rows["Peso de mercado (na data final), todas as ações"] = _te_pair(weights / weights.sum(), train, test,
                                                               idx_train, idx_test)
    # Tamanhos de carteira testados para a curva "TE x número de ações" (limitados ao nº de ações disponíveis).
    ns = sorted({n for n in [5, 10, 15, 20, 30, 40, 50, 75] if n < stocks.shape[1]} | {stocks.shape[1]})
    curve = {"top-N por peso": [], "top-N por peso, pesos OLS": [], "top-N por correlação (treino), pesos OLS": []}
    # Ranking das ações por correlação com o índice, calculado SÓ no treino (sem olhar o teste).
    corr_rank = train.corrwith(idx_train).sort_values(ascending=False).index
    for n in ns:
        top = weights.nlargest(n).index  # as N maiores ações pelo peso
        # (a) as N maiores, com os pesos de mercado renormalizados para somar 1.
        curve["top-N por peso"].append(_te_pair(weights[top] / weights[top].sum(), train, test,
                                                idx_train, idx_test)[1])
        # (b) as N maiores, com pesos estimados por regressão (OLS) no treino.
        curve["top-N por peso, pesos OLS"].append(_te_pair(_ols_weights(train[top], idx_train), train, test,
                                                           idx_train, idx_test)[1])
        # (c) as N mais correlacionadas no treino, com pesos por OLS.
        cr = corr_rank[:n]
        curve["top-N por correlação (treino), pesos OLS"].append(
            _te_pair(_ols_weights(train[cr], idx_train), train, test, idx_train, idx_test)[1])
    # Algumas combinações também entram na tabela, com o TE de treino e de teste.
    for n in (n for n in (10, 20, 30) if n < stocks.shape[1]):
        top = weights.nlargest(n).index
        rows[f"Top {n} por peso de mercado, renormalizado"] = _te_pair(weights[top] / weights[top].sum(), train,
                                                                      test, idx_train, idx_test)
        rows[f"Top {n}, pesos OLS (soma 1) ajustados no treino"] = _te_pair(_ols_weights(train[top], idx_train),
                                                                           train, test, idx_train, idx_test)
    # Checagem de consistência: carteira buy-and-hold com os pesos que o índice teria no início do teste.
    rows["Buy-and-hold, pesos implícitos pelo valor de mercado final (só teste)"] = (
        np.nan, tracking_error(_buy_and_hold_implied(test, weights), idx_test))
    # Monta a tabela: dicionário -> DataFrame (colunas = estratégias) -> transposta (linhas = estratégias).
    te = pd.DataFrame(rows, index=["TE treino", "TE teste"]).T
    table(te.map(lambda v: "-" if pd.isna(v) else f"{v:.2%}"))  # formata em %, com "-" onde não se aplica
    note(
        "- **Por que pesos constantes replicam mal:** um índice ponderado por valor de mercado é uma carteira "
        "buy-and-hold, então seus pesos mudam com os preços (ver o gráfico de deriva dos pesos). Pesos "
        "constantes não acompanham essa deriva, o que aumenta o TE em janelas longas. A linha buy-and-hold "
        "reconstrói os pesos do início do teste a partir do valor de mercado na data final; um TE de ~1% nela "
        "confirma que os dados das ações e do índice são consistentes. O resíduo vem das ações removidas, "
        "do ajuste por free float e das mudanças na composição.\n"
        "- Implicação para o modelo: simular a carteira como **buy-and-hold entre rebalanceamentos** "
        "(mensal/trimestral) e reajustar os pesos em janela móvel."
    )
    _weight_drift_figure(stocks, weights, fig_dir)  # gráfico de como os pesos mudaram ao longo do tempo

    # Gráfico: TE fora da amostra em função do número de ações, para as 3 estratégias.
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for (label, ys), color in zip(curve.items(), [BLUE, ORANGE, AQUA]):
        ax.plot(ns, ys, color=color, marker="o", markersize=5, label=label)
    ax.set_title("Tracking error fora da amostra x número de ações")
    ax.set_xlabel("número de ações na carteira de tracking")
    ax.set_ylabel("TE no teste (anualizado)")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylim(bottom=0)  # eixo y começando em zero, para não exagerar as diferenças
    ax.legend(loc="upper right")
    save(fig, "17_tracking_error_vs_n", fig_dir)
    note(
        "- A curva \"top-N por peso\" tem **viés de look-ahead** (usa o valor de mercado da data final); serve só "
        "como referência. Os pesos OLS ajustados no treino são o primeiro baseline honesto de ML.\n"
        "- O TE cai rápido e depois se estabiliza: o problema de ML é escolher *quais* poucas ações e *quais* "
        "pesos (regressão esparsa / otimização com restrição de cardinalidade, por exemplo LASSO, elastic "
        "net, seleção gulosa, seleção por clusters), com restrições de giro e de peso."
    )

    # Exemplo visual: carteira de 20 ações com pesos OLS (ajustados no treino) vs índice, no teste.
    top = weights.nlargest(min(20, stocks.shape[1])).index
    w = _ols_weights(train[top], idx_train)  # pesos estimados só com dados de treino
    port = test[top] @ w  # retorno diário da carteira = soma de peso x retorno (produto matricial)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True, height_ratios=[2, 1])
    # cumprod de (1 + r) = quanto vale 1 dólar investido no início do teste.
    a1.plot((1 + idx_test).cumprod(), color=BLUE, label="S&P 100")
    a1.plot((1 + port).cumprod(), color=ORANGE, label=f"carteira top-{len(top)} OLS")
    a1.set_title(f"Teste: índice x carteira de {len(top)} ações (valor de 1 investido)")
    a1.legend()
    # Diferença acumulada entre carteira e índice (acima de 0 = carteira à frente do índice).
    diff = (1 + port).cumprod() - (1 + idx_test).cumprod()
    a2.plot(diff.index, diff.values, color=TEXT_2)
    a2.axhline(0, color=GRID)  # linha de referência em zero
    a2.set_title("Diferença acumulada (carteira - índice)")
    save(fig, "18_exemplo_tracking_teste", fig_dir)
    return te


def _implied_weight_history(stocks: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """Pesos históricos aproximados: o valor de mercado da data final "descontado" pela valorização posterior.

    Supõe número de ações constante (ignora recompras/emissões) e a composição atual do índice.
    """
    # Para cada data t: quanto a ação valorizou de t+1 até a data final.
    # Inverte a série no tempo ([::-1]), acumula o produto, desinverte e desloca 1 dia (shift(-1)).
    # O último dia não tem crescimento futuro, então recebe 1 (fillna(1)).
    growth_to_today = (1 + stocks).iloc[::-1].cumprod().iloc[::-1].shift(-1).fillna(1)
    # Valor de mercado relativo em t = valor na data final / crescimento de t até a data final.
    caps = weights.reindex(stocks.columns) / growth_to_today
    # Normaliza cada linha (data) para os pesos somarem 1.
    return caps.div(caps.sum(axis=1), axis=0)


def _buy_and_hold_implied(stocks: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Retornos de uma carteira comprada no 1º dia com os pesos implícitos do índice e nunca rebalanceada."""
    w0 = _implied_weight_history(stocks, weights).iloc[0]  # pesos implícitos no primeiro dia
    # Valor da carteira: cada ação cresce com seu próprio retorno acumulado, ponderada pelo peso inicial.
    value = ((1 + stocks).cumprod() * w0).sum(axis=1)
    # Retorno diário da carteira; o 1º dia (NaN no pct_change) é o valor final do dia 1 menos 1.
    return value.pct_change().fillna(value.iloc[0] - 1)


def _weight_drift_figure(stocks: pd.DataFrame, weights: pd.Series, fig_dir: Path) -> None:
    """Gráfico do peso implícito das 5 maiores ações ao longo do tempo."""
    hist = _implied_weight_history(stocks, weights)
    top = weights.nlargest(5).index  # as 5 maiores na data final
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for t, color in zip(top, SERIES):
        ax.plot(hist.index, hist[t], color=color, label=t)
        # Rótulo com o ticker no fim de cada linha.
        ax.annotate(t, (hist.index[-1], hist[t].iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    fontsize=8, color=TEXT_2, va="center")
    ax.set_title("Peso implícito no índice das 5 maiores ações (deriva do buy-and-hold)")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="upper left", ncol=5)  # legenda em uma linha
    save(fig, "16_deriva_dos_pesos", fig_dir)
    note(f"- O peso implícito da {top[0]} foi de {hist[top[0]].iloc[0]:.1%} para {hist[top[0]].iloc[-1]:.1%}: "
         "os pesos estão longe de ser estacionários. (Aproximação: número de ações constante.)")


def _ols_weights(x: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Pesos de mínimos quadrados com soma(w) = 1 (100% investido, venda a descoberto permitida), sem intercepto."""
    xv, yv = x.values, y.values  # matrizes NumPy: X (dias x ações) e y (retornos do índice)
    n = xv.shape[1]  # número de ações
    # Problema: minimizar ||Xw - y||^2 sujeito a 1'w = 1. Com um multiplicador de Lagrange (lambda),
    # as condições de ótimo (KKT) formam um sistema linear:
    #   [ 2X'X  1 ] [ w      ]   [ 2X'y ]
    #   [ 1'    0 ] [ lambda ] = [ 1    ]
    kkt = np.block([[2 * xv.T @ xv, np.ones((n, 1))], [np.ones((1, n)), np.zeros((1, 1))]])
    rhs = np.concatenate([2 * xv.T @ yv, [1.0]])  # lado direito do sistema
    # lstsq em vez de solve: continua funcionando se o sistema for quase singular (poucos dias de treino).
    return pd.Series(np.linalg.lstsq(kkt, rhs, rcond=None)[0][:n], index=x.columns)


def _te_pair(w: pd.Series, train, test, idx_train, idx_test) -> tuple[float, float]:
    """TE de treino e de teste de uma carteira com pesos fixos `w`."""
    # train[w.index] @ w = retorno diário da carteira (soma de peso x retorno de cada ação).
    return (tracking_error(train[w.index] @ w, idx_train), tracking_error(test[w.index] @ w, idx_test))

#################################################################################################
# execução completa


def run_eda(input_folder: Path,
            output_folder: Path,
            split_date: str | None,
            outlier_threshold: float,
            return_type: str,
            highlights: list[str]
            ) -> None:
    """Executa todas as seções da EDA e salva o relatório em `output_folder`."""
    fig_dir = output_folder / "figures"  # pasta dos gráficos
    fig_dir.mkdir(parents=True, exist_ok=True)  # cria a pasta se não existir
    prices, volume, returns, constituents = load(input_folder, return_type)  # carrega os dados

    # Data de corte treino/teste: a informada, ou a que deixa DEFAULT_TRAIN_FRACTION dos pregões no treino.
    dates = returns.index
    split = pd.Timestamp(split_date) if split_date else dates[int(len(dates) * DEFAULT_TRAIN_FRACTION)]
    # A data de corte precisa deixar pelo menos 21 pregões (~1 mês) de cada lado.
    if not dates[21] <= split <= dates[-21]:
        raise SystemExit(f"split_date {split.date()} is outside the data range "
                         f"({dates[0].date()} -> {dates[-1].date()}); choose a date inside it.")

    report.clear()  # começa o relatório do zero
    report.append("# S&P 100 index tracking: análise exploratória de dados\n")  # título do relatório
    note(f"Gerado por `src/eda/run_eda.py`. Fontes: Yahoo Finance (preços) e Wikipedia (composição). "
         f"Período: {dates[0].date()} a {dates[-1].date()}.\n\n"
         f"**Retornos:** simples, r_t = P_t / P_(t-1) - 1. O guia aceita simples ou logarítmico; usamos o "
         "simples porque o retorno de uma carteira é a média ponderada dos retornos simples das ações (com "
         "log isso não vale), e é exatamente isso que o modelo de tracking calcula.\n\n"
         + ("**Preço:** `Close` ajustado só por splits (`-rt price`). O guia recomenda `Adj Close`, que também "
            "desconta os dividendos. Mas o ^OEX é um índice de *preço* (não reinveste dividendos); com Adj Close, "
            "as ações ganhariam ~1,5% ao ano de dividendos que o índice não tem, e o modelo aprenderia uma "
            "diferença falsa. Para comparar, rode com `-rt total`."
            if return_type == "price" else
            "**Preço:** `Adj Close`, ajustado por splits e dividendos (`-rt total`), como recomenda o guia. "
            "Atenção: o ^OEX não inclui dividendos (ver seção 7)."))
    # Executa as seções em ordem; cada uma acrescenta texto, tabelas e gráficos ao relatório.
    data_quality(prices, volume, returns, outlier_threshold, fig_dir)
    index_analysis(prices, returns, fig_dir)
    stock_stats = stock_returns_analysis(returns, highlights, fig_dir)
    per_stock = index_correlation_analysis(returns, constituents, fig_dir)
    stock_correlation_analysis(returns, per_stock, fig_dir)
    volatility_analysis(returns, fig_dir)
    te = tracking_baselines(returns, per_stock, split, return_type, fig_dir)

    # Salva as tabelas por ação e dos baselines para uso posterior (seleção de ativos, comparação de modelos).
    # (a volatilidade já está nas duas tabelas, então é removida de uma delas antes de juntar)
    per_stock.join(stock_stats.drop(columns="volatilidade_anual")).to_csv(output_folder / "per_stock_stats.csv")
    te.to_csv(output_folder / "tracking_baselines.csv")
    # Junta todos os trechos e grava o markdown final.
    (output_folder / "eda_summary.md").write_text("\n".join(report))
    print(f"Wrote {output_folder / 'eda_summary.md'} and {len(list(fig_dir.glob('*.png')))} figures")

#################################################################################################
# funções de leitura dos parâmetros do terminal


def get_args_dict() -> dict:
    """
    Lê os parâmetros do terminal e devolve um dicionário com eles.
    """
    # criando o parser; a docstring do módulo (com exemplos de uso) aparece no --help
    parser = ArgumentParser(description=__doc__,
                            formatter_class=RawDescriptionHelpFormatter)

    # pasta de entrada (saída do data_mining.py)
    parser.add_argument('-i', '--input_folder',
                        dest='input_folder',
                        type=Path,
                        required=True,
                        help='defines path to data folder created by data_mining')

    # pasta de saída do relatório
    parser.add_argument('-o', '--output_folder',
                        dest='output_folder',
                        type=Path,
                        required=True,
                        help='defines path to report output folder')

    # data de corte treino/teste
    parser.add_argument('-sd', '--split_date',
                        dest='split_date',
                        default=None,
                        help=f'defines train/test split date, YYYY-MM-DD '
                             f'(default: first {DEFAULT_TRAIN_FRACTION:.0%} of trading days are train)')

    # limite de retorno diário para marcar outliers
    parser.add_argument('-ot', '--outlier_threshold',
                        dest='outlier_threshold',
                        type=float,
                        default=DEFAULT_OUTLIER_THRESHOLD,
                        help=f'defines absolute daily return flagged as outlier (default: {DEFAULT_OUTLIER_THRESHOLD})')

    # tipo de retorno: só preço ou total (com dividendos)
    parser.add_argument('-rt', '--return_type',
                        dest='return_type',
                        choices=['price', 'total'],
                        default='price',
                        help='defines "price" (Close, matches ^OEX) or "total" (Adj Close, with dividends) '
                             'returns (default: price)')

    # ações destacadas no gráfico de base 100
    parser.add_argument('-hl', '--highlights',
                        dest='highlights',
                        nargs='+',
                        default=DEFAULT_HIGHLIGHTS,
                        help=f'defines up to 4 tickers highlighted in the normalized performance chart '
                             f'(default: {" ".join(DEFAULT_HIGHLIGHTS)})')

    # lendo os argumentos digitados
    args = parser.parse_args()

    # validando a data de corte (pd.Timestamp dá erro se o formato for inválido)
    if args.split_date is not None:
        try:
            pd.Timestamp(args.split_date)
        except ValueError as exc:
            parser.error(f'invalid split_date: {exc}')

    # a pasta de entrada precisa ter os arquivos do data_mining
    if not (args.input_folder / 'processed' / 'returns.parquet').exists():
        parser.error(f'no data found in {args.input_folder}; run "python3 -m src.eda.data_mining -o <folder>" first')

    # criando o dicionário de argumentos
    args_dict = vars(args)

    # devolvendo o dicionário de argumentos
    return args_dict

#################################################################################################
# função principal


def main():
    """Executa o código principal."""
    # lendo o dicionário de argumentos
    args_dict = get_args_dict()

    # pasta de entrada
    input_folder = args_dict['input_folder']

    # pasta de saída
    output_folder = args_dict['output_folder']

    # data de corte treino/teste
    split_date = args_dict['split_date']

    # limite de outlier
    outlier_threshold = args_dict['outlier_threshold']

    # tipo de retorno
    return_type = args_dict['return_type']

    # ações destacadas
    highlights = args_dict['highlights']

    # imprimindo os parâmetros de execução
    print_execution_parameters(params_dict=args_dict)

    # executando a EDA
    run_eda(input_folder=input_folder,
            output_folder=output_folder,
            split_date=split_date,
            outlier_threshold=outlier_threshold,
            return_type=return_type,
            highlights=highlights)

#################################################################################################
# executando a função principal


# Só executa quando o arquivo é rodado diretamente (python3 -m src.eda.run_eda).
if __name__ == "__main__":
    main()

#################################################################################################
# fim do módulo
