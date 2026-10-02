"""Análise exploratória (EDA) do S&P 100 para o projeto de index tracking.

Lê a pasta criada pelo data_mining.py e escreve em <output_folder>:
  eda_summary.md, figures/*.png, per_stock_stats.csv, tracking_baselines.csv

Seções (checklist do Guia de Dados & EDA):
  1. Qualidade dos dados   2. O índice   3. Retornos das ações   4. Correlação com o índice
  5. Correlação entre ações   6. Volatilidade e crises   7. Baseline de tracking
  8. Comparação com o Ibovespa

Uso:
  python3 -m src.eda.run_eda -i data -o reports                       # todo o período baixado
  python3 -m src.eda.run_eda -i data -o reports/2025 -y 2025          # só um ano
  python3 -m src.eda.run_eda -i data -o reports/covid -s 2020-01-01 -e 2021-01-01
  python3 -m src.eda.run_eda -i data -o reports/total -sd 2023-01-01 -rt total
"""
#################################################################################################
# imports
import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # salva os gráficos em arquivo, sem abrir janelas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy import stats

from src._execution_formatting import print_execution_parameters

#################################################################################################
# constantes

INDEX = "^OEX"
IBOV = "^BVSP"
DAYS = 252  # pregões por ano, para anualizar
OUTLIER = 0.15  # retorno diário acima de ±15% é extremo (valor do guia)
HIGHLIGHTS = ["AAPL", "MSFT", "NVDA", "TSLA"]  # destaques do gráfico de base 100 (os do guia)
EVENTS = {  # linha do tempo do guia (seção 7.1)
    "2018-10-01": "Guerra comercial EUA x China",
    "2020-03-23": "COVID-19 (mínimo do mercado)",
    "2020-11-09": "Anúncio das vacinas",
    "2022-01-03": "Início da alta de juros (Fed)",
    "2022-11-11": "Crise FTX (cripto)",
    "2023-10-09": "Conflito Israel-Hamas",
    "2024-01-02": "Rally de IA (NVDA, MSFT)",
}

# cores
BLUE, ORANGE, AQUA, YELLOW, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948"
NAVY, GRAY, LIGHT = "#0d366b", "#52514e", "#c3c2b7"
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})

#################################################################################################
# funções auxiliares


def save(fig: plt.Figure, fig_dir: Path, name: str) -> str:
    """Salva a figura em <fig_dir>/<name>.png e devolve o link markdown para colocar no relatório."""
    fig.tight_layout()  # ajusta os espaços para nada ficar cortado
    fig.savefig(fig_dir / f"{name}.png", dpi=150)
    plt.close(fig)  # libera a memória da figura
    return f"![{name}](figures/{name}.png)"


def mark_events(ax: plt.Axes, dates: pd.DatetimeIndex) -> None:
    """Desenha no gráfico uma linha pontilhada numerada em cada evento de EVENTS.

    Só marca os eventos que caem dentro do período dos dados. O número de cada linha
    corresponde à tabela de eventos da seção 6 do relatório.
    """
    for i, day in enumerate(pd.to_datetime(list(EVENTS)), start=1):
        if dates[0] <= day <= dates[-1]:
            ax.axvline(day, color=GRAY, linestyle=":", linewidth=1)
            # número do evento no topo do gráfico (y = 1 = topo do eixo)
            ax.annotate(str(i), (day, 1), xycoords=("data", "axes fraction"), xytext=(2, -10),
                        textcoords="offset points", fontsize=8, color=GRAY)


def cut(df: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    """Recorta as linhas entre start_date e end_date (fim exclusivo, como no data_mining)."""
    df = df.loc[start_date:]  # None = desde o início
    return df[df.index < end_date] if end_date else df


def ols_weights(x: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Pesos da carteira que melhor replicam o índice por mínimos quadrados (OLS), somando 1.

    x = retornos diários das ações escolhidas (dias x ações); y = retornos diários do índice.
    Procura os pesos w que deixam sum(w_i * x_i) o mais perto possível de y.
    A soma 1 (100% investido) é garantida escrevendo a última ação como 1 - soma das outras:
    y - x_última = sum(w_i * (x_i - x_última)), que é uma regressão comum.
    """
    base = x.iloc[:, -1]  # retornos da última ação
    # regressão de (y - base) nas diferenças (x_i - base); [0] pega só os coeficientes
    w = np.linalg.lstsq(x.iloc[:, :-1].sub(base, axis=0), y - base, rcond=None)[0]
    return pd.Series(np.append(w, 1 - w.sum()), index=x.columns)  # a última recebe 1 - soma

#################################################################################################
# seções do relatório (cada uma devolve seu texto em markdown)


def data_quality(prices: pd.DataFrame, returns: pd.DataFrame) -> str:
    """Seção 1: os dados estão completos e confiáveis?

    Extrai: quantas ações foram baixadas e mantidas, o % de dias sem preço por ação,
    quais ações foram removidas e os retornos diários extremos (outliers).
    Mostra: se dá para confiar nos dados e o que foi feito com os problemas encontrados.
    """
    stocks = prices.drop(columns=INDEX)[prices[INDEX].notna()]  # só dias de pregão do índice
    missing = stocks.isna().mean()  # fração de dias sem preço de cada ação
    missing = missing[missing > 0].sort_values(ascending=False)  # só quem tem algum buraco
    dropped = sorted(set(stocks.columns) - set(returns.columns))  # baixadas, mas fora dos retornos
    # stack() transforma a tabela (data x ação) numa lista longa de (data, ação) -> retorno
    moves = returns.drop(columns=INDEX).stack()
    outliers = moves[moves.abs() > OUTLIER]  # dias com alta ou queda maior que 15%
    # os 10 maiores em módulo, com rótulo "data ticker" para a tabela
    top_moves = outliers.reindex(outliers.abs().sort_values(ascending=False).index).head(10)
    top_moves.index = [f"{day:%Y-%m-%d} {ticker}" for day, ticker in top_moves.index]
    return "\n\n".join([
        "## 1. Qualidade dos dados",
        f"- {stocks.shape[1]} ações baixadas; **{returns.shape[1] - 1} mantidas**, {len(returns)} pregões "
        f"({returns.index[0].date()} a {returns.index[-1].date()}).\n"
        f"- Removidas por terem menos de 95% do histórico (IPOs recentes): {', '.join(dropped) or 'nenhuma'}. "
        "O critério é mais rígido que os 20% do guia porque o modelo precisa de um painel sem buracos: uma "
        "ação com 15% faltando cortaria 15% do histórico de todas as outras.\n"
        "- Buracos de até 5 pregões foram preenchidos com o último preço (forward fill).\n"
        "- **Viés de sobrevivência:** usamos a composição de hoje em todo o período; ações que saíram do "
        "índice não aparecem. Os backtests tendem a ser otimistas.",
        "Ações com dados faltantes:", missing.to_frame("faltante").to_markdown(floatfmt=".1%"),
        f"**Outliers:** {len(outliers)} retornos diários acima de ±{OUTLIER:.0%} em "
        f"{outliers.index.get_level_values(1).nunique()} ações. Batem com eventos reais (resultados, crises), "
        "então foram mantidos. Os 10 maiores:",
        top_moves.to_frame("retorno").to_markdown(floatfmt=".1%"),
    ])


def index_analysis(prices: pd.DataFrame, returns: pd.DataFrame, fig_dir: Path) -> str:
    """Seção 2: como o próprio índice (^OEX) se comportou, ou seja, o alvo que queremos replicar.

    Extrai: retorno e volatilidade anuais, Sharpe, pior e melhor dia, drawdown (queda desde o
    pico), retorno de cada mês e de cada ano, e o teste de normalidade dos retornos.
    Gráficos: nível + drawdown, histograma dos retornos x curva normal, barras de retorno mensal.
    Mostra: as crises do período e que os retornos têm caudas pesadas (dias extremos são mais
    comuns do que uma distribuição normal preveria).
    """
    r = returns[INDEX]  # retornos diários do índice
    level = prices[INDEX].dropna()  # pontuação do índice
    summary = pd.Series({
        # retorno composto: quanto 1 dólar cresceu por ano, em média
        "retorno anual (composto)": (1 + r).prod() ** (DAYS / len(r)) - 1,
        # risco: desvio-padrão diário x raiz de 252 (a variância cresce com o tempo)
        "volatilidade anual": r.std() * np.sqrt(DAYS),
        # retorno por unidade de risco (taxa livre de risco = 0)
        "sharpe (rf = 0)": r.mean() / r.std() * np.sqrt(DAYS),
        "pior dia": r.min(),
        "melhor dia": r.max(),
    })
    drawdown = level / level.cummax() - 1  # queda desde o pico anterior (cummax = maior valor até a data)
    monthly = (1 + r).resample("ME").prod() - 1  # retorno acumulado de cada mês ("ME" = fim do mês)
    annual = (1 + r).resample("YE").prod() - 1  # retorno acumulado de cada ano ("YE" = fim do ano)
    annual.index = annual.index.year  # mostra só o ano na tabela

    # gráfico 1: nível do índice (em cima) e drawdown (embaixo), com o mesmo eixo de datas
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, height_ratios=[2, 1])
    a1.plot(level, color=BLUE)
    a1.set_title("S&P 100 (^OEX): nível do índice")
    a2.fill_between(drawdown.index, drawdown, 0, color=RED, alpha=0.35)
    a2.set_title("Drawdown (queda desde o pico)")
    a2.yaxis.set_major_formatter(PercentFormatter(1))  # eixo em %
    fig_level = save(fig, fig_dir, "01_indice_drawdown")

    # gráfico 2: histograma dos retornos comparado com uma normal de mesma média e desvio
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.hist(r, bins=120, density=True, color=BLUE, alpha=0.8, label="distribuição real")
    x = np.linspace(r.min(), r.max(), 400)  # pontos para desenhar a curva normal
    ax.plot(x, stats.norm.pdf(x, r.mean(), r.std()), color=ORANGE, label="normal teórica")
    ax.set_title("Distribuição dos retornos diários do S&P 100")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.legend()
    fig_hist = save(fig, fig_dir, "02_distribuicao_retornos")

    # gráfico 3: uma barra por mês, azul se o mês fechou em alta e vermelha se fechou em queda
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.bar(monthly.index, monthly, width=20, color=np.where(monthly >= 0, BLUE, RED))
    ax.set_title("Retorno mensal do S&P 100 (azul = alta, vermelho = queda)")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig_month = save(fig, fig_dir, "03_retorno_mensal")

    return "\n\n".join([
        "## 2. O índice (^OEX)",
        summary.to_frame("^OEX").to_markdown(floatfmt=".3f"),
        # curtose > 0 e p-valor do Jarque-Bera ~0 = retornos não normais
        f"- Curtose em excesso de **{r.kurt():.1f}** (a normal tem 0) e Jarque-Bera p = "
        f"{stats.jarque_bera(r).pvalue:.1e}: os retornos **não são normais**, têm caudas pesadas.\n"
        f"- Drawdown máximo: **{drawdown.min():.1%}** em {drawdown.idxmin().date()}.\n"
        f"- Pior mês: **{monthly.min():.1%}** ({monthly.idxmin():%m/%Y}); melhor: {monthly.max():.1%} "
        f"({monthly.idxmax():%m/%Y}).",
        fig_level, fig_hist, fig_month,
        "Retorno por ano (o primeiro e o último ano podem estar incompletos):",
        annual.to_frame("retorno").to_markdown(floatfmt=".1%"),
    ])


def stock_analysis(returns: pd.DataFrame, fig_dir: Path) -> tuple[str, pd.DataFrame]:
    """Seção 3: o perfil de risco e retorno de cada ação.

    Extrai: por ação, retorno médio anual, volatilidade anual, Sharpe, pior e melhor dia.
    Gráficos: performance em base 100 (todas as ações + destaques + índice) e risco x retorno.
    Mostra: o quanto as ações se afastam do índice e que o índice é bem menos volátil que
    uma ação típica (efeito da diversificação).
    Devolve também a tabela por ação, que vai para o per_stock_stats.csv.
    """
    stocks = returns.drop(columns=INDEX)  # só as ações, sem o índice
    table = pd.DataFrame({
        "retorno_anual": stocks.mean() * DAYS,  # média diária x 252 (como no guia)
        "volatilidade_anual": stocks.std() * np.sqrt(DAYS),
        "sharpe": stocks.mean() / stocks.std() * np.sqrt(DAYS),
        "pior_dia": stocks.min(),
        "melhor_dia": stocks.max(),
    })

    # base 100: quanto valeriam 100 dólares investidos no primeiro dia em cada ação e no índice
    norm = 100 * (1 + returns).cumprod()
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(norm[stocks.columns], color=LIGHT, linewidth=0.5)  # todas as ações, ao fundo
    for ticker, color in zip(HIGHLIGHTS, [BLUE, ORANGE, AQUA, YELLOW]):
        ax.plot(norm[ticker], color=color, label=ticker)  # ações em destaque
    ax.plot(norm[INDEX], color=NAVY, linewidth=2.5, label="S&P 100 (índice)")  # índice por cima
    mark_events(ax, norm.index)
    ax.set_yscale("log")  # sem escala log, a NVDA achataria todas as outras
    ax.set_title(f"Performance normalizada (base 100 em {norm.index[0]:%m/%Y}, escala log)")
    ax.legend(loc="upper left")
    fig_norm = save(fig, fig_dir, "04_performance_normalizada")

    # risco x retorno: cada ponto é uma ação; o losango é o índice
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(table["volatilidade_anual"], table["retorno_anual"], color=BLUE, alpha=0.6, label="ações")
    ax.scatter(returns[INDEX].std() * np.sqrt(DAYS), returns[INDEX].mean() * DAYS, color=NAVY, marker="D",
               s=100, label="S&P 100 (índice)")
    for t in table.nlargest(5, "sharpe").index:  # escreve o nome das 5 com melhor Sharpe
        ax.annotate(t, (table.at[t, "volatilidade_anual"], table.at[t, "retorno_anual"]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_title("Risco x retorno das ações")
    ax.set_xlabel("volatilidade anual")
    ax.set_ylabel("retorno médio anual")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend()
    fig_risk = save(fig, fig_dir, "05_risco_retorno")

    text = "\n\n".join([
        "## 3. Retornos das ações",
        "Retornos simples, r = P_t / P_(t-1) - 1: o retorno de uma carteira é a média ponderada dos retornos "
        "simples das ações (com log isso não vale), que é o que o modelo de tracking calcula.",
        fig_norm,
        f"O índice tem volatilidade de {returns[INDEX].std() * np.sqrt(DAYS):.1%}, contra a mediana de "
        f"{table['volatilidade_anual'].median():.1%} das ações: a diversificação elimina boa parte do risco "
        "de cada empresa. A carteira de tracking precisa de ações suficientes para isso.",
        fig_risk,
        "Maior retorno:", table.nlargest(10, "retorno_anual").to_markdown(floatfmt=".3f"),
        "Menor volatilidade:", table.nsmallest(10, "volatilidade_anual").to_markdown(floatfmt=".3f"),
        "Melhor Sharpe:", table.nlargest(10, "sharpe").to_markdown(floatfmt=".3f"),
    ])
    return text, table


def index_correlation(returns: pd.DataFrame, constituents: pd.DataFrame, fig_dir: Path) -> tuple[str, pd.DataFrame]:
    """Seção 4 (a mais importante): quais ações se movem como o índice?

    Extrai: por ação, a correlação com o índice (quanto ela sobe e cai junto com ele, de -1 a 1)
    e o beta (quanto ela se move quando o índice se move 1%); por setor, a correlação média,
    o número de ações e o peso no índice.
    Gráficos: ranking de correlação de todas as ações e boxplot da correlação por setor.
    Mostra: as melhores candidatas para a carteira de tracking e quais setores são mais
    fáceis de replicar. Devolve a tabela ordenada da maior para a menor correlação.
    """
    idx = returns[INDEX]
    stocks = returns.drop(columns=INDEX)
    table = pd.DataFrame({
        "setor": constituents["sector"],
        "peso": constituents["approx_weight"],  # peso aproximado no índice (do data_mining)
        "corr_indice": stocks.corrwith(idx),  # correlação de cada coluna com o índice
        "beta": stocks.apply(lambda s: s.cov(idx)) / idx.var(),  # beta = cov(ação, índice) / var(índice)
    }).loc[stocks.columns].sort_values("corr_indice", ascending=False)  # só ações mantidas, da maior para a menor
    # resumo por setor: correlação média, quantas ações e quanto do índice o setor representa
    sectors = table.groupby("setor").agg(corr_media=("corr_indice", "mean"), n_acoes=("corr_indice", "size"),
                                         peso=("peso", "sum")).sort_values("corr_media", ascending=False)

    # ranking: uma barra por ação, com as linhas de referência 0,7 e 0,5 do guia
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.bar(table.index, table["corr_indice"], color=BLUE)
    for level in (0.7, 0.5):
        ax.axhline(level, color=GRAY, linestyle="--", linewidth=1)
    ax.tick_params(axis="x", rotation=90, labelsize=6)  # tickers na vertical
    ax.set_title("Correlação de cada ação com o S&P 100 (linhas de referência: 0,7 e 0,5)")
    fig_bar = save(fig, fig_dir, "06_correlacao_com_indice")

    # boxplot: a distribuição da correlação dentro de cada setor
    order = sectors.index[::-1]  # maior correlação no topo
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.boxplot([table.loc[table["setor"] == s, "corr_indice"] for s in order], vert=False, tick_labels=order)
    ax.set_title("Correlação com o S&P 100 por setor")
    fig_box = save(fig, fig_dir, "07_correlacao_por_setor")

    text = "\n\n".join([
        "## 4. Correlação com o índice (a análise central do projeto)",
        f"- Mediana **{table['corr_indice'].median():.2f}**, de {table['corr_indice'].min():.2f} a "
        f"{table['corr_indice'].max():.2f}. **{(table['corr_indice'] > 0.7).mean():.0%}** das ações têm "
        f"correlação > 0,7 e {(table['corr_indice'] > 0.5).mean():.0%} têm > 0,5.\n"
        f"- As 10 maiores ações somam **{constituents['approx_weight'].nlargest(10).sum():.1%}** do índice.",
        fig_bar,
        "As 15 mais correlacionadas (melhores candidatas para o tracking):",
        table.head(15).to_markdown(floatfmt=".3f"),
        "As 10 menos correlacionadas:", table.tail(10).to_markdown(floatfmt=".3f"),
        "Por setor (setores com correlação alta são mais fáceis de replicar):",
        sectors.to_markdown(floatfmt=".3f"), fig_box,
    ])
    return text, table


def stock_correlation(returns: pd.DataFrame, corr_ranking: pd.Index, fig_dir: Path) -> str:
    """Seção 5: quais ações andam juntas entre si?

    Extrai: a correlação de cada par de ações, a correlação média entre pares e os pares
    acima de 0,9 (redundantes).
    Gráfico: heatmap das 30 ações mais correlacionadas com o índice (corr_ranking vem da seção 4).
    Mostra: ações muito parecidas entre si não diversificam, então pôr as duas na carteira
    não ajuda a replicar o índice.
    """
    corr = returns.drop(columns=INDEX).corr()  # matriz ação x ação
    # a matriz é simétrica (A/B = B/A): pega só o triângulo acima da diagonal para ter cada par uma vez
    pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    pairs = pairs.sort_values(ascending=False)
    pairs.index = [f"{a} / {b}" for a, b in pairs.index]  # rótulo "AÇÃO1 / AÇÃO2"

    top30 = corr.loc[corr_ranking[:30], corr_ranking[:30]]  # recorte com as 30 do topo do ranking
    lower = top30.where(np.tril(np.ones(top30.shape, dtype=bool), k=-1))  # só abaixo da diagonal
    fig, ax = plt.subplots(figsize=(12, 10.5))
    im = ax.imshow(lower, cmap="RdBu_r", vmin=-1, vmax=1)  # vermelho = correlação positiva
    for (i, j), v in np.ndenumerate(lower.values):  # escreve o valor em cada célula
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5, color="white" if v > 0.6 else "black")
    ax.set_xticks(range(30), top30.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(30), top30.index, fontsize=7)
    ax.grid(False)
    ax.set_title("Correlação entre as 30 ações mais correlacionadas com o S&P 100")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig_heat = save(fig, fig_dir, "08_heatmap_top30")

    return "\n\n".join([
        "## 5. Correlação entre ações",
        f"- Correlação média entre pares: **{pairs.mean():.2f}**.\n"
        f"- Pares com correlação > 0,90 (redundantes: colocar os dois na carteira não diversifica): "
        f"**{(pairs > 0.9).sum()}**. GOOG / GOOGL são duas classes da mesma empresa (Alphabet).",
        fig_heat,
        "Os 10 pares mais correlacionados:", pairs.head(10).to_frame("correlacao").to_markdown(floatfmt=".3f"),
    ])


def volatility(returns: pd.DataFrame, fig_dir: Path) -> str:
    """Seção 6: quando o mercado ficou mais arriscado?

    Extrai: a volatilidade móvel do índice (desvio-padrão dos últimos 21 pregões, anualizado),
    sua média, o pico e a data do pico, e a tabela de eventos dentro do período.
    Gráfico: volatilidade ao longo do tempo com os eventos numerados.
    Mostra: como crises (COVID, alta de juros) aparecem nos dados como picos de risco.
    """
    r = returns[INDEX]
    vol = r.rolling(21).std() * np.sqrt(DAYS)  # janela de ~1 mês, anualizada
    # tabela de eventos numerada (mesma numeração dos gráficos), só com os que estão no período
    events = pd.DataFrame({"data": list(EVENTS), "evento": list(EVENTS.values())}, index=range(1, len(EVENTS) + 1))
    events = events[pd.to_datetime(events["data"]).between(r.index[0], r.index[-1])]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(vol.index, vol, 0, color=BLUE, alpha=0.3)
    ax.axhline(vol.mean(), color=GRAY, linestyle="--", linewidth=1, label=f"média: {vol.mean():.1%}")
    mark_events(ax, vol.index)
    ax.set_title("Volatilidade móvel do S&P 100 (21 pregões, anualizada)")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="upper right")
    fig_vol = save(fig, fig_dir, "09_volatilidade_movel")

    return "\n\n".join([
        "## 6. Volatilidade e crises",
        "Eventos (numerados nos gráficos):", events.to_markdown(),
        fig_vol,
        f"Volatilidade média de **{vol.mean():.1%}** ao ano; máxima de **{vol.max():.1%}** em "
        f"{vol.idxmax().date()}.",
    ])


def tracking_baseline(returns: pd.DataFrame, split: pd.Timestamp, fig_dir: Path) -> tuple[str, pd.DataFrame]:
    """Seção 7: um primeiro teste de tracking, que serve de referência para o modelo de ML.

    Extrai: o tracking error (TE) no treino e no teste de carteiras simples: peso igual em todas
    as ações, e as top N mais correlacionadas no treino (N = 5, 10, 20, 30, 50) com pesos OLS.
    Gráfico: TE x número de ações, no treino e no teste.
    Mostra: quanto o erro cai ao adicionar ações e o sobreajuste (TE de teste > TE de treino).
    TE = desvio-padrão anualizado da diferença diária entre o retorno da carteira e o do índice.
    """
    idx = returns[INDEX]
    stocks = returns.drop(columns=INDEX)
    train = returns.index < split  # True nos dias de treino (divisão no tempo, nunca aleatória)
    # ranking de correlação calculado só no treino, para não usar informação do futuro
    ranking = stocks[train].corrwith(idx[train]).sort_values(ascending=False).index

    def tracking_error(w: pd.Series) -> pd.Series:
        """TE no treino e no teste de uma carteira com pesos fixos w."""
        diff = stocks[w.index] @ w - idx  # retorno da carteira (soma de peso x retorno) - retorno do índice
        return pd.Series({"TE treino": diff[train].std() * np.sqrt(DAYS),
                          "TE teste": diff[~train].std() * np.sqrt(DAYS)})  # ~train = dias de teste

    # carteira 1: peso igual (1/N) em todas as ações
    rows = {"peso igual, todas as ações": tracking_error(pd.Series(1 / stocks.shape[1], stocks.columns))}
    # carteiras 2 em diante: as N mais correlacionadas, com pesos ajustados no treino
    for n in (5, 10, 20, 30, 50):
        rows[f"top {n} por correlação, pesos OLS"] = tracking_error(
            ols_weights(stocks.loc[train, ranking[:n]], idx[train]))
    table = pd.DataFrame(rows).T  # uma linha por carteira, colunas TE treino / TE teste

    ols_rows = table.iloc[1:]  # só as carteiras top N (sem a de peso igual)
    ns = [int(name.split()[1]) for name in ols_rows.index]  # "top 20 por ..." -> 20
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(ns, ols_rows["TE treino"], color=BLUE, marker="o", label="treino")
    ax.plot(ns, ols_rows["TE teste"], color=ORANGE, marker="o", label="teste (fora da amostra)")
    ax.set_title("Tracking error x número de ações (top N por correlação, pesos OLS)")
    ax.set_xlabel("número de ações")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylim(bottom=0)  # eixo começando em zero para não exagerar as diferenças
    ax.legend()
    fig_te = save(fig, fig_dir, "10_tracking_error_vs_n")

    text = "\n\n".join([
        "## 7. Baseline de tracking",
        f"Treino antes de {split.date()} ({train.sum()} pregões), teste a partir daí ({(~train).sum()} pregões). "
        "Tracking error (TE) = desvio-padrão anualizado da diferença diária de retorno entre carteira e índice. "
        "Ações escolhidas pela correlação no treino; pesos por mínimos quadrados (somando 1), fixos no teste.",
        table.to_markdown(floatfmt=".2%"), fig_te,
        "- A diferença entre treino e teste mostra o sobreajuste: com mais ações, o TE de treino sempre cai, "
        "mas o de teste não acompanha.\n"
        "- Os pesos ficam fixos, mas o índice é ponderado por valor de mercado: seus pesos mudam com os "
        "preços. Reestimar os pesos periodicamente (rebalanceamento) deve reduzir o TE.",
    ])
    return text, table


def ibovespa_comparison(returns: pd.DataFrame, ibov: pd.Series, fig_dir: Path) -> str:
    """Seção 8: o S&P 100 comparado com o Ibovespa (referência para o investidor brasileiro).

    Extrai: retorno e volatilidade anuais, Sharpe e drawdown máximo dos dois índices, e a
    correlação entre eles com retornos diários e semanais.
    Gráfico: os dois em base 100, cada um no calendário da sua bolsa.
    Mostra: o quanto os dois mercados andam juntos. Não é comparação exata: moedas diferentes
    (US$ x R$), o ^OEX é só preço e o Ibovespa reinveste dividendos.
    """
    both = {"S&P 100 (US$)": returns[INDEX], "Ibovespa (R$)": ibov}

    def summary(r: pd.Series) -> pd.Series:
        level = (1 + r).cumprod()
        return pd.Series({"retorno anual (composto)": level.iloc[-1] ** (DAYS / len(r)) - 1,
                          "volatilidade anual": r.std() * np.sqrt(DAYS),
                          "sharpe (rf = 0)": r.mean() / r.std() * np.sqrt(DAYS),
                          "drawdown máximo": (level / level.cummax() - 1).min()})

    table = pd.DataFrame({name: summary(r) for name, r in both.items()})
    daily = pd.concat(both, axis=1, join="inner")  # só os dias em que as duas bolsas abriram
    # semanal: as bolsas fecham em horários diferentes, o que reduz a correlação diária
    weekly = pd.concat(both, axis=1).fillna(0).add(1).resample("W-FRI").prod() - 1

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for (name, r), color in zip(both.items(), [NAVY, AQUA]):
        ax.plot(100 * (1 + r).cumprod(), color=color, label=name)
    mark_events(ax, returns.index)
    ax.set_title("S&P 100 x Ibovespa (base 100, cada um na sua moeda)")
    ax.legend(loc="best")
    fig_ibov = save(fig, fig_dir, "11_sp100_vs_ibovespa")

    return "\n\n".join([
        "## 8. Comparação com o Ibovespa (^BVSP)",
        "Referência, não alvo do tracking: moedas diferentes (sem conversão pelo câmbio), calendários "
        "diferentes (B3 x NYSE) e o Ibovespa inclui dividendos, enquanto o ^OEX não.",
        table.to_markdown(floatfmt=".3f"), fig_ibov,
        f"- Correlação diária (dias em comum): **{daily.corr().iloc[0, 1]:.2f}**; semanal: "
        f"**{weekly.corr().iloc[0, 1]:.2f}**.",
    ])

#################################################################################################
# execução completa


def run_eda(input_folder: Path, output_folder: Path, start_date: str | None, end_date: str | None,
            split_date: str | None, return_type: str) -> None:
    """Carrega os dados, roda as 7 seções em ordem e salva o relatório, os gráficos e as tabelas."""
    fig_dir = output_folder / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # entradas geradas pelo data_mining.py
    prices = pd.read_parquet(input_folder / "raw" / "prices.parquet")  # preços (data x ticker), com ^OEX
    # -rt price usa só o preço (como o ^OEX); -rt total inclui dividendos (Adj Close)
    returns_file = "returns.parquet" if return_type == "price" else "returns_total.parquet"
    returns = pd.read_parquet(input_folder / "processed" / returns_file)  # retornos limpos, com ^OEX
    constituents = pd.read_csv(input_folder / "raw" / "constituents.csv", index_col="yahoo_ticker")
    ibov = pd.read_parquet(input_folder / "processed" / "returns_ibovespa.parquet")[IBOV]
    # período pedido no terminal (sem -s/-e, usa tudo o que foi baixado)
    prices, returns, ibov = (cut(df, start_date, end_date) for df in (prices, returns, ibov))
    # sem data de corte, os primeiros 70% dos pregões são treino
    split = pd.Timestamp(split_date) if split_date else returns.index[int(len(returns) * 0.7)]

    price_note = ("**Preço:** `Close` (ajustado só por splits). O guia sugere `Adj Close`, mas o ^OEX é um "
                  "índice de preço, sem dividendos; com Adj Close as ações ganhariam ~1,5% ao ano que o índice "
                  "não tem. Para comparar, rode com `-rt total`." if return_type == "price" else
                  "**Preço:** `Adj Close` (com dividendos). Atenção: o ^OEX não inclui dividendos.")
    # as seções que também devolvem tabelas rodam antes, porque as tabelas são usadas depois
    stock_text, stock_table = stock_analysis(returns, fig_dir)
    corr_text, corr_table = index_correlation(returns, constituents, fig_dir)
    te_text, te_table = tracking_baseline(returns, split, fig_dir)
    report = [
        "# S&P 100 index tracking: análise exploratória",
        f"**Período:** {returns.index[0].date()} a {returns.index[-1].date()}. "
        f"Gerado com `python3 -m src.eda.run_eda {' '.join(sys.argv[1:])}`.",
        f"Fontes: Yahoo Finance e Wikipedia. {price_note}",
        data_quality(prices, returns),
        index_analysis(prices, returns, fig_dir),
        stock_text,
        corr_text,
        stock_correlation(returns, corr_table.index, fig_dir),  # usa o ranking de correlação da seção 4
        volatility(returns, fig_dir),
        te_text,
        ibovespa_comparison(returns, ibov, fig_dir),
    ]
    (output_folder / "eda_summary.md").write_text("\n\n".join(report))
    corr_table.join(stock_table).to_csv(output_folder / "per_stock_stats.csv")  # uma linha por ação
    te_table.to_csv(output_folder / "tracking_baselines.csv")  # TE de cada carteira de referência
    print(f"report saved to {output_folder / 'eda_summary.md'}")

#################################################################################################
# argumentos do terminal


def get_args_dict() -> dict:
    """Lê os parâmetros do terminal."""
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument('-i', '--input_folder',
                        dest='input_folder',
                        type=Path,
                        required=True,
                        help='defines path to data folder created by data_mining')

    parser.add_argument('-o', '--output_folder',
                        dest='output_folder',
                        type=Path,
                        required=True,
                        help='defines path to report output folder')

    parser.add_argument('-s', '--start_date',
                        dest='start_date',
                        default=None,
                        help='defines first date of the analysis, YYYY-MM-DD (default: start of the data)')

    parser.add_argument('-e', '--end_date',
                        dest='end_date',
                        default=None,
                        help='defines end date of the analysis, YYYY-MM-DD, exclusive (default: end of the data)')

    parser.add_argument('-y', '--year',
                        dest='year',
                        type=int,
                        default=None,
                        help='analyses a single full year, e.g. 2025 (overrides -s/-e)')

    parser.add_argument('-sd', '--split_date',
                        dest='split_date',
                        default=None,
                        help='defines train/test split date, YYYY-MM-DD (default: first 70%% of days are train)')

    parser.add_argument('-rt', '--return_type',
                        dest='return_type',
                        choices=['price', 'total'],
                        default='price',
                        help='defines "price" (Close, like ^OEX) or "total" (Adj Close, with dividends) returns')

    args_dict = vars(parser.parse_args())

    # -y vira o período de 1º de janeiro até 1º de janeiro do ano seguinte
    if args_dict['year']:
        args_dict['start_date'] = f"{args_dict['year']}-01-01"
        args_dict['end_date'] = f"{args_dict['year'] + 1}-01-01"

    return args_dict

#################################################################################################
# função principal


def main():
    """Executa o código principal."""
    args_dict = get_args_dict()
    print_execution_parameters(params_dict=args_dict)
    run_eda(input_folder=args_dict['input_folder'],
            output_folder=args_dict['output_folder'],
            start_date=args_dict['start_date'],
            end_date=args_dict['end_date'],
            split_date=args_dict['split_date'],
            return_type=args_dict['return_type'])


if __name__ == "__main__":
    main()
