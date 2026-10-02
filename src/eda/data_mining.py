"""Downloads os dados do S&P 100 para o projeto de index ‘tracking’.

Expected ‘outputs’ in <output_folder>:
  raw/constituents.csv ⇾ ticker, nome, setor, valor de mercado e peso aproximado no índice
  raw/prices.parquet               fechamento ajustado só por splits (data x ticker), com o índice ^OEX
  raw/prices_total_return.parquet  fechamento ajustado por splits e dividendos (Adj Close)
  raw/volume.parquet               volume negociado
  processed/returns.parquet        retornos diários de preço (mesma base do ^OEX)
  processed/returns_total.parquet  retornos diários totais (com dividendos)
  raw/ibovespa.parquet             fechamento e volume do Ibovespa (^BVSP), no calendário da B3
  processed/returns_ibovespa.parquet  retornos diários do Ibovespa
  raw/ibov_constituents.csv        ticker, nome, setor e peso oficial (B3) das ações do Ibovespa
  raw/ibov_prices.parquet          ações do Ibovespa + ^BVSP: mesmos arquivos e ajustes do S&P 100,
  raw/ibov_prices_total_return.parquet   com prefixo ibov_
  raw/ibov_volume.parquet
  processed/ibov_returns.parquet
  processed/ibov_returns_total.parquet
  ibov_report.md                   o que foi baixado e quais tickers ficaram de fora (e por quê)

Use:
  python3 -m src.eda.data_mining -o data                          # período padrão: 2024-01-01 a 2026-09-30
  python3 -m src.eda.data_mining -o data/2025 -y 2025             # um ano inteiro
  python3 -m src.eda.data_mining -o data/2015 -s 2015-01-01 -e 2026-01-01
"""
#################################################################################################
# imports
import base64
import json
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
import yfinance as yf

from src._execution_formatting import print_execution_parameters

#################################################################################################
# constants

INDEX_TICKER = "^OEX"
IBOV_TICKER = "^BVSP"
FILL_LIMIT = 5  # buracos de até 5 pregões são preenchidos com o último preço (retorno 0 no dia)
WIKI_URL = "https://en.wikipedia.org/wiki/S%26P_100"
# carteira do dia do Ibovespa na B3; o parâmetro é um JSON em base64 (segment "2" = agrupado por setor)
B3_PORTFOLIO_URL = "https://sistemaswebb3-listados.b3.com.br/indexProxy/indexCall/GetPortfolioDay/"
# ex-membros do Ibovespa: entram nos dados com peso 0, para ampliar o universo (ticker: nome, setor, nota).
# Os que não negociam mais vêm vazios do Yahoo e caem no filtro de cobertura; a nota vai para o relatório.
IBOV_EXTRA_TICKERS = {
    "BRKM5": ("BRASKEM", "Materiais Básicos / Químicos", ""),
    "EZTC3": ("EZTEC", "Consumo Cíclico / Construção Civil", ""),
    "RECV3": ("PETRORECSA", "Petróleo, Gás e Biocombustíveis", ""),
    "SMTO3": ("SAO MARTINHO", "Consumo não Cíclico / Alimentos Processados", ""),
    "JBSS32": ("JBS", "Consumo não Cíclico / Alimentos Processados", "BDR que substituiu JBSS3"),
    "ELET3": ("ELETROBRAS", "Utilidade Públ / Energ Elétrica", "renomeada AXIA3, com o histórico"),
    "ELET6": ("ELETROBRAS", "Utilidade Públ / Energ Elétrica", "renomeada AXIA6"),
    "CCRO3": ("CCR", "Bens Indls / Transporte", "renomeada MOTV3, com o histórico"),
    "BRFS3": ("BRF", "Consumo não Cíclico / Alimentos Processados", "incorporada pela Marfrig: MBRF3"),
    "JBSS3": ("JBS", "Consumo não Cíclico / Alimentos Processados", "saiu da B3, ficou o BDR JBSS32"),
    "CRFB3": ("CARREFOUR BR", "Consumo não Cíclico / Comércio e Distribuição", "fechou o capital"),
    "STBP3": ("SANTOS BRP", "Bens Indls / Transporte", "fechou o capital"),
    "CPLE6": ("COPEL", "Utilidade Públ / Energ Elétrica", "PNB convertida em CPLE3"),
}

#################################################################################################
# funções


def get_constituents() -> pd.DataFrame:
    """Tabela de composição do S&P 100 da Wikipedia (ticker, nome, setor)."""
    # sem User-Agent a Wikipedia responde 403; match pega a tabela que tem a coluna "Symbol"
    df = pd.read_html(WIKI_URL, match="Symbol", storage_options={"User-Agent": "Mozilla/5.0"})[0]
    df.columns = ["ticker", "name", "sector"]
    df["yahoo_ticker"] = df["ticker"].str.replace(".", "-")  # BRK.B -> BRK-B
    return df


def get_weights(constituents: pd.DataFrame, last_close: pd.Series) -> pd.DataFrame:
    """Peso aproximado de cada ação no índice, pelo valor de mercado na última data dos dados."""
    info = {t: yf.Ticker(t).fast_info for t in constituents["yahoo_ticker"]}  # uma consulta por ticker
    df = constituents.copy()
    df["market_cap"] = df["yahoo_ticker"].map(lambda t: info[t]["marketCap"])
    # o Yahoo dá o valor da empresa inteira em GOOG e em GOOGL: divide entre as classes
    company = df["name"].str.replace(r"\s*\(Class \w+\)", "", regex=True)
    df["market_cap"] /= company.map(company.value_counts())
    # o valor de mercado é de hoje; traz para a última data dos dados pela variação do preço
    df["market_cap_at_end"] = df["market_cap"] * df["yahoo_ticker"].map(
        lambda t: last_close[t] / info[t]["lastPrice"])
    df["approx_weight"] = df["market_cap_at_end"] / df["market_cap_at_end"].sum()
    return df


def get_ibov_constituents() -> pd.DataFrame:
    """Carteira atual do Ibovespa na B3 (ticker, nome, setor, peso oficial), mais os ex-membros extras."""
    query = {"language": "pt-br", "pageNumber": 1, "pageSize": 200, "index": "IBOV", "segment": "2"}
    url = B3_PORTFOLIO_URL + base64.b64encode(json.dumps(query).encode()).decode()
    with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30) as response:
        rows = json.load(response)["results"]
    df = pd.DataFrame(rows)[["cod", "asset", "segment", "part"]]
    df.columns = ["ticker", "name", "sector", "weight"]
    df["weight"] = df["weight"].str.replace(",", ".").astype(float) / 100  # "9,883" (%) -> 0.09883
    df["in_index"] = True
    df["note"] = ""
    extra = pd.DataFrame([(t, n, s, 0.0, False, note) for t, (n, s, note) in IBOV_EXTRA_TICKERS.items()
                          if t not in set(df["ticker"])], columns=df.columns)
    df = pd.concat([df, extra], ignore_index=True)
    df["yahoo_ticker"] = df["ticker"] + ".SA"
    return df


def coverage(prices: pd.DataFrame, index_ticker: str) -> pd.Series:
    """Fração dos pregões do índice em que cada ação tem preço."""
    return prices[prices[index_ticker].notna()].notna().mean()


def to_returns(prices: pd.DataFrame, min_coverage: float, index_ticker: str = INDEX_TICKER,
               complete_days: bool = True) -> pd.DataFrame:
    """Retornos diários simples, só com ações que têm pelo menos `min_coverage` do histórico.

    complete_days=True deixa só os dias em que todas as ações têm retorno (o EDA do S&P precisa disso);
    False mantém NaN antes da estreia de uma ação, para um IPO tardio não cortar o começo de todas.
    """
    prices = prices[prices[index_ticker].notna()]  # calendário de pregões do índice
    prices = prices.loc[:, coverage(prices, index_ticker) >= min_coverage]  # remove IPOs recentes
    returns = prices.ffill(limit=FILL_LIMIT).pct_change(fill_method=None)  # preenche buracos curtos
    return returns.dropna() if complete_days else returns.iloc[1:]


def mine_data(start_date: str, end_date: str, output_folder: Path, min_coverage: float) -> None:
    """Baixa tudo e salva em `output_folder`."""
    (output_folder / "raw").mkdir(parents=True, exist_ok=True)
    (output_folder / "processed").mkdir(parents=True, exist_ok=True)

    print("getting constituents...")
    constituents = get_constituents()
    tickers = constituents["yahoo_ticker"].tolist()

    print(f"downloading prices for {len(tickers)} tickers...")
    # auto_adjust=False para ter "Close" (só splits, como o ^OEX) e "Adj Close" (com dividendos)
    data = yf.download(tickers + [INDEX_TICKER], start=start_date, end=end_date, auto_adjust=False, progress=False)

    # Ibovespa à parte: tem o calendário da B3, e no mesmo DataFrame viraria mais uma "ação" para o EDA
    print("downloading Ibovespa...")
    ibov = yf.download(IBOV_TICKER, start=start_date, end=end_date, progress=False, multi_level_index=False)
    ibov = ibov[["Close", "Volume"]]  # o Ibovespa já é de retorno total (reinveste dividendos)
    ibov_returns = ibov["Close"].pct_change().dropna().to_frame(IBOV_TICKER)

    print("getting market caps...")
    constituents = get_weights(constituents, data["Close"].ffill().iloc[-1])

    returns = to_returns(data["Close"], min_coverage)
    total_returns = to_returns(data["Adj Close"], min_coverage).reindex_like(returns)
    print(f"removed for short history: {sorted(set(data['Close'].columns) - set(returns.columns))}")

    constituents.to_csv(output_folder / "raw" / "constituents.csv", index=False)
    data["Close"].to_parquet(output_folder / "raw" / "prices.parquet")
    data["Adj Close"].to_parquet(output_folder / "raw" / "prices_total_return.parquet")
    data["Volume"].to_parquet(output_folder / "raw" / "volume.parquet")
    ibov.to_parquet(output_folder / "raw" / "ibovespa.parquet")
    returns.to_parquet(output_folder / "processed" / "returns.parquet")
    total_returns.to_parquet(output_folder / "processed" / "returns_total.parquet")
    ibov_returns.to_parquet(output_folder / "processed" / "returns_ibovespa.parquet")
    print(f"saved {returns.shape[0]} days x {returns.shape[1]} tickers to {output_folder}")

    mine_ibov_data(start_date, end_date, output_folder, min_coverage)


def mine_ibov_data(start_date: str, end_date: str, output_folder: Path, min_coverage: float) -> None:
    """Baixa as ações do Ibovespa e salva em `output_folder` com prefixo ibov_."""
    print("getting Ibovespa constituents...")
    constituents = get_ibov_constituents()
    tickers = constituents["yahoo_ticker"].tolist()

    print(f"downloading prices for {len(tickers)} Ibovespa tickers...")
    data = yf.download(tickers + [IBOV_TICKER], start=start_date, end=end_date, auto_adjust=False, progress=False)

    returns = to_returns(data["Close"], min_coverage, IBOV_TICKER, complete_days=False)
    total_returns = to_returns(data["Adj Close"], min_coverage, IBOV_TICKER, complete_days=False)
    total_returns = total_returns.reindex_like(returns)
    print(f"removed for short history: {sorted(set(data['Close'].columns) - set(returns.columns))}")

    constituents.to_csv(output_folder / "raw" / "ibov_constituents.csv", index=False)
    data["Close"].to_parquet(output_folder / "raw" / "ibov_prices.parquet")
    data["Adj Close"].to_parquet(output_folder / "raw" / "ibov_prices_total_return.parquet")
    data["Volume"].to_parquet(output_folder / "raw" / "ibov_volume.parquet")
    returns.to_parquet(output_folder / "processed" / "ibov_returns.parquet")
    total_returns.to_parquet(output_folder / "processed" / "ibov_returns_total.parquet")
    print(f"saved {returns.shape[0]} days x {returns.shape[1]} Ibovespa tickers to {output_folder}")

    report = ibov_report(constituents, data["Close"], data["Volume"], returns, start_date, end_date, min_coverage)
    (output_folder / "ibov_report.md").write_text(report)
    print(f"report saved to {output_folder / 'ibov_report.md'}")


def treatment_stats(prices: pd.DataFrame, volume: pd.DataFrame, tickers: list[str],
                    index_ticker: str) -> pd.DataFrame:
    """Quanto cada tratamento de `to_returns` mexeu em cada ação mantida (contagem de pregões)."""
    off_calendar = prices.loc[prices[index_ticker].isna(), tickers].notna().sum()
    on_calendar = prices.loc[prices[index_ticker].notna(), tickers]
    started = on_calendar.notna().cummax()  # antes da estreia não é buraco, é a ação não existir
    missing = on_calendar.isna() & started
    filled = missing & on_calendar.ffill(limit=FILL_LIMIT).notna()
    no_trades = on_calendar.notna() & (volume.reindex_like(on_calendar) == 0)
    return pd.DataFrame({
        "off_calendar": off_calendar,     # preços em dias sem pregão do índice: descartados
        "filled": filled.sum(),           # repetem o último preço: retorno 0 no dia
        "left_nan": (missing & ~filled).sum(),  # buracos longos: ficam NaN
        "before_start": (~started).sum(),  # antes da estreia: ficam NaN
        "zero_volume": no_trades.sum(),   # o Yahoo repete o fechamento anterior: retorno 0, sem tratamento nosso
    })


def ibov_report(constituents: pd.DataFrame, prices: pd.DataFrame, volume: pd.DataFrame, returns: pd.DataFrame,
                start_date: str, end_date: str, min_coverage: float) -> str:
    """Relatório em markdown do que foi baixado do Ibovespa e do que foi cortado."""
    df = constituents.set_index("yahoo_ticker")
    df["coverage"] = coverage(prices, IBOV_TICKER).reindex(df.index).fillna(0)
    df["first_date"] = prices.apply(lambda c: c.first_valid_index()).reindex(df.index)
    df["kept"] = df.index.isin(returns.columns)
    df["reason"] = ""
    no_data = df["coverage"] == 0
    df.loc[no_data, "reason"] = "sem dados no Yahoo"
    df.loc[~no_data & ~df["kept"], "reason"] = "histórico curto"
    df["reason"] = df["reason"].where(df["note"] == "", df["reason"] + ": " + df["note"])

    kept, cut = df[df["kept"]], df[~df["kept"]]
    late = kept[kept["coverage"] < 1].sort_values("first_date")
    stats = treatment_stats(prices, volume, kept.index.tolist(), IBOV_TICKER)
    touched = stats[stats.drop(columns="before_start").sum(axis=1) > 0]
    touched = touched.join(df["ticker"]).sort_values(["filled", "left_nan", "zero_volume", "off_calendar"],
                                                     ascending=False)
    index_days = int(prices[IBOV_TICKER].notna().sum())
    n_cells = index_days * len(kept)
    cut_table = cut.assign(
        coverage=cut["coverage"].map("{:.0%}".format),
        weight=cut["weight"].map("{:.2%}".format),
        first_date=cut["first_date"].map(lambda d: d.date() if pd.notna(d) else "-"),
    ).sort_values(["in_index", "coverage"], ascending=[False, False])
    cut_table = cut_table[["ticker", "name", "in_index", "weight", "coverage", "first_date", "reason"]]

    return "\n".join([
        "# Ibovespa: dados baixados",
        "",
        f"- **Período:** {start_date} a {end_date} (fim exclusivo); "
        f"{len(returns)} pregões de {returns.index[0].date()} a {returns.index[-1].date()}.",
        f"- **Fonte da carteira:** carteira do dia da B3 ({int(df['in_index'].sum())} ações, pesos oficiais), "
        f"mais {int((~df['in_index']).sum())} ex-membros com peso 0.",
        f"- **Filtro de cobertura:** a ação precisa ter preço em pelo menos {min_coverage:.0%} dos pregões "
        f"do ^BVSP. Buracos de até 5 pregões são preenchidos com o último preço.",
        f"- **Mantidas:** {len(kept)} de {len(df)} ({int(kept['in_index'].sum())} do índice atual, "
        f"cobrindo {kept['weight'].sum():.1%} do peso do Ibovespa).",
        f"- **Cortadas:** {len(cut)} ({int(cut['in_index'].sum())} do índice atual, "
        f"{cut['weight'].sum():.1%} do peso).",
        "",
        "## Tickers cortados",
        "",
        cut_table.to_markdown(index=False) if len(cut) else "Nenhum.",
        "",
        "## Mantidas com estreia depois do início",
        "",
        "Nos retornos, estas ações ficam com NaN antes da primeira data.",
        "",
        late.assign(coverage=late["coverage"].map("{:.0%}".format), first_date=late["first_date"].dt.date)
        [["ticker", "name", "coverage", "first_date"]].to_markdown(index=False) if len(late) else "Nenhuma.",
        "",
        "## Tratamentos nos dados",
        "",
        "Os arquivos `raw/` guardam os preços como vieram do Yahoo; os tratamentos abaixo valem para "
        "`processed/ibov_returns*.parquet`.",
        "",
        f"1. **Calendário:** só ficam os {index_days} pregões em que o ^BVSP tem preço; "
        f"{int(stats['off_calendar'].sum())} preços de ações em outros dias foram descartados.",
        f"2. **Preenchimento:** buracos de até {FILL_LIMIT} pregões repetem o último preço (retorno 0 no dia): "
        f"{int(stats['filled'].sum())} preços preenchidos de {n_cells} ({stats['filled'].sum() / n_cells:.2%}).",
        f"3. **Buracos longos:** {int(stats['left_nan'].sum())} pregões ficaram NaN por estarem em buracos "
        f"de mais de {FILL_LIMIT} pregões (e o retorno do dia seguinte ao buraco também fica NaN).",
        f"4. **Antes da estreia:** {int(stats['before_start'].sum())} pregões ficam NaN porque a ação "
        "ainda não negociava.",
        "5. **Primeiro pregão:** sai dos retornos, por não ter preço anterior.",
        f"6. **Dias sem negócio:** em {int(stats['zero_volume'].sum())} pregões a ação teve volume 0 e o Yahoo "
        "repetiu o fechamento anterior (retorno 0). É um preenchimento da fonte; os dados ficam como vieram.",
        "7. **Ajustes do Yahoo:** `Close` vem ajustado por desdobramentos e grupamentos; `Adj Close` "
        "(retorno total) também por dividendos e JCP. Não há ajuste nosso.",
        "8. **Pesos:** são os pesos oficiais da B3 na data do download, não no período baixado. "
        "Ex-membros entram com peso 0.",
        "",
        "Ações mantidas com algum preenchimento, buraco longo, dia sem negócio ou preço fora do calendário:",
        "",
        touched[["ticker", "filled", "left_nan", "zero_volume", "off_calendar"]].to_markdown(index=False)
        if len(touched) else "Nenhuma.",
        "",
    ])

#################################################################################################
# argumentos do terminal


def get_args_dict() -> dict:
    """Lê os parâmetros do terminal."""
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument('-o', '--output_folder',
                        dest='output_folder',
                        type=Path,
                        required=True,
                        help='defines path to output folder where data is saved')

    parser.add_argument('-s', '--start_date',
                        dest='start_date',
                        default='2024-01-01',
                        help='defines first date, YYYY-MM-DD (default: 2024-01-01)')

    parser.add_argument('-e', '--end_date',
                        dest='end_date',
                        default='2026-10-01',
                        help='defines end date, YYYY-MM-DD, exclusive (default: 2026-10-01, i.e. through 2026-09-30)')

    parser.add_argument('-y', '--year',
                        dest='year',
                        type=int,
                        default=None,
                        help='downloads a single full year, e.g. 2025 (overrides -s/-e)')

    parser.add_argument('-mc', '--min_coverage',
                        dest='min_coverage',
                        type=float,
                        default=0.80,
                        help='defines minimum fraction of valid prices to keep a ticker (default: 0.80)')

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
    mine_data(start_date=args_dict['start_date'],
              end_date=args_dict['end_date'],
              output_folder=args_dict['output_folder'],
              min_coverage=args_dict['min_coverage'])


if __name__ == "__main__":
    main()
