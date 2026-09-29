"""Downloads os dados do S&P 100 para o projeto de index ‘tracking’.

Expected ‘outputs’ in <output_folder>:
  raw/constituents.csv ⇾ ticker, nome, setor, valor de mercado e peso aproximado no índice
  raw/prices.parquet               fechamento ajustado só por splits (data x ticker), com o índice ^OEX
  raw/prices_total_return.parquet  fechamento ajustado por splits e dividendos (Adj Close)
  raw/volume.parquet               volume negociado
  processed/returns.parquet        retornos diários de preço (mesma base do ^OEX)
  processed/returns_total.parquet  retornos diários totais (com dividendos)

Use:
  python3 -m src.eda.data_mining -o data                          # período do briefing: 2018-01-01 a 2025-05-01
  python3 -m src.eda.data_mining -o data/2025 -y 2025             # um ano inteiro
  python3 -m src.eda.data_mining -o data/2015 -s 2015-01-01 -e 2026-01-01
"""
#################################################################################################
# imports
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path

import pandas as pd
import yfinance as yf

from src._execution_formatting import print_execution_parameters

#################################################################################################
# constants

INDEX_TICKER = "^OEX"
WIKI_URL = "https://en.wikipedia.org/wiki/S%26P_100"

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


def to_returns(prices: pd.DataFrame, min_coverage: float) -> pd.DataFrame:
    """Retornos diários simples, só com ações que têm pelo menos `min_coverage` do histórico."""
    prices = prices[prices[INDEX_TICKER].notna()]  # calendário de pregões do índice
    prices = prices.loc[:, prices.notna().mean() >= min_coverage]  # remove IPOs recentes
    return prices.ffill(limit=5).pct_change(fill_method=None).dropna()  # preenche buracos de até 5 dias


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

    print("getting market caps...")
    constituents = get_weights(constituents, data["Close"].ffill().iloc[-1])

    returns = to_returns(data["Close"], min_coverage)
    total_returns = to_returns(data["Adj Close"], min_coverage).reindex_like(returns)
    print(f"removed for short history: {sorted(set(data['Close'].columns) - set(returns.columns))}")

    constituents.to_csv(output_folder / "raw" / "constituents.csv", index=False)
    data["Close"].to_parquet(output_folder / "raw" / "prices.parquet")
    data["Adj Close"].to_parquet(output_folder / "raw" / "prices_total_return.parquet")
    data["Volume"].to_parquet(output_folder / "raw" / "volume.parquet")
    returns.to_parquet(output_folder / "processed" / "returns.parquet")
    total_returns.to_parquet(output_folder / "processed" / "returns_total.parquet")
    print(f"saved {returns.shape[0]} days x {returns.shape[1]} tickers to {output_folder}")

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
                        default='2025-01-01',
                        help='defines first date, YYYY-MM-DD (default: 2018-01-01)')

    parser.add_argument('-e', '--end_date',
                        dest='end_date',
                        default='2026-09-01',
                        help='defines end date, YYYY-MM-DD, exclusive (default: 2025-05-01)')

    parser.add_argument('-y', '--year',
                        dest='year',
                        type=int,
                        default=None,
                        help='downloads a single full year, e.g. 2025 (overrides -s/-e)')

    parser.add_argument('-mc', '--min_coverage',
                        dest='min_coverage',
                        type=float,
                        default=0.95,
                        help='defines minimum fraction of valid prices to keep a ticker (default: 0.95)')

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
