# bootcamp_fundo_amanha_grp4
Repositório destinado ao projeto do Bootcamp do Fundo Amanhã, edição 2026/2, do Grupo 4

Projeto de index tracking: replicar o S&P 100 (`^OEX`) com um subconjunto de suas ações.

## Ambiente

`conda activate bootcamp-fundo-amanha` e `pip install -r requirements.txt`.

Todos os comandos abaixo rodam a partir da raiz do repositório. Use `-h` em qualquer script para ver todos os parâmetros.

## Passo a passo

### Coleta dos dados

Para baixar a composição do índice, os preços e os volumes, rode:

`python3 -m src.eda.data_mining -s <START_DATE> -e <END_DATE> -o <DATA_FOLDER_OUTPUT_PATH>`

O `-o` é obrigatório. Sem `-s`/`-e`, usa o período do briefing (2018-01-01 a 2025-05-01). O `-e` é exclusivo (o dia informado não entra). Para baixar um ano inteiro, use `-y <YEAR>` no lugar de `-s`/`-e`. Ações com menos de 95% de dias com preço são removidas; ajuste com `-mc <MIN_COVERAGE>`.

### Análise exploratória (EDA)

Para gerar o relatório e os gráficos, rode:

`python3 -m src.eda.run_eda -i <DATA_FOLDER_INPUT_PATH> -o <REPORT_FOLDER_OUTPUT_PATH>`

`-i` e `-o` são obrigatórios. Escreve `<REPORT_FOLDER_OUTPUT_PATH>/eda_summary.md`, a pasta `figures/` e as tabelas `per_stock_stats.csv` e `tracking_baselines.csv`. Parâmetros opcionais:

- `-sd <SPLIT_DATE>`: data de corte treino/teste (padrão: os primeiros 70% dos pregões são treino);
- `-rt price|total`: retornos só de preço (padrão; mesma base do `^OEX`, que não inclui dividendos) ou totais (`Adj Close`, com dividendos);
- `-ot <OUTLIER_THRESHOLD>`: retorno diário marcado como extremo (padrão: 0.15);
- `-hl <TICKER> <TICKER> ...`: até 4 ações destacadas no gráfico de performance normalizada.

Exemplo, analisando só 2025 em pastas separadas:

`python3 -m src.eda.data_mining -y 2025 -o data/2025`

`python3 -m src.eda.run_eda -i data/2025 -o reports/2025`
