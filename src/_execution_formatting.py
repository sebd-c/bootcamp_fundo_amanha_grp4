# módulo de funções auxiliares de formatação da execução no terminal

######################################################################
# funções


def spacer(char: str = '_',
           reps: int = 50
           ) -> None:
    """
    Imprime uma linha separadora formada por `char` repetido `reps` vezes.
    """
    # montando e imprimindo a linha separadora
    print(char * reps)


def print_execution_parameters(params_dict: dict) -> None:
    """
    Imprime os parâmetros de execução no terminal, no formato:
    '''
    --Execution parameters--
    start_date: 2015-01-01
    output_folder: data
    '''
    """
    # cabeçalho da lista de parâmetros
    params_string = '--Execution parameters--'

    # adicionando uma linha "nome: valor" para cada parâmetro
    for param_key, param_value in params_dict.items():
        params_string += f'\n{param_key}: {param_value}'

    # imprimindo os parâmetros entre duas linhas separadoras
    spacer()
    print(params_string)
    spacer()


def enter_to_continue() -> None:
    """
    Pausa a execução até o usuário pressionar "Enter".
    """
    # esperando o "Enter" do usuário
    input('press "Enter" to continue')

######################################################################
# fim do módulo
