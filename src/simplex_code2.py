import numpy as np
import pandas as pd


def cria_tabela(coeficientes, independentes, f_obj):
    num_variaveis = coeficientes.shape[1]
    num_restricoes = coeficientes.shape[0]

    # Cria matriz identidade (variáveis de folga)
    identidade = np.eye(num_restricoes)

    # Cria a coluna "Z" com -1 para a função objetivo e 0 para as restrições
    coluna_z = np.array([-1] + [0] * num_restricoes).reshape(-1, 1)

    # Junta coeficientes e folgas
    matriz_restricoes = np.hstack((coeficientes, identidade, independentes.reshape(-1, 1)))
    linha_objetivo = np.hstack((f_obj, np.zeros(num_restricoes), [0]))  # Z = 10x1 + 12x2

    # Junta a linha objetivo com as restrições
    tabela = np.vstack((linha_objetivo, matriz_restricoes))

    # Adiciona a coluna Z (com -1 na função objetivo, 0 nas restrições)
    tabela_com_z = np.hstack((coluna_z, tabela))

    # Nomes das colunas: Z, X1, X2, S1, ..., b
    nomes_variaveis = [f"X{i+1}" for i in range(num_variaveis)]
    nomes_folgas = [f"S{i+1}" for i in range(num_restricoes)]
    nomes_colunas = ["Z"] + nomes_variaveis + nomes_folgas + ["b"]

    # Nomes das linhas: Z, S1, S2, ...
    nomes_linhas = ["Z"] + nomes_folgas

    # Cria DataFrame para visualização
    df = pd.DataFrame(tabela_com_z, columns=nomes_colunas, index=nomes_linhas)

    print(tabela_com_z)
    # Imprime a tabela inicial
    print("\nTabela Inicial do Simplex:\n")
    print(df.round(2))


def main():
    A = np.array([[2, 1],
                  [1, 2]])
    b = np.array([20, 20])
    z = np.array([10, 12])  # Maximizar Z = 10x1 + 12x2

    cria_tabela(A, b, z)


if __name__ == "__main__":
    main()
