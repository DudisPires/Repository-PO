import numpy as np
import pandas as pd

class Simplex:
    def __init__(self, coeficientes, independentes, f_obj, tipos_restricoes):
        self.coeficientes = coeficientes
        self.independentes = independentes
        self.f_obj = f_obj
        self.tipos_restricoes = tipos_restricoes
        self.M = 1e5  # Constante grande para penalidade de variáveis artificiais
        self.tabela = None

    def cria_tabela(self):
        print("\n\033[31mComeço do SIMPLEX (M Grande)\033[0m")
        print("\033[31m-------------------------------------\033[0m")

        self.num_variaveis = self.coeficientes.shape[1]
        self.num_restricoes = self.coeficientes.shape[0]

        colunas_extra = []
        linha_objetivo = list(self.f_obj.copy())
        matriz_restricoes = []
        self.variaveis_base = []
        self.var_artificiais = []

        for i in range(self.num_restricoes):
            tipo = self.tipos_restricoes[i]
            linha = list(self.coeficientes[i])
            
            # Adiciona variáveis de folga/excesso
            for j in range(self.num_restricoes):
                if i == j:
                    if tipo == '<=':
                        linha.append(1)
                        linha_objetivo.append(0)
                        colunas_extra.append(f'S{i+1}')
                        self.variaveis_base.append(f'S{i+1}')
                    elif tipo == '>=':
                        linha.append(-1)
                        linha_objetivo.append(0)
                        colunas_extra.append(f'S{i+1}')
                else:
                    linha.append(0)
                    linha_objetivo.append(0)
                    if j >= len(colunas_extra):
                        colunas_extra.append(f'S{j+1}')
            
            # Se for >=, adiciona variável artificial
            if tipo == '>=':
                linha.append(1)
                linha_objetivo.append(self.M)
                colunas_extra.append(f'A{i+1}')
                self.variaveis_base.append(f'A{i+1}')
                self.var_artificiais.append(f'A{i+1}')

            matriz_restricoes.append(linha + [self.independentes[i]])

        linha_objetivo.append(0)  # coluna b

        tabela = np.vstack((linha_objetivo, matriz_restricoes))
        coluna_z = np.array([-1] + [0] * (tabela.shape[1] - 1)).reshape(-1, 1)
        self.tabela = np.hstack((coluna_z, tabela))

        nomes_variaveis = [f"X{i+1}" for i in range(self.num_variaveis)]
        self.nomes_colunas = ["Z"] + nomes_variaveis + colunas_extra + ["b"]
        self.nomes_linhas = ["Z"] + self.variaveis_base

        df = pd.DataFrame(self.tabela, columns=self.nomes_colunas, index=self.nomes_linhas)
        print("\n\033[32mTabela Inicial do Simplex:\033[0m\n")
        print(df.round(2))

        self.variaveis_base = np.array(self.variaveis_base)
        self.exibe_solucao()

    def exibe_solucao(self):
        linha_func_obj = self.tabela[0, :]
        valor_z = linha_func_obj[-1]
        print(f"\033[33m\nValor de Z: {-1 * valor_z:.3f}\n\033[0m")

        print("Valores das variáveis na base:")
        for i, var in enumerate(self.variaveis_base):
            valor = self.tabela[i + 1, -1]
            print(f"\033[35m{var} = {valor:.2f}\033[0m")

        if all(linha_func_obj[1:-1] <= 0):
            print("\n\033[32mFim do SIMPLEX!\033[0m")
            self.imprime_tab()
        else:
            self.quem_entra_quem_sai()

    def quem_entra_quem_sai(self):
        linha_func_obj = self.tabela[0, :]
        candidatos = linha_func_obj[1:-1]

        col_index = np.argmax(candidatos)
        if candidatos[col_index] <= 0:
            print("\n\033[32mNenhuma variável positiva para entrar na base\033[0m")
            return

        coluna_entrada = col_index + 1
        nome_entrada = self.nomes_colunas[coluna_entrada]

        print(f"\n\033[36m{nome_entrada} entra na base\033[0m")

        col_b = self.tabela[1:, -1]
        col_pivo = self.tabela[1:, coluna_entrada]

        razoes = []
        for i in range(len(col_b)):
            if col_pivo[i] > 0:
                razoes.append(col_b[i] / col_pivo[i])
            else:
                razoes.append(np.inf)

        linha_saida_index = np.argmin(razoes)
        nome_saida = self.variaveis_base[linha_saida_index]

        print(f"\033[36m{nome_saida} sai da base\033[0m")

        self.variaveis_base[linha_saida_index] = nome_entrada
        self.escalonar(coluna_entrada, linha_saida_index + 1)

    def escalonar(self, col_pivo, linha_pivo):
        A = self.tabela.copy().astype(float)
        pivo = A[linha_pivo, col_pivo]
        A[linha_pivo] = A[linha_pivo] / pivo

        for i in range(A.shape[0]):
            if i != linha_pivo:
                A[i] = A[i] - A[i, col_pivo] * A[linha_pivo]

        self.tabela = A
        self.imprime_tab()
        self.exibe_solucao()

    def imprime_tab(self):
        df = pd.DataFrame(self.tabela, columns=self.nomes_colunas, index=["Z"] + list(self.variaveis_base))
        print("\n\033[34mTabela Atual:\033[0m")
        print(df.round(2))
