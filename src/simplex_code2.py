import numpy as np
import pandas as pd

class Simplex:
    def __init__(self, coeficientes, independentes, f_obj, operadores):
        self.coeficientes = coeficientes
        self.operadores= operadores
        self.independentes = independentes
        self.f_obj = f_obj
        self.tabela = None
        self.nomes_colunas = []
        self.nomes_linhas = []
        self.variaveis_base = np.array([])
        self.num_variaveis = 0
        self.num_restricoes = 0


    def cria_tabela(self):
        # Esta função é mais adequada para problemas padrão (não Big-M)
        print("\n\033[31mComeço do SIMPLEX (Método Padrão)\033[0m")
        # ... (o resto desta função pode permanecer como está, pois o M-Grande não a utiliza)
        self.num_variaveis = self.coeficientes.shape[1]
        self.num_restricoes = self.coeficientes.shape[0]

        identidade = np.eye(self.num_restricoes)
        coluna_z = np.array([-1] + [0] * self.num_restricoes).reshape(-1, 1)

        matriz_restricoes = np.hstack((self.coeficientes, identidade, self.independentes.reshape(-1, 1)))
        linha_objetivo = np.hstack((self.f_obj, np.zeros(self.num_restricoes), [0]))

        tabela = np.vstack((linha_objetivo, matriz_restricoes))
        tabela_com_z = np.hstack((coluna_z, tabela))

        nomes_variaveis = [f"X{i+1}" for i in range(self.num_variaveis)]
        self.variaveis_base = np.array([f"S{i+1}" for i in range(self.num_restricoes)])
        self.nomes_colunas = ["Z"] + nomes_variaveis + list(self.variaveis_base) + ["b"]
        self.nomes_linhas = ["Z"] + list(self.variaveis_base)

        df = pd.DataFrame(tabela_com_z, columns=self.nomes_colunas, index=self.nomes_linhas)
        self.tabela = tabela_com_z
        
        print("\n\033[32mTabela do Simplex:\033[0m\n")
        print(df.round(2))

        self.exibe_solucao()


    def exibe_solucao(self):
        indice_coluna_b = self.tabela.shape[1] - 1
        linha_func_obj = self.tabela[0, :]
        
        valor_z = linha_func_obj[indice_coluna_b]
        print(f"\033[33m\nO valor de Z é = {-1 * valor_z:.3f}\n\033[0m")

        print("Os valores das variaveis na base são: ")
        for j in range(self.num_restricoes):
            valor_base = self.tabela[j + 1, indice_coluna_b]
            print(f"\033[35m\nVariável {self.variaveis_base[j]} = {valor_base:.2f} \033[0m")

        # --- LÓGICA DE PARADA CORRIGIDA ---
        # Para o método padrão (indicadores negativos), a solução é ótima quando
        # não há mais valores negativos na linha Z.
        if any(linha_func_obj[1:-1] < -1e-9): # Procura por negativos
            self.quem_entra_quem_sai()
        else:
            print("----------------------------------------")
            print("\nFim do SIMPLEX! Solução ótima encontrada.")
            self.exibe_solucao_final()


    def quem_entra_quem_sai(self):
        linha_func_obj = self.tabela[0, :]
        
        # --- LÓGICA DE PIVOTEAMENTO CORRIGIDA ---
        # Para maximização, a variável que entra é a que tem o coeficiente
        # mais negativo na linha Z.
        menor_valor_z = np.min(linha_func_obj[1:-1])
        
        if menor_valor_z >= -1e-9:
             print("\n\033[32mNão há mais valores negativos na linha Z. Solução ótima encontrada.\033[0m")
             self.exibe_solucao_final()
             return

        # +1 para compensar a coluna Z que foi ignorada
        coluna_pivo_idx = np.argmin(linha_func_obj[1:-1]) + 1
        variavel_entra = self.nomes_colunas[coluna_pivo_idx]

        print("----------------------------------------")
        print("\nAtualizando a tabela do SIMPLEX:\n")
        print(f"O menor valor (mais negativo) é: {menor_valor_z:.2f}, da variavel {variavel_entra}")
        print(f"\033[36m\nPortanto {variavel_entra} entra na base\033[0m")

        coluna_b = self.tabela[1:, -1]
        coluna_pivo = self.tabela[1:, coluna_pivo_idx]

        divisao = []
        for i in range(len(coluna_pivo)):
            if coluna_pivo[i] > 1e-9:
                divisao.append(coluna_b[i] / coluna_pivo[i])
            else:
                divisao.append(float('inf'))

        if all(d == float('inf') for d in divisao):
            print("\n\033[31mNão existe pivô válido (todos os coeficientes na coluna pivô são <= 0).\033[0m")
            print("\033[31mO problema possui SOLUÇÃO ILIMITADA.\033[0m")
            return

        linha_pivo_idx = np.argmin(divisao)
        variavel_sai = self.variaveis_base[linha_pivo_idx]
        
        print(f"\nMenor valor da razão: {min(divisao):.2f}, da variável {variavel_sai}")
        print(f"\033[36m\nPortanto quem sai é {variavel_sai}\033[0m")

        self.variaveis_base[linha_pivo_idx] = variavel_entra
        print(f"\n ->  NOVA BASE: {self.variaveis_base}")

        self.escalona_tabela(linha_pivo_idx + 1, coluna_pivo_idx)


    def escalona_tabela(self, linha_pivo_idx, coluna_pivo_idx):
        pivo = self.tabela[linha_pivo_idx, coluna_pivo_idx]
        
        self.tabela[linha_pivo_idx, :] /= pivo
        
        for i in range(self.tabela.shape[0]):
            if i != linha_pivo_idx:
                fator = self.tabela[i, coluna_pivo_idx]
                self.tabela[i, :] -= fator * self.tabela[linha_pivo_idx, :]
        
        self.imprime_tab()
        self.exibe_solucao()


    def imprime_tab(self):
        self.nomes_linhas = ["Z"] + list(self.variaveis_base)
        df = pd.DataFrame(self.tabela, columns=self.nomes_colunas, index=self.nomes_linhas)
        print("\n\033[32mTabela do Simplex:\033[0m\n")
        print(df.round(2))
        print("\n")

    def exibe_solucao_final(self):
        self.imprime_tab()
        indice_coluna_b = self.tabela.shape[1] - 1
        valor_z = self.tabela[0, indice_coluna_b]
        print(f"\033[33m\nValor Ótimo de Z é = {valor_z:.3f}\n\033[0m")

        print("Valores das variáveis na solução ótima: ")
        for i in range(len(self.variaveis_base)):
            valor_base = self.tabela[i + 1, indice_coluna_b]
            print(f"\033[35m\nVariável {self.variaveis_base[i]} = {valor_base:.2f} \033[0m")