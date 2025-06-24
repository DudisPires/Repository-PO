import numpy as np
import pandas as pd

class Simplex:
    def __init__(self, coeficientes, independentes, f_obj, tipo_restricoes):
        self.coeficientes = coeficientes
        self.independentes = independentes
        self.f_obj = f_obj
        self.tipo_restricoes = tipo_restricoes
        self.tabela = None

    def cria_tabela(self):
        print("\n\033[31mComeço do SIMPLEX\033[0m")
        print("\033[31m-------------------------------------\033[0m")
        self.num_variaveis = self.coeficientes.shape[1]
        self.num_restricoes = self.coeficientes.shape[0]

        M = 1e5  # constante grande
        matriz_expandida = []
        self.variaveis_base_inicial = []
        self.nomes_colunas = [f"X{i+1}" for i in range(self.num_variaveis)]
        self.artificiais_idx = []

        func_obj = self.f_obj.tolist()
        artificial_count = 0

        for i in range(self.num_restricoes):
            linha = self.coeficientes[i].tolist()

            if self.tipo_restricoes[i] == '<=':
                slack = [0] * self.num_restricoes
                slack[i] = 1
                linha.extend(slack)
                func_obj.extend([0] * self.num_restricoes)
                self.nomes_colunas.append(f"S{i+1}")
                self.variaveis_base_inicial.append(f"S{i+1}")

            elif self.tipo_restricoes[i] == '>=':
                # adiciona excesso (-1)
                excesso = [0] * self.num_restricoes
                excesso[i] = -1
                linha.extend(excesso)
                func_obj.extend([0] * self.num_restricoes)
                self.nomes_colunas.append(f"E{i+1}")

                # adiciona variável artificial (+1)
                artificiais = [0] * artificial_count + [1]
                linha.extend(artificiais)
                func_obj.extend([-M])
                self.nomes_colunas.append(f"A{i+1}")
                self.variaveis_base_inicial.append(f"A{i+1}")
                artificial_count += 1
                self.artificiais_idx.append(len(func_obj) - 1)

            matriz_expandida.append(linha)

        # Garante que func_obj tenha o mesmo tamanho das restrições
        max_cols = max(len(func_obj), *(len(l) for l in matriz_expandida))
        func_obj += [0] * (max_cols - len(func_obj))
        for linha in matriz_expandida:
            linha += [0] * (max_cols - len(linha))

        # Adiciona coluna b (termo independente)
        for i, linha in enumerate(matriz_expandida):
            linha.append(self.independentes[i])
        func_obj.append(0)
        self.nomes_colunas.append("b")

        # Monta a tabela final
        tabela = np.vstack([func_obj] + matriz_expandida)
        self.nomes_linhas = ["Z"] + self.variaveis_base_inicial
        self.variaveis_base = np.array(self.variaveis_base_inicial)
        df = pd.DataFrame(tabela, columns=self.nomes_colunas, index=self.nomes_linhas)
        print("\n\033[32mTabela do Simplex:\033[0m\n")
        print(df.round(2))

        self.tabela = tabela
        self.exibe_solucao()

    def exibe_solucao(self):
        linha_func_obj = self.tabela[0, :]
        matriz_sol = np.array([linha_func_obj])

        for i in range(1, self.num_restricoes + 1):
            linha_base = self.tabela[i, :]
            matriz_sol = np.vstack((matriz_sol, linha_base))

        valor_z = linha_func_obj[-1]
        print(f"\033[33m\nO valor de Z é = {-1 * valor_z:.3f}\n\033[0m")

        valor_base = []
        for j in range(self.num_restricoes):
            valor_base.append(matriz_sol[j + 1, -1])

        print("Os valores das variaveis na base são: ")
        for valor in range(self.num_restricoes):
            print(f"\033[35m\nVariável {self.variaveis_base[valor]} = {valor_base[valor]:.2f} \033[0m")

        if all(c <= 0 for c in linha_func_obj[1:-1]):
            print("----------------------------------------")
            print("\nFim do SIMPLEX!")
            self.imprime_tab()
        else:
            self.quem_entra_quem_sai()

    def quem_entra_quem_sai(self):
        linha_func_obj = self.tabela[0, :]
        candidatos = []
        variaveis = []
        n_coluna = []

        for i in range(self.num_variaveis + self.num_restricoes):
            if linha_func_obj[i + 1] > 0:
                candidatos.append(linha_func_obj[i + 1])
                variaveis.append(self.nomes_colunas[i + 1])
                n_coluna.append(i + 1)

        maior = max(candidatos)
        posicao = candidatos.index(maior)
        coluna_maior = n_coluna[posicao]

        print("----------------------------------------")
        print("\nAtualizando a tabela do SIMPLEX:\n")
        print(f"O maior valor é: {maior:.2f}, da variável {variaveis[posicao]}, na coluna {coluna_maior}")
        print(f"\033[36m\nPortanto {variaveis[posicao]} entra na base\033[0m")

        coluna_b = self.tabela[:, -1]
        coluna_comp = self.tabela[:, coluna_maior]
        divisao = []

        for b in range(self.num_restricoes):
            try:
                divisao.append(coluna_b[b + 1] / coluna_comp[b + 1])
            except ZeroDivisionError:
                divisao.append(np.inf)

        menor = min([v for v in divisao if v > 0], default=None)
        posicao_div = divisao.index(menor) if menor is not None else 0

        print(f"\nMenor valor: {menor:.2f} da variável {self.variaveis_base[posicao_div]}")
        print(f"\033[36m\nPortanto quem sai será o {self.variaveis_base[posicao_div]} e quem entra será o {variaveis[posicao]} \033[0m")

        self.variaveis_base[posicao_div] = variaveis[posicao]
        print(f"\n ->  Nova BASE: {self.variaveis_base}")
        self.cria_nova_tabela()

    def cria_nova_tabela(self):
        self.nomes_linhas = ["Z"] + self.variaveis_base.tolist()
        df = pd.DataFrame(self.tabela, columns=self.nomes_colunas, index=self.nomes_linhas)
        self.imprime_tab()
        self.escalonar_coluna_por_pivo()

    def imprime_tab(self):
        df = pd.DataFrame(self.tabela, columns=self.nomes_colunas, index=self.nomes_linhas)
        print("\n\033[32mTabela do Simplex:\033[0m\n")
        print(df.round(2))
        print("\n")

    def escalonar_coluna_por_pivo(self):
        base = self.variaveis_base
        posicao = []

        for i in range(len(base)):
            aux = base[i]
            for j in range(len(self.nomes_colunas)):
                if aux == self.nomes_colunas[j]:
                    posicao.append(j)

        for i in range(len(posicao)):
            coluna_pivo = posicao[i]
            linha_pivo = i + 1
            A = self.tabela.astype(float).copy()
            pivo = A[linha_pivo, coluna_pivo]
            if pivo == 0:
                raise ValueError("O pivô é zero. Não é possível escalonar com esse pivô.")
            A[linha_pivo] = A[linha_pivo] / pivo
            for j in range(A.shape[0]):
                if j != linha_pivo:
                    fator = A[j, coluna_pivo]
                    A[j] = A[j] - fator * A[linha_pivo]
            self.tabela = A

        linha_func_obj = self.tabela[0, :]
        if any(linha_func_obj[1:-1] > 0):
            self.exibe_solucao()
        else:
            self.exibe_solucao()
