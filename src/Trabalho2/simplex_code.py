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
        self.variaveis_base = np.array([])

    def resolver(self, verbose=True):
        """
        Loop principal do Simplex que itera até encontrar a solução ótima.
        """
        if verbose:
            self.imprime_tab("Tabela Inicial")

        while np.min(self.tabela[0, 1:-1]) < -1e-9:
            status, linha_pivo_idx, coluna_pivo_idx = self.encontrar_pivo()

            if status == 'ilimitada':
                if verbose:
                    print("\n\033[31mProblema com Solução Ilimitada.\033[0m")
                return 'ilimitada'
            
            if status == 'otima':
                break

            if verbose:
                variavel_entra = self.nomes_colunas[coluna_pivo_idx]
                variavel_sai = self.variaveis_base[linha_pivo_idx]
                print("----------------------------------------")
                print(f"-> Entra na base: {variavel_entra}, Sai da base: {variavel_sai}")

            self.escalona_tabela(linha_pivo_idx + 1, coluna_pivo_idx)
            self.variaveis_base[linha_pivo_idx] = self.nomes_colunas[coluna_pivo_idx]

            if verbose:
                self.imprime_tab("Tabela Atualizada")
        
        if verbose:
            print("\n\033[32mSolução ótima encontrada!\033[0m")
            self.imprime_tab("Tabela Final")
            self._exibe_solucao_final_verbose()
        
        return 'otima'

    def encontrar_pivo(self):
        linha_func_obj = self.tabela[0, 1:-1]
        
        if np.min(linha_func_obj) >= -1e-9:
            return 'otima', -1, -1

        coluna_pivo_idx = np.argmin(linha_func_obj) + 1
        coluna_pivo = self.tabela[1:, coluna_pivo_idx]

        if np.all(coluna_pivo <= 1e-9):
            return 'ilimitada', -1, -1

        coluna_b = self.tabela[1:, -1]
        ratios = np.full_like(coluna_b, float('inf'))
        
        mascara_positiva = coluna_pivo > 1e-9
        ratios[mascara_positiva] = coluna_b[mascara_positiva] / coluna_pivo[mascara_positiva]

        linha_pivo_idx = np.argmin(ratios)
        
        return 'continua', linha_pivo_idx, coluna_pivo_idx

    def escalona_tabela(self, linha_pivo_idx, coluna_pivo_idx):
        pivo = self.tabela[linha_pivo_idx, coluna_pivo_idx]
        self.tabela[linha_pivo_idx, :] /= pivo
        
        for i in range(self.tabela.shape[0]):
            if i != linha_pivo_idx:
                fator = self.tabela[i, coluna_pivo_idx]
                self.tabela[i, :] -= fator * self.tabela[linha_pivo_idx, :]

    def get_solucao(self):
        """
        Extrai e retorna a solução final do tableau de forma estruturada.
        """
        status_final = 'otima'
        if np.any(np.isinf(self.tabela)):
             status_final = 'ilimitada'

        # MODIFICAÇÃO AQUI: convertendo Z para float padrão
        valor_z = float(self.tabela[0, -1])
        
        nomes_vars_originais = [nome for nome in self.nomes_colunas if nome.startswith('X')]
        solucao_dict = {var: 0.0 for var in nomes_vars_originais}

        for i, var_base in enumerate(self.variaveis_base):
            if var_base in solucao_dict:
                # MODIFICAÇÃO PRINCIPAL AQUI: convertendo o valor da variável para float padrão
                solucao_dict[var_base] = float(self.tabela[i + 1, -1])
        
        return status_final, valor_z, solucao_dict
    
    def imprime_tab(self, titulo="Tabela do Simplex"):
        nomes_linhas = ["Z"] + list(self.variaveis_base)
        df = pd.DataFrame(self.tabela, columns=self.nomes_colunas, index=nomes_linhas)
        print(f"\n\033[32m{titulo}:\033[0m\n")
        print(df.round(3))

    def _exibe_solucao_final_verbose(self):
        status, valor_z, solucao = self.get_solucao()
        print(f"\033[33m\nValor Ótimo de Z é = {valor_z:.3f}\n\033[0m")
        print("Valores das variáveis na solução ótima: ")
        for var, val in solucao.items():
            if val > 1e-9:
                print(f"\033[35mVariável {var} = {val:.3f}\033[0m")