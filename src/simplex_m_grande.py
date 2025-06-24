import numpy as np
import pandas as pd
from simplex_code2 import Simplex

class SimplexMGrande:
    def __init__(self, A, b, c, operadores):
        self.A_original = A.astype(float)
        self.b = b.astype(float)
        self.c_original = c.astype(float)
        self.operadores = operadores
        self.M = 1e6  

    def _preparar_problema(self):
        num_restricoes, num_vars_originais = self.A_original.shape
        
        A_expandida = self.A_original.copy()
        f_obj_expandida = list(self.c_original)
        
        nomes_variaveis = [f"X{i+1}" for i in range(num_vars_originais)]
        base_inicial = [None] * num_restricoes
        indices_artificiais = []

        for i, op in enumerate(self.operadores):
            coluna = np.zeros((num_restricoes, 1))
            if op == '<=':
                coluna[i, 0] = 1
                A_expandida = np.hstack([A_expandida, coluna])
                f_obj_expandida.append(0)
                var_name = f"S{i+1}"
                nomes_variaveis.append(var_name)
                base_inicial[i] = var_name
            elif op == '>=':
                coluna_excesso = np.zeros((num_restricoes, 1))
                coluna_excesso[i, 0] = -1
                A_expandida = np.hstack([A_expandida, coluna_excesso])
                f_obj_expandida.append(0)
                nomes_variaveis.append(f"E{i+1}")

                coluna_artificial = np.zeros((num_restricoes, 1))
                coluna_artificial[i, 0] = 1
                A_expandida = np.hstack([A_expandida, coluna_artificial])
                f_obj_expandida.append(-self.M)  
                var_name = f"A{i+1}"
                nomes_variaveis.append(var_name)
                base_inicial[i] = var_name
                indices_artificiais.append(len(nomes_variaveis) - 1)
            elif op == '=':
                coluna_artificial = np.zeros((num_restricoes, 1))
                coluna_artificial[i, 0] = 1
                A_expandida = np.hstack([A_expandida, coluna_artificial])
                f_obj_expandida.append(-self.M)
                var_name = f"A{i+1}"
                nomes_variaveis.append(var_name)
                base_inicial[i] = var_name
                indices_artificiais.append(len(nomes_variaveis) - 1)

        linha_z = -np.array(f_obj_expandida)
        
        valor_z = 0.0
        for idx_var_artificial in indices_artificiais:
            linha_pivo = np.where(A_expandida[:, idx_var_artificial] == 1)[0][0]
            fator_m = linha_z[idx_var_artificial]
            
            linha_z -= fator_m * A_expandida[linha_pivo, :]
            valor_z -= fator_m * self.b[linha_pivo]
            
        linha_z_final = np.append(linha_z, valor_z)
        
        return A_expandida, linha_z_final, nomes_variaveis, base_inicial

    def resolver(self):
        print("\n\033[36mResolvendo com o Método do M Grande...\033[0m")
        
        A_expandida, linha_z_final, nomes_variaveis, base_inicial = self._preparar_problema()
        
        num_restricoes, num_total_vars = A_expandida.shape

        linha_objetivo = linha_z_final
        matriz_restricoes = np.hstack([A_expandida, self.b.reshape(-1, 1)])
        tabela_sem_coluna_z = np.vstack([linha_objetivo, matriz_restricoes])
        coluna_z = np.array([-1] + [0] * num_restricoes).reshape(-1, 1)
        tableau_completo = np.hstack([coluna_z, tabela_sem_coluna_z])

        solver = Simplex(self.A_original, self.b, self.c_original, self.operadores)

        solver.tabela = tableau_completo
        solver.nomes_colunas = ["Z"] + nomes_variaveis + ["b"]
        solver.nomes_linhas = ["Z"] + base_inicial
        solver.variaveis_base = np.array(base_inicial)
        solver.num_variaveis = num_total_vars
        solver.num_restricoes = num_restricoes
        
        print("\n\033[33mTabela Inicial Pronta para o M-Grande:\033[0m")
        solver.imprime_tab()
        solver.exibe_solucao() 