import numpy as np
from simplex_code import Simplex

class TwoPhaseSimplex:
    def __init__(self, coeficientes, independentes, f_obj, operadores):
        self.A_original = coeficientes.astype(float)
        self.b = independentes.astype(float)
        self.c_original = f_obj.astype(float)
        self.operadores = operadores

    def _preparar_fase1(self):
        num_restricoes, num_vars_originais = self.A_original.shape
        
        A_fase1 = self.A_original.copy()
        
        nomes_variaveis = [f"X{i+1}" for i in range(num_vars_originais)]
        base_inicial = [None] * num_restricoes
        indices_artificiais = []

        for i, op in enumerate(self.operadores):
            if op == '<=':
                coluna = np.zeros((num_restricoes, 1))
                coluna[i, 0] = 1
                A_fase1 = np.hstack([A_fase1, coluna])
                var_name = f"S{i+1}"
                nomes_variaveis.append(var_name)
                base_inicial[i] = var_name
            elif op == '>=' or op == '=':
                if op == '>=':
                    coluna_excesso = np.zeros((num_restricoes, 1))
                    coluna_excesso[i, 0] = -1
                    A_fase1 = np.hstack([A_fase1, coluna_excesso])
                    nomes_variaveis.append(f"E{i+1}")

                coluna_artificial = np.zeros((num_restricoes, 1))
                coluna_artificial[i, 0] = 1
                A_fase1 = np.hstack([A_fase1, coluna_artificial])
                var_name = f"A{i+1}"
                nomes_variaveis.append(var_name)
                base_inicial[i] = var_name
                indices_artificiais.append(A_fase1.shape[1] - 1)

        f_obj_fase1 = np.zeros(A_fase1.shape[1])
        for idx in indices_artificiais:
            f_obj_fase1[idx] = -1  
        linha_w = -f_obj_fase1
        valor_w = 0.0

        for idx_col_artificial in indices_artificiais:
            linha_pivo = np.where(A_fase1[:, idx_col_artificial] == 1)[0][0]
            fator = linha_w[idx_col_artificial]
            linha_w -= fator * A_fase1[linha_pivo, :]
            valor_w -= fator * self.b[linha_pivo]
        
        linha_w_final = np.append(linha_w, valor_w)

        return A_fase1, linha_w_final, nomes_variaveis, base_inicial, indices_artificiais

    def resolver(self):
        print("\n\033[34mResolvendo com o Método das Duas Fases...\033[0m")
        
        print("\n\033[36m--- INÍCIO DA FASE 1 ---\033[0m")        
        A_f1, linha_w, nomes_vars_f1, base_f1, idx_artificiais = self._preparar_fase1()
        
        num_restricoes, num_total_vars_f1 = A_f1.shape
        matriz_restricoes = np.hstack([A_f1, self.b.reshape(-1, 1)])
        tabela_f1_sem_z = np.vstack([linha_w, matriz_restricoes])
        coluna_z = np.array([-1] + [0] * num_restricoes).reshape(-1, 1)
        tableau_f1 = np.hstack([coluna_z, tabela_f1_sem_z])

        solver_f1 = Simplex(self.A_original, self.b, self.c_original, self.operadores)
        solver_f1.tabela = tableau_f1
        solver_f1.nomes_colunas = ["W'"] + nomes_vars_f1 + ["b"]
        solver_f1.nomes_linhas = ["W'"] + base_f1
        solver_f1.variaveis_base = np.array(base_f1)
        solver_f1.num_variaveis = num_total_vars_f1
        solver_f1.num_restricoes = num_restricoes

        solver_f1.imprime_tab()
        solver_f1.exibe_solucao()

        valor_otimo_w = -solver_f1.tabela[0, -1]
        if abs(valor_otimo_w) > 1e-9:
            print("\n\033[31m--- FIM DA FASE 1 ---\033[0m")
            print(f"Resultado da Fase 1: W' ótimo = {valor_otimo_w:.4f} (deveria ser 0).")
            print("\033[31mO problema original é INVIÁVEL.\033[0m")
            return

        print("\n\033[32m--- FIM DA FASE 1 ---\033[0m")
        print("Resultado da Fase 1 é zero. O problema tem solução factível.")

        print("\n\033[36m--- INÍCIO DA FASE 2 ---\033[0m")
        A_f2 = np.delete(solver_f1.tabela[1:, 1:-1], idx_artificiais, axis=1)
        nomes_vars_f2 = [nome for i, nome in enumerate(nomes_vars_f1) if i not in idx_artificiais]
        base_f2 = solver_f1.variaveis_base

        f_obj_f2 = np.zeros(len(nomes_vars_f2))
        for i in range(len(self.c_original)):
            f_obj_f2[i] = self.c_original[i]
        
        linha_z_f2 = -f_obj_f2
        valor_z_f2 = 0.0

        for i, var_base in enumerate(base_f2):
            if var_base in nomes_vars_f2:
                idx_col_base = nomes_vars_f2.index(var_base)
                fator = linha_z_f2[idx_col_base]
                if abs(fator) > 1e-9:
                    linha_z_f2 -= fator * A_f2[i, :]
                    valor_z_f2 -= fator * solver_f1.tabela[i + 1, -1]
        
        linha_z_f2_final = np.append(linha_z_f2, valor_z_f2)

        matriz_restricoes_f2 = np.hstack([A_f2, solver_f1.tabela[1:, -1].reshape(-1, 1)])
        tabela_f2_sem_z = np.vstack([linha_z_f2_final, matriz_restricoes_f2])
        coluna_z = np.array([-1] + [0] * num_restricoes).reshape(-1, 1)
        tableau_f2 = np.hstack([coluna_z, tabela_f2_sem_z])

        solver_f2 = Simplex(self.A_original, self.b, self.c_original, self.operadores)
        solver_f2.tabela = tableau_f2
        solver_f2.nomes_colunas = ["Z"] + nomes_vars_f2 + ["b"]
        solver_f2.nomes_linhas = ["Z"] + list(base_f2)
        solver_f2.variaveis_base = base_f2
        solver_f2.num_variaveis = A_f2.shape[1]
        solver_f2.num_restricoes = num_restricoes

        solver_f2.imprime_tab()
        solver_f2.exibe_solucao()