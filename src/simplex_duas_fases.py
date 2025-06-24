import numpy as np
from simplex_code2 import Simplex

class TwoPhaseSimplex:
    def __init__(self, coeficientes, independentes, f_obj, operadores):
        self.A_original = coeficientes.astype(float)
        self.b = independentes.astype(float)
        self.c_original = f_obj.astype(float)
        self.operadores = operadores

    def _preparar_fase1(self):
        """Prepara e retorna o tableau inicial e canônico para a Fase 1."""
        num_restricoes, num_vars_originais = self.A_original.shape
        
        A_fase1 = self.A_original.copy()
        
        nomes_variaveis = [f"X{i+1}" for i in range(num_vars_originais)]
        base_inicial = [None] * num_restricoes
        indices_artificiais = []

        # Adicionar variáveis de folga, excesso e artificiais
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
                    # Variável de excesso
                    coluna_excesso = np.zeros((num_restricoes, 1))
                    coluna_excesso[i, 0] = -1
                    A_fase1 = np.hstack([A_fase1, coluna_excesso])
                    nomes_variaveis.append(f"E{i+1}")

                # Variável artificial
                coluna_artificial = np.zeros((num_restricoes, 1))
                coluna_artificial[i, 0] = 1
                A_fase1 = np.hstack([A_fase1, coluna_artificial])
                var_name = f"A{i+1}"
                nomes_variaveis.append(var_name)
                base_inicial[i] = var_name
                # Guarda o índice da coluna da var. artificial no A_fase1
                indices_artificiais.append(A_fase1.shape[1] - 1)

        # Montar a função objetivo da Fase 1: Maximizar W' = -Soma(Artificiais)
        f_obj_fase1 = np.zeros(A_fase1.shape[1])
        for idx in indices_artificiais:
            f_obj_fase1[idx] = -1  # Coeficiente -1 para maximizar W'

        # Converter para a linha do tableau Z - cX = 0
        linha_w = -f_obj_fase1
        valor_w = 0.0

        # Tornar o tableau da Fase 1 canônico
        for idx_col_artificial in indices_artificiais:
            linha_pivo = np.where(A_fase1[:, idx_col_artificial] == 1)[0][0]
            fator = linha_w[idx_col_artificial]
            linha_w -= fator * A_fase1[linha_pivo, :]
            valor_w -= fator * self.b[linha_pivo]
        
        linha_w_final = np.append(linha_w, valor_w)

        return A_fase1, linha_w_final, nomes_variaveis, base_inicial, indices_artificiais

    def resolver(self):
        print("\n\033[34mResolvendo com o Método das Duas Fases...\033[0m")
        
        # --- FASE 1 ---
        print("\n\033[36m--- INÍCIO DA FASE 1 ---\033[0m")
        print("Objetivo: Maximizar W' = - (Soma das variáveis artificiais)")
        
        A_f1, linha_w, nomes_vars_f1, base_f1, idx_artificiais = self._preparar_fase1()
        
        # Monta o tableau completo da Fase 1
        num_restricoes, num_total_vars_f1 = A_f1.shape
        matriz_restricoes = np.hstack([A_f1, self.b.reshape(-1, 1)])
        tabela_f1_sem_z = np.vstack([linha_w, matriz_restricoes])
        coluna_z = np.array([-1] + [0] * num_restricoes).reshape(-1, 1)
        tableau_f1 = np.hstack([coluna_z, tabela_f1_sem_z])

        # Instancia e configura o solver para a Fase 1
        solver_f1 = Simplex(self.A_original, self.b, self.c_original, self.operadores)
        solver_f1.tabela = tableau_f1
        solver_f1.nomes_colunas = ["W'"] + nomes_vars_f1 + ["b"]
        solver_f1.nomes_linhas = ["W'"] + base_f1
        solver_f1.variaveis_base = np.array(base_f1)
        solver_f1.num_variaveis = num_total_vars_f1
        solver_f1.num_restricoes = num_restricoes

        # Inicia o processo de solução da Fase 1
        solver_f1.imprime_tab()
        solver_f1.exibe_solucao()

        # Verifica o resultado da Fase 1
        valor_otimo_w = -solver_f1.tabela[0, -1]
        if abs(valor_otimo_w) > 1e-9:
            print("\n\033[31m--- FIM DA FASE 1 ---\033[0m")
            print(f"Resultado da Fase 1: W' ótimo = {valor_otimo_w:.4f} (deveria ser 0).")
            print("\033[31mO problema original é INVIÁVEL.\033[0m")
            return

        print("\n\033[32m--- FIM DA FASE 1 ---\033[0m")
        print("Resultado da Fase 1 é zero. O problema tem solução factível.")

        # --- FASE 2 ---
        print("\n\033[36m--- INÍCIO DA FASE 2 ---\033[0m")
        # Prepara a tabela para a Fase 2
        # Remove colunas das variáveis artificiais
        A_f2 = np.delete(solver_f1.tabela[1:, 1:-1], idx_artificiais, axis=1)
        nomes_vars_f2 = [nome for i, nome in enumerate(nomes_vars_f1) if i not in idx_artificiais]
        base_f2 = solver_f1.variaveis_base

        # Cria a nova linha Z com a função objetivo original
        f_obj_f2 = np.zeros(len(nomes_vars_f2))
        for i in range(len(self.c_original)):
            f_obj_f2[i] = self.c_original[i]
        
        linha_z_f2 = -f_obj_f2
        valor_z_f2 = 0.0

        # Torna o tableau da Fase 2 canônico
        for i, var_base in enumerate(base_f2):
            if var_base in nomes_vars_f2:
                idx_col_base = nomes_vars_f2.index(var_base)
                fator = linha_z_f2[idx_col_base]
                if abs(fator) > 1e-9:
                    linha_z_f2 -= fator * A_f2[i, :]
                    valor_z_f2 -= fator * solver_f1.tabela[i + 1, -1]
        
        linha_z_f2_final = np.append(linha_z_f2, valor_z_f2)

        # Monta o tableau completo da Fase 2
        matriz_restricoes_f2 = np.hstack([A_f2, solver_f1.tabela[1:, -1].reshape(-1, 1)])
        tabela_f2_sem_z = np.vstack([linha_z_f2_final, matriz_restricoes_f2])
        coluna_z = np.array([-1] + [0] * num_restricoes).reshape(-1, 1)
        tableau_f2 = np.hstack([coluna_z, tabela_f2_sem_z])

        # Instancia e configura o solver para a Fase 2
        solver_f2 = Simplex(self.A_original, self.b, self.c_original, self.operadores)
        solver_f2.tabela = tableau_f2
        solver_f2.nomes_colunas = ["Z"] + nomes_vars_f2 + ["b"]
        solver_f2.nomes_linhas = ["Z"] + list(base_f2)
        solver_f2.variaveis_base = base_f2
        solver_f2.num_variaveis = A_f2.shape[1]
        solver_f2.num_restricoes = num_restricoes

        # Inicia o processo de solução da Fase 2
        solver_f2.imprime_tab()
        solver_f2.exibe_solucao()