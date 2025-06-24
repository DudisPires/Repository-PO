import numpy as np
import pandas as pd
from simplex_code2 import Simplex

class TwoPhaseSimplex:
    def __init__(self, coeficientes, independentes, f_obj, operadores):
        self.coeficientes = coeficientes
        self.independentes = independentes
        self.f_obj_original = f_obj
        self.operadores = operadores

    def resolver(self):
        if any(op == '>=' or op == '=' for op in self.operadores):
            print("\033[34mIniciando o Método das Duas Fases\033[0m")
            A_f1, b_f1, fase1_obj, base_f1, artificiais_idx = self._montar_fase1()

            simplex_fase1 = Simplex(A_f1, b_f1, fase1_obj, np.array(["<="] * len(b_f1)))
            simplex_fase1.cria_tabela()

            valor_fase1 = simplex_fase1.tabela[0, -1]
            if round(valor_fase1, 6) > 0:
                print("\033[31mO problema é inviável. A Fase 1 retornou valor positivo na F.O.\033[0m")
                return

            print("\033[34m\nFim da Fase 1. Iniciando Fase 2...\033[0m")
            A_f2, b_f2 = self._remover_artificiais(simplex_fase1.tabela, artificiais_idx)

            # Expande o vetor da função objetivo para o tamanho da fase 2
            tamanho_total = A_f2.shape[1]
            f_obj_exp = self._expandir_funcao_objetivo(self.f_obj_original, tamanho_total)

            simplex_fase2 = Simplex(A_f2, b_f2, f_obj_exp, np.array(["<="] * len(b_f2)))
            simplex_fase2.cria_tabela()
        else:
            print("\033[34mResolvendo com Simplex Padrão (Fase Única)\033[0m")
            simplex = Simplex(self.coeficientes, self.independentes, self.f_obj_original, self.operadores)
            simplex.cria_tabela()

    def _montar_fase1(self):
        num_rest = self.coeficientes.shape[0]
        num_vars = self.coeficientes.shape[1]

        A_f1 = []
        artificiais_idx = []
        base = []

        for i in range(num_rest):
            linha = list(self.coeficientes[i])

            if self.operadores[i] == "<=" or self.operadores[i] == "=":
                slack = [0] * num_rest
                slack[i] = 1
                linha += slack
                if self.operadores[i] == "=":
                    artificiais = [0] * num_rest
                    artificiais[i] = 1
                    linha += artificiais
                    artificiais_idx.append(num_vars + num_rest + i)
                    base.append(f"A{i+1}")
                else:
                    artificiais = [0] * num_rest
                    linha += artificiais
                    base.append(f"S{i+1}")

            elif self.operadores[i] == ">=":
                slack = [0] * num_rest
                slack[i] = -1  # subtrai excedente
                linha += slack
                artificiais = [0] * num_rest
                artificiais[i] = 1
                linha += artificiais
                artificiais_idx.append(num_vars + num_rest + i)
                base.append(f"A{i+1}")

            A_f1.append(linha)

        A_f1 = np.array(A_f1)
        b_f1 = self.independentes

        fase1_obj = np.array([0] * A_f1.shape[1])
        for idx in artificiais_idx:
            fase1_obj[idx] = 1

        return A_f1, b_f1, fase1_obj, base, artificiais_idx

    def _remover_artificiais(self, tabela, artificiais_idx):
        tabela_sem_z = tabela[1:, :]  # ignora linha Z
        # Ajusta os índices +1 para coluna da tabela (porque primeira coluna é variável base)
        tabela_sem_z = np.delete(tabela_sem_z, [idx + 1 for idx in artificiais_idx], axis=1)

        A_f2 = tabela_sem_z[:, 1:-1]  # remove coluna variável base e coluna b
        b_f2 = tabela_sem_z[:, -1]   # ultima coluna é b
        return A_f2, b_f2

    def _expandir_funcao_objetivo(self, f_obj, tamanho_total):
        """Expande o vetor da função objetivo para tamanho_total adicionando zeros no final."""
        f_obj_exp = np.zeros(tamanho_total)
        tamanho_f_obj = len(f_obj)
        f_obj_exp[:tamanho_f_obj] = f_obj
        return f_obj_exp
