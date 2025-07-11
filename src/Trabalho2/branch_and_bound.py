# Arquivo: branch_and_bound.py

import numpy as np
import math
import copy
from simplex_duas_fases import TwoPhaseSimplex

class NoB_B:
    """Estrutura de dados para representar um nó da árvore de B&B."""
    def __init__(self, problema, node_id, parent_id, depth, branch_info=""):
        self.problema = problema
        self.id = node_id
        self.parent_id = parent_id
        self.depth = depth
        self.branch_info = branch_info # NOVO: Info sobre o branch (ex: X1 <= 2)

class BranchAndBoundSolver:
    """Classe que orquestra o algoritmo Branch and Bound."""
    def __init__(self, problema_inicial):
        self.problema_inicial = problema_inicial
        self.fila_nos = []
        self.melhor_z = -np.inf
        self.melhor_solucao_inteira = None
        self.next_node_id = 0
        self.arvore_final = [] # NOVO: Lista para registrar a árvore para desenhar depois

    def _get_next_id(self):
        res = self.next_node_id
        self.next_node_id += 1
        return res

    def resolver_pi(self):
        id_raiz = self._get_next_id()
        no_raiz = NoB_B(self.problema_inicial, node_id=id_raiz, parent_id=-1, depth=0)
        self.fila_nos.append(no_raiz)

        while self.fila_nos:
            no_atual = self.fila_nos.pop(0)

            solver = TwoPhaseSimplex(
                no_atual.problema['coeficientes'],
                no_atual.problema['independentes'],
                no_atual.problema['f_obj'],
                no_atual.problema['operadores']
            )
            status, z, solucao = solver.resolver(verbose=False)
            
            node_info = {
                'id': no_atual.id,
                'parent_id': no_atual.parent_id,
                'depth': no_atual.depth,
                'branch_info': no_atual.branch_info
            }

            if status == 'inviavel':
                node_info['label'] = f"Nó {no_atual.id}\nINVIÁVEL\nPodado"
                self.arvore_final.append(node_info)
                continue
            
            node_info['label'] = f"Nó {no_atual.id}\nZ={z:.2f}"
            
            if z <= self.melhor_z:
                node_info['label'] += f"\nPodado\n(pior que Z*={self.melhor_z:.2f})"
                self.arvore_final.append(node_info)
                continue

            var_nome, valor_frac = self._encontrar_variavel_fracionaria(solucao)

            if var_nome is None:
                node_info['label'] += "\nINTEIRO ✔️"
                if z > self.melhor_z:
                    self.melhor_z = z
                    self.melhor_solucao_inteira = solucao
                    node_info['label'] += "\nNOVO Z*!"
                self.arvore_final.append(node_info)
                continue
            else:
                node_info['label'] += f"\nFrac: {var_nome}={valor_frac:.2f}\nRamificando"
                self.arvore_final.append(node_info)
                self._ramificar(no_atual, var_nome, valor_frac)

        # Depois que a fila estiver vazia, desenha a árvore
        self.desenhar_arvore_ascii()
        return self.melhor_z, self.melhor_solucao_inteira

    def _encontrar_variavel_fracionaria(self, solucao):
        for var, valor in solucao.items():
            if abs(valor - round(valor)) > 1e-6:
                return var, valor
        return None, None

    def _ramificar(self, no_pai, var_nome, valor_var):
        problema_pai = no_pai.problema
        indice_var = int(var_nome.replace('X', '')) - 1
        nova_profundidade = no_pai.depth + 1

        # --- Nó da Esquerda (<= floor) ---
        branch_esq_info = f"{var_nome} <= {math.floor(valor_var)}"
        p_esq = copy.deepcopy(problema_pai)
        nova_restricao_esq = np.zeros(p_esq['coeficientes'].shape[1])
        nova_restricao_esq[indice_var] = 1.0
        p_esq['coeficientes'] = np.vstack([p_esq['coeficientes'], nova_restricao_esq])
        p_esq['independentes'] = np.append(p_esq['independentes'], math.floor(valor_var))
        p_esq['operadores'] = np.append(p_esq['operadores'], '<=')
        id_esq = self._get_next_id()
        no_esq = NoB_B(p_esq, node_id=id_esq, parent_id=no_pai.id, depth=nova_profundidade, branch_info=branch_esq_info)
        self.fila_nos.append(no_esq)
        
        # --- Nó da Direita (>= ceil) ---
        branch_dir_info = f"{var_nome} >= {math.ceil(valor_var)}"
        p_dir = copy.deepcopy(problema_pai)
        nova_restricao_dir = np.zeros(p_dir['coeficientes'].shape[1])
        nova_restricao_dir[indice_var] = 1.0
        p_dir['coeficientes'] = np.vstack([p_dir['coeficientes'], nova_restricao_dir])
        p_dir['independentes'] = np.append(p_dir['independentes'], math.ceil(valor_var))
        p_dir['operadores'] = np.append(p_dir['operadores'], '>=')
        id_dir = self._get_next_id()
        no_dir = NoB_B(p_dir, node_id=id_dir, parent_id=no_pai.id, depth=nova_profundidade, branch_info=branch_dir_info)
        self.fila_nos.append(no_dir)

    def desenhar_arvore_ascii(self):
        print("\n\n" + "="*20 + " ÁRVORE DE BRANCH AND BOUND " + "="*20)
        if not self.arvore_final:
            print("Nenhuma árvore para desenhar.")
            return

        nodes_por_id = {node['id']: node for node in self.arvore_final}
        max_depth = max(node['depth'] for node in self.arvore_final)
        
        # Estrutura para guardar posições: {id: (linha, coluna)}
        posicoes = {}
        # Estrutura para guardar nós por nível: {profundidade: [lista_de_nos]}
        niveis = {i: [] for i in range(max_depth + 1)}
        for node in self.arvore_final:
            niveis[node['depth']].append(node)

        # Determinar posições x e y
        largura_total = 0
        for i in range(max_depth + 1):
            largura_nivel = len(niveis[i]) * 25 # Largura de cada "caixa" de nó
            if largura_nivel > largura_total:
                largura_total = largura_nivel
        
        largura_total = max(100, largura_total)
        altura_total = (max_depth + 1) * 8 # Altura de cada nível

        # Cria uma "tela" de caracteres vazia
        tela = [[' ' for _ in range(largura_total)] for _ in range(altura_total)]

        # Posiciona os nós na tela
        for depth, nos_no_nivel in niveis.items():
            y = depth * 8 + 1
            num_nos = len(nos_no_nivel)
            espacamento = largura_total // (num_nos + 1)
            for i, node in enumerate(nos_no_nivel):
                x = espacamento * (i + 1)
                posicoes[node['id']] = (y, x)
                
                # Desenha a caixa e o texto do nó
                texto = node['label'].split('\n')
                box_width = max(len(t) for t in texto) + 2
                for k, line in enumerate(texto):
                    start_x = x - len(line) // 2
                    for char_idx, char in enumerate(line):
                        tela[y + k][start_x + char_idx] = char
                
                # Desenha a caixa
                start_x_box = x - box_width//2 -1
                end_x_box = x + box_width//2 + 1
                for i_y in range(len(texto)+2):
                     tela[y-1+i_y][start_x_box] = '|'
                     tela[y-1+i_y][end_x_box] = '|'
                for i_x in range(start_x_box, end_x_box+1):
                    tela[y-1][i_x] = '-'
                    tela[y+len(texto)][i_x] = '-'


        # Desenha as conexões
        for node_id, pos in posicoes.items():
            if nodes_por_id[node_id]['parent_id'] != -1:
                parent_pos = posicoes[nodes_por_id[node_id]['parent_id']]
                y_pai, x_pai = parent_pos
                label_pai_lines = nodes_por_id[nodes_por_id[node_id]['parent_id']]['label'].split('\n')
                y_pai += len(label_pai_lines) +1
                y_filho, x_filho = pos
                
                # Conexão vertical do pai para baixo
                for y_i in range(y_pai, y_pai + 3):
                    tela[y_i][x_pai] = '|'
                
                # Linha horizontal
                y_horizontal = y_pai + 3
                x_start = min(x_pai, x_filho)
                x_end = max(x_pai, x_filho)
                for x_i in range(x_start, x_end + 1):
                    tela[y_horizontal][x_i] = '-'
                
                # Conexão vertical do filho para cima
                for y_i in range(y_horizontal, y_filho -1):
                    tela[y_i][x_filho] = '|'

                # Adiciona a informação do branch
                branch_info = nodes_por_id[node_id]['branch_info']
                info_y = y_pai + 4
                info_x = (x_pai + x_filho) // 2 - len(branch_info)//2
                for i, char in enumerate(branch_info):
                    tela[info_y][info_x + i] = char


        # Imprime a tela final
        for row in tela:
            print("".join(row))
        print("\n" + "="*20 + " FIM DA ÁRVORE " + "="*20 + "\n")