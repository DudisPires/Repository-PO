import numpy as np
import pandas as pd

class Simplex:
    def __init__(self, coeficientes, independentes, f_obj):
        self.coeficientes = coeficientes
        self.independentes = independentes
        self.f_obj = f_obj
        self.tabela = None

    def cria_tabela(self):
        print("\n\033[31mComeço do SIMPLEX\033[0m")
        print("\033[31m-------------------------------------\033[0m")
        self.num_variaveis = self.coeficientes.shape[1]
        self.num_restricoes = self.coeficientes.shape[0]

        identidade = np.eye(self.num_restricoes)
        coluna_z = np.array([-1] + [0] * self.num_restricoes).reshape(-1, 1)

        matriz_restricoes = np.hstack((self.coeficientes, identidade, self.independentes.reshape(-1, 1)))
        linha_objetivo = np.hstack((self.f_obj, np.zeros(self.num_restricoes), [0]))

        tabela = np.vstack((linha_objetivo, matriz_restricoes))
        tabela_com_z = np.hstack((coluna_z, tabela))

        nomes_variaveis = [f"X{i+1}" for i in range(self.num_variaveis)]
        self.variaveis_base_inicial = [f"S{i+1}" for i in range(self.num_restricoes)]
        self.nomes_colunas = ["Z"] + nomes_variaveis + self.variaveis_base_inicial + ["b"]
        self.nomes_linhas = ["Z"] + self.variaveis_base_inicial
        #self.variaveis_base= self.variaveis_base
        self.variaveis_base_inicial= np.array((self.variaveis_base_inicial))

        df = pd.DataFrame(tabela_com_z, columns=self.nomes_colunas, index=self.nomes_linhas)
        self.variaveis_base= self.variaveis_base_inicial
        #print(tabela_com_z)
        print("\n\033[32mTabela do Simplex:\033[0m\n")
        print(df.round(2))

        self.tabela = tabela_com_z
        self.exibe_solucao()
        #return tabela_com_z

    def exibe_solucao(self):
        linha_func_obj = self.tabela[0, :]
        matriz_sol = np.array([linha_func_obj])  # inicializa com a linha Z

        for i in range(1, self.num_restricoes + 1):
            linha_base = self.tabela[i, :]
            matriz_sol = np.vstack((matriz_sol, linha_base))

        print("\nLinha da função objetivo:")
        print(f"{linha_func_obj}")
        valor_z= linha_func_obj[self.num_variaveis + self.num_restricoes + 1 ]
        #print(f"\nO valor de Z é = {-1 * valor_z:.3f}\n")
        print(f"\033[33m\nO valor de Z é = {-1 * valor_z:.3f}\n\033[0m")

        valor_base=[]
        for j in range(self.num_restricoes):
            valor_base.append(matriz_sol[ j + 1 , self.num_variaveis + self.num_restricoes + 1])
        
        print("Os valores das variaveis na base são: ")
        for valor in range(self.num_restricoes):
            print(f"\033[35m\nVariável {self.variaveis_base[valor]} = {valor_base[valor]:.2f} \033[0m")

            #print(f"  \nVariável {self.variaveis_base[valor]} = {valor_base[valor]:.2f} ")

        #print("\nTabela de solução:")
        #self.cria_nova_tabela(self.variaveis_base)
        #self.imprime_tab()
        aux=0
        for k in range(len(linha_func_obj)):
            if linha_func_obj[k] > 0:
                aux= aux + 1
        if (aux == 0):
            print("----------------------------------------")
            print("\nFim do SIMPLEX!")
            self.imprime_tab()
        else:
            self.quem_entra_quem_sai()
    
        #self.quem_entra_quem_sai()

    def quem_entra_quem_sai(self):
        
        linha_func_obj = self.tabela[0, :]
        candidatos=[]
        variaveis= []
        n_coluna= []
        valor=0
        for i in range(self.num_variaveis + self.num_restricoes):
            if(linha_func_obj[i+1] > 0):
                candidatos.append (linha_func_obj[i+1])
                variaveis.append (self.nomes_colunas[i+1])
                n_coluna.append (i+1)
                valor= valor +1 

        variaveis= np.array((variaveis))
        candidatos= np.array((candidatos))
        n_coluna= np.array((n_coluna))

        
        #print(f"Variáveis: {variaveis}")
        #print(f"Valores: {candidatos}")
        #print(f"Colunas: {n_coluna}")
        maior= 0
        for n in range(valor):
            if candidatos[n] > maior:
                maior= candidatos[n]
                posicao= n
        coluna_maior= n_coluna[posicao] #indica a coluna do maior coeficiente 
        print("----------------------------------------")
        print("\nAtualizando a tabela do SIMPLEX:\n")
        print(f"O maior valor é: {maior:.2f} , da variavel {variaveis[posicao]}, na coluna {n_coluna[posicao]}")
        #print(f"\nPortanto {variaveis[posicao]} entra na base")
        print(f"\033[36m\nPortanto {variaveis[posicao]} entra na base\033[0m")

        coluna_b= self.tabela[:, self.num_variaveis + self.num_restricoes + 1]
        #print(f"Coluna b: {coluna_b}")
        coluna_comp= self.tabela[:, coluna_maior]
        #print(f"Coluna do {variaveis[posicao]}: {coluna_comp}")
        divisao=[]
        for b in range(self.num_restricoes):
            divisao.append( coluna_b[b+1]/coluna_comp[b+1] )
        divisao= np.array((divisao))

        print(f"\nValores da divisao: {divisao}")
        menor=divisao[0]
        for k in range(self.num_restricoes):
            if divisao[k] < menor :
                menor= divisao[k]
                posicao_div=k
            else:
                posicao_div=0 
        print(f"\nMenor valor:  {menor:.2f} da variavel {self.variaveis_base[posicao_div]} ")
        print(f"\033[36m\nPortanto quem sai sera o {self.variaveis_base[posicao_div]} e quem entra sera o {variaveis[posicao]} \033[0m")

        self.variaveis_base[posicao_div]=variaveis[posicao]
        print(f"\n ->  Nova BASE: {self.variaveis_base}")
        #nova_base= self.variaveis_base
        #return nova_base
        self.cria_nova_tabela()

    def cria_nova_tabela(self):

        self.num_variaveis = self.coeficientes.shape[1]
        self.num_restricoes = self.coeficientes.shape[0]
        #print(self.nomes_colunas)
        #base= np.array((base))
        self.nomes_linhas = ["Z"] + self.variaveis_base.tolist()
        #print(self.nomes_linhas)
        df = pd.DataFrame(self.tabela, columns=self.nomes_colunas, index=self.nomes_linhas)
        self.imprime_tab()
        #print("\nTabela do Simplex nova :\n")
        #print(df.round(2))
        #self.escalona()
        self.escalonar_coluna_por_pivo()

    def imprime_tab(self):
        self.num_variaveis = self.coeficientes.shape[1]
        self.num_restricoes = self.coeficientes.shape[0]
        self.nomes_linhas = ["Z"] + self.variaveis_base.tolist()
        df = pd.DataFrame(self.tabela, columns=self.nomes_colunas, index=self.nomes_linhas)
        print("\n\033[32mTabela do Simplex:\033[0m\n")
        print(df.round(2))
        print("\n")



    def escalonar_coluna_por_pivo(self):

        base= self.variaveis_base
        #print(base)
        #print(range(len(base)))
        posicao= []
        for i in range(len(base)):
            aux=base[i]
            for j in range(len(self.nomes_colunas)):
                if aux == self.nomes_colunas[j]:
                    posicao.append(j)
                    #print(posicao)

        for i in range(len(posicao)):
            coluna_pivo= posicao[i]
            linha_pivo= i +1 
            A = self.tabela.astype(float).copy()
            pivo = A[linha_pivo, coluna_pivo]
            if pivo == 0:
                raise ValueError("O pivô é zero. Não é possível escalonar com esse pivô.")

            A[linha_pivo] = A[linha_pivo] / pivo
        
            for i in range(A.shape[0]):
                if i != linha_pivo:
                    fator = A[i, coluna_pivo]
                    A[i] = A[i] - fator * A[linha_pivo]
            self.tabela = A
            #print(self.tabela)
        
        linha_func_obj = self.tabela[0, :]
        if any(linha_func_obj[1:self.num_variaveis + self.num_restricoes + 1] > 0):
            self.exibe_solucao()
        else:
            #print(self.tabela)
            #print("Fim do SIMPLEX!")
            #self.imprime_tab()
            self.exibe_solucao()


    """
    def escalona(self):
        base= self.variaveis_base
        #print(base)
        #print(range(len(base)))
        posicao= []
        for i in range(len(base)):
            aux=base[i]
            for j in range(len(self.nomes_colunas)):
                if aux == self.nomes_colunas[j]:
                    posicao.append(j)
                    print(posicao)
        for k in range(len(posicao)):
            coluna_escalona= self.tabela[:, posicao[k]]
            #print(coluna_escalona)
            for n in range(len(self.variaveis_base)):
                for j in range(len(coluna_escalona)):
                    if coluna_escalona[j] != 0:
                        coluna_escalona
                    if j == n + 1:
                        if coluna_escalona
                        print("ok")
                    else:
                        print("vish")
    """
