import numpy as np
import pandas as pd

class Simplex:
    def __init__(self, coeficientes, independentes, f_obj):
        self.coeficientes = coeficientes
        self.independentes = independentes
        self.f_obj = f_obj
        self.tabela = None

    def cria_tabela(self):
        self.num_variaveis = self.coeficientes.shape[1]
        self.num_restricoes = self.coeficientes.shape[0]

        identidade = np.eye(self.num_restricoes)
        coluna_z = np.array([-1] + [0] * self.num_restricoes).reshape(-1, 1)

        matriz_restricoes = np.hstack((self.coeficientes, identidade, self.independentes.reshape(-1, 1)))
        linha_objetivo = np.hstack((self.f_obj, np.zeros(self.num_restricoes), [0]))

        tabela = np.vstack((linha_objetivo, matriz_restricoes))
        tabela_com_z = np.hstack((coluna_z, tabela))

        nomes_variaveis = [f"X{i+1}" for i in range(self.num_variaveis)]
        self.variaveis_base = [f"S{i+1}" for i in range(self.num_restricoes)]
        self.nomes_colunas = ["Z"] + nomes_variaveis + self.variaveis_base + ["b"]
        nomes_linhas = ["Z"] + self.variaveis_base
        #self.variaveis_base= self.variaveis_base
        self.variaveis_base= np.array((self.variaveis_base))

        df = pd.DataFrame(tabela_com_z, columns=self.nomes_colunas, index=nomes_linhas)

        print(tabela_com_z)
        print("\nTabela do Simplex:\n")
        print(df.round(2))

        self.tabela = tabela_com_z
        return tabela_com_z

    def exibe_solucao(self):
        linha_func_obj = self.tabela[0, :]
        matriz_sol = np.array([linha_func_obj])  # inicializa com a linha Z

        for i in range(1, self.num_restricoes + 1):
            linha_base = self.tabela[i, :]
            matriz_sol = np.vstack((matriz_sol, linha_base))

        print("\nLinha da função objetivo:")
        print(linha_func_obj)
        valor_z= linha_func_obj[self.num_variaveis + self.num_restricoes + 1 ]
        print(f"\nO valor de Z é = {valor_z}\n")
        valor_base=[]
        for j in range(self.num_restricoes):
            valor_base.append(matriz_sol[ j + 1 , self.num_variaveis + self.num_restricoes + 1])
        
        print("Os valores das variaveis na base são: ")
        for valor in range(self.num_restricoes):
            print(f"  \nVariável {self.variaveis_base[valor]} = {valor_base[valor]} ")

        print("\nMatriz de solução:")
        print(matriz_sol)

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

        
        print(f"Variáveis: {variaveis}")
        print(f"Valores: {candidatos}")
        print(f"Colunas: {n_coluna}")
        maior= 0
        for n in range(valor):
            if candidatos[n] > maior:
                maior= candidatos[n]
                posicao= n
        coluna_maior= n_coluna[posicao] #indica a coluna do maior coeficiente 
        
        print(f"O maior valor é: {maior} , da variavel {variaveis[posicao]}, na coluna {n_coluna[posicao]}")
        print(f"Portanto {variaveis[posicao]} entra na base")

        coluna_b= self.tabela[:, self.num_variaveis + self.num_restricoes + 1]
        print(f"Coluna b: {coluna_b}")
        coluna_comp= self.tabela[:, coluna_maior]
        print(f"Coluna do {variaveis[posicao]}: {coluna_comp}")
        divisao=[]
        for b in range(self.num_restricoes):
            divisao.append( coluna_b[b+1]/coluna_comp[b+1] )
        divisao= np.array((divisao))

        print(f"Valores da divisao: {divisao}")
        menor=divisao[0]
        for k in range(self.num_restricoes):
            if divisao[k] < menor :
                menor= divisao[k]
                posicao_div=k
        print(f"Menor valor:  {menor} da variavel {self.variaveis_base[posicao_div]} ")
        print(f"Portanto quem sai sera o {self.variaveis_base[posicao_div]} e quem entra sera o {variaveis[posicao]} ")
        self.variaveis_base[posicao_div]=variaveis[posicao]
        print(f"Nova BASE: {self.variaveis_base}")
        self.cria_tabela()
        



                




def main():
    A = np.array([[2, 1],
                  [1, 2]])
    b = np.array([20, 20])
    z = np.array([10, 12])  # Maximizar Z = 10 x1 + 12 x2

    simplex = Simplex(A, b, z)
    simplex.cria_tabela()
    simplex.exibe_solucao()
    simplex.quem_entra_quem_sai()

if __name__ == "__main__":
    main()
