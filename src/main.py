#from simplex_code2 import Simplex
from teste import Simplex
import numpy as np

def main():
    A = np.array([[2, 1],
                  [1, 2]])
    operadores= np.array((['<=','>=']))
    b = np.array([20, 20])
    z = np.array([10, 12])  # Maximizar Z = 10 x1 + 12 x2

    simplex = Simplex(A, b, z, operadores)
    #simplex.m_grande()
    simplex.cria_tabela()
    #simplex.exibe_solucao()
    #simplex.quem_entra_quem_sai()
    #simplex.cria_nova_tabela()


if __name__ == "__main__":
    main()
