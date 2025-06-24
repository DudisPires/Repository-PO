import numpy as np
import json
from simplex_duas_fases import TwoPhaseSimplex
from simplex_m_grande import SimplexMGrande


def carregar_problema_json(caminho_arquivo):
    with open(caminho_arquivo, 'r') as f:
        dados = json.load(f)

    coeficientes = np.array(dados['coeficientes'], dtype=float)
    independentes = np.array(dados['independentes'], dtype=float)
    f_obj = np.array(dados['f_obj'], dtype=float)
    operadores = np.array(dados['operadores'])

    return coeficientes, independentes, f_obj, operadores

def main():
    caminho = 'src/problema.json'
    coef, indep, fobj, ops = carregar_problema_json(caminho)
    simplex = SimplexMGrande(coef, indep, fobj, ops)
    solver = TwoPhaseSimplex(coef, indep, fobj, ops)


    while True:
        print("\n ----------------- MENU ---------------------\n")
        print("1- Utilizar o método M Grande\n")
        print("2- Utilizar o método Simplex duas fases\n")
        print("3- Sair\n")
        opcao = input("Escolha uma opção: ")
        print("\n")

        if opcao == '1' :
            simplex.resolver()
        
        if opcao == '2':
            solver.resolver()

        if opcao == '3':
            break

if __name__ == "__main__":
    main()
