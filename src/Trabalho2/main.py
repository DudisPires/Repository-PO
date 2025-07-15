import numpy as np
import json
from simplex_duas_fases import TwoPhaseSimplex
from simplex_m_grande import SimplexMGrande
from branch_and_bound import BranchAndBoundSolver



def carregar_problema_json(caminho_arquivo):
    with open(caminho_arquivo, 'r') as f:
        dados = json.load(f)

    coeficientes = np.array(dados['coeficientes'], dtype=float)
    independentes = np.array(dados['independentes'], dtype=float)
    f_obj = np.array(dados['f_obj'], dtype=float)
    operadores = np.array(dados['operadores'])

    return coeficientes, independentes, f_obj, operadores

def main():
    caminho = 'problema2.json'
    coef, indep, fobj, ops = carregar_problema_json(caminho)

    while True:
        print("\n ----------------- MENU ---------------------\n")
        print("1- Resolver PL com Simplex M-Grande\n")
        print("2- Resolver PL com Simplex Duas Fases\n")
        print("3- Resolver PI com Branch and Bound\n") 
        print("4- Sair\n")
        opcao = input("Escolha uma opção: ")
        print("\n")

        if opcao == '1':
            print("Resolvendo com M-Grande...")
            simplex_m = SimplexMGrande(coef, indep, fobj, ops)
            simplex_m.resolver() 

        if opcao == '2':
            print("Resolvendo com Duas Fases (modo verboso)...")
            solver_2p = TwoPhaseSimplex(coef, indep, fobj, ops)
            solver_2p.resolver(verbose=True)

        if opcao == '3':
            print("Iniciando resolvedor de Programação Inteira com Branch and Bound...")
            problema_inicial_pi = {
                'coeficientes': coef,
                'independentes': indep,
                'f_obj': fobj,
                'operadores': ops
            }
            bb_solver = BranchAndBoundSolver(problema_inicial_pi)
            z_otimo, solucao_otima = bb_solver.resolver_pi()

            if solucao_otima:
                print("\n\033[32m--- SOLUÇÃO ÓTIMA INTEIRA ENCONTRADA ---\033[0m")
                print(f"Valor Ótimo de Z = {z_otimo:.3f}")
                print(f"Solução: {solucao_otima}")
            else:
                print("\n\033[31mNão foi encontrada solução inteira para o problema.\033[0m")

        if opcao == '4':
            break

if __name__ == "__main__":
    main()
