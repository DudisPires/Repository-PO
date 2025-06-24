import numpy as np
import json
from simplex_duas_fases import TwoPhaseSimplex

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

    solver = TwoPhaseSimplex(coef, indep, fobj, ops)
    solver.resolver()

if __name__ == "__main__":
    main()
