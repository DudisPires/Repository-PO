import numpy as np
from simplex_m_grande import SimplexMGrande

def main():
    A = np.array([
        [1, 1],
        [2, 1],
        [1, 0]
    ])

    b = np.array([4, 5, 2])
    c = np.array([3, 2])
    operadores = [">=", "<=", "<="]

    simplex = SimplexMGrande(A, b, c, operadores)
    simplex.resolver()

if __name__ == "__main__":
    main()
