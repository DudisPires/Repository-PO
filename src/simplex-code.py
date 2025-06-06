import numpy as np



def simplex_solver(coeficientes, independentes, f_obj):
    
    num_coeficientes= coeficientes.shape[1]
    num_restricoes= coeficientes.shape[0]

    identidade = np.eye(num_restricoes)  #cria uma matriz identidade com de acordo com as variaveis de folga
    matriz_principal = np.hstack((coeficientes, identidade))


    #matriz_principal= np.hstack((matriz_principal,independentes)) #coluna dos independentes

    #coluna_base=['-','Z']
    #for i in range(num_restricoes):
    #    coluna_base.append(f'X')   

    #print(coluna_base)

    #coluna_z= ["Z", -1]

    #for i in range(num_restricoes):
    #    coluna_z.append(0)   

    

    #print(coluna_z)

    #coluna_z= np.array((['Z', 0]))
    #matriz_principal= np.hstack((matriz_principal,coluna_z)) #coluna do Z


    variaveis = [f'-', f'Z']

    for i in range(num_coeficientes + num_restricoes):
        variaveis.append(f'X{i+1} ')  # Usa f-string para criar 'X1', 'X2', ...  
    variaveis.append(f'b')
    variaveis_array= np.array((variaveis))
    print(variaveis)

    
    #matriz_principal = np.vstack((variaveis_array, matriz_principal))
    #matriz_completa=  = np.vstack((variaveis_array, matriz_principal))
    
    

    #matriz_principal= np.vstack((teste , matriz_principal))
    print(matriz_principal)


def main():

    A = np.array([[2, 1], 
                  [1, 2]])
    
    b = np.array([20, 20])
    z = np.array([10, 12])

    simplex_solver(A, b, z)

if __name__ == "__main__":
    main()