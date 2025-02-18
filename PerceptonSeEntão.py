def perceptron_implication(x1, x2, w1=-1, w2=1, theta=0):
    """
    Implementa um perceptron para a função lógica de implicação.

    Args:
      x1: Primeira entrada.
      x2: Segunda entrada.
      w1: Peso sináptico da primeira entrada (default: -1).
      w2: Peso sináptico da segunda entrada (default: 1).
      theta: Limiar (default: 0).

    Returns:
      A saída do perceptron (0 ou 1).
    """
    
    soma = w1 * x1 + w2 * x2
    if soma >= theta:
        return 1
    else:
        return 0

# Testando a função
print(perceptron_implication(0, 0)) 
print(perceptron_implication(0, 1))  
print(perceptron_implication(1, 0))  
print(perceptron_implication(1, 1))  
