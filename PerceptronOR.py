def perceptron_or(x1, x2, w1=1, w2=1, theta=0.5):
    """
    Implementa um perceptron para a função lógica OR.

    Args:
      x1: Primeira entrada (0 ou 1).
      x2: Segunda entrada (0 ou 1).
      w1: Peso da primeira entrada (default: 1).
      w2: Peso da segunda entrada (default: 1).
      theta: Limiar de ativação (default: 0.5).

    Returns:
      A saída do perceptron (0 ou 1).
    """
    
    soma = w1 * x1 + w2 * x2
    if soma >= theta:
        return 1
    else:
        return 0


print(perceptron_or(0, 0))  
print(perceptron_or(0, 1))  
print(perceptron_or(1, 0))  
print(perceptron_or(1, 1))  
