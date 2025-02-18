def step_function(sum):
  """Função de ativação do perceptron. Retorna 1 se a soma for maior ou igual a 0, caso contrário, retorna 0."""
  return 1 if sum >= 0 else 0

def perceptron_and(x1, x2, w1=1, w2=1, bias=-1.5):
  """Implementa a porta lógica AND usando um perceptron.

  Args:
    x1: Primeira entrada.
    x2: Segunda entrada.
    w1: Peso da primeira entrada.
    w2: Peso da segunda entrada.
    bias: Viés.

  Returns:
    Saída do perceptron (0 ou 1).
  """

  sum = w1*x1 + w2*x2 + bias
  return step_function(sum)

if __name__ == "__main__":

  for x1 in [0, 1]:
    for x2 in [0, 1]:
      output = perceptron_and(x1, x2)
      print(f"Entrada: ({x1}, {x2}), Saída: {output}")
