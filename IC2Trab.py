import numpy as np
import matplotlib.pyplot as plt

idade = np.linspace(0, 100, 100)  
custo_funerario = np.linspace(0, 10000, 100)  

def triangular(x, a, b, c):
    if x <= a:
        return 0
    elif a < x < b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return (c - x) / (c - b)
    else:
        return 0


def pertinencia_idade(idade_falecido):
    if idade_falecido <= 45:
        return 'baixo'
    elif 45 < idade_falecido < 60:
        return 'moderado'
    else:
        return 'alto'

def pertinencia_doenca(doenca):
    if doenca == 'grave':
        return 'grave'
    elif doenca == 'moderada':
        return 'moderada'
    elif doenca == 'leve':
        return 'leve'
    else:
        return 'nenhuma'


def calcular_mortalidade(idade_falecido, doenca, historico_familiar):
    idade_cat = pertinencia_idade(idade_falecido)
    doenca_cat = pertinencia_doenca(doenca)

    if idade_cat == 'baixo':
        if doenca_cat == 'grave':
            return 'alta'
        elif doenca_cat == 'moderada':
            return 'moderada'
        elif doenca_cat == 'leve':
            return 'baixa'
        else:
            return 'baixa'

    elif idade_cat == 'moderado':
        if doenca_cat == 'grave':
            return 'alta'
        elif doenca_cat == 'moderada':
            return 'moderada'
        elif doenca_cat == 'leve':
            return 'moderada'
        else:
            return 'moderada'

    elif idade_cat == 'alto':
        if doenca_cat == 'grave':
            return 'alta'
        elif doenca_cat == 'moderada':
            return 'alta'
        elif doenca_cat == 'leve':
            return 'alta'
        elif historico_familiar == 'alto_risco':
            return 'alta'
        elif historico_familiar == 'moderado':
            return 'alta'
        else:
            return 'alta'


def plano_funerario(idade_falecido, custo_funerario_desejado, doenca=None, historico_familiar=None):
    mortalidade = calcular_mortalidade(idade_falecido, doenca, historico_familiar)

    risco_custo = {
        'alta': 0.8,     
        'moderada': 0.5,   
        'baixa': 0.2,      
    }

    custo_funerario_calculado = custo_funerario_desejado * risco_custo[mortalidade]

    lucro_minimo = 0.6 * custo_funerario_calculado
    custo_mensal = custo_funerario_calculado / 12 + lucro_minimo / 12  

    return custo_funerario_calculado, custo_mensal

idade_falecido = 50
custo_funerario_desejado = 5000
doenca = "diabetes"  
historico_familiar = "moderado"  
custo_funerario_calculado, custo_mensal = plano_funerario(idade_falecido, custo_funerario_desejado, doenca, historico_familiar)
print(f"Custo do funeral calculado: {custo_funerario_calculado:.2f}")
print(f"Custo mensal do plano funerário: {custo_mensal:.2f}")


plt.plot(idade, [triangular(x, 0, 45, 60) for x in idade], label="Risco Baixo (0-45 anos)")
plt.plot(idade, [triangular(x, 45, 60, 75) for x in idade], label="Risco Moderado (45-60 anos)")
plt.plot(idade, [triangular(x, 60, 75, 100) for x in idade], label="Risco Alto (61+ anos)")
plt.xlabel("Idade")
plt.ylabel("Pertinência")
plt.title("Funções de Pertinência para Risco de Morte")
plt.legend()
plt.grid()
plt.show()
