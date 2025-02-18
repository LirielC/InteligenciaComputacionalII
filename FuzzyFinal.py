import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

def obter_entrada_usuario(pergunta, min_val, max_val, opcoes):
    print(pergunta)
    for key, value in opcoes.items():
        print(f"{key}: {value}")
    entrada = float(input(f"Escolha um valor entre {min_val} e {max_val}: "))
    while entrada < min_val or entrada > max_val:
        print(f"Valor fora do intervalo. Por favor, insira um valor entre {min_val} e {max_val}.")
        entrada = float(input(f"Escolha um valor entre {min_val} e {max_val}: "))
    return entrada

idade = ctrl.Antecedent(np.arange(0, 101, 1), 'idade')
doenca_normal = ctrl.Antecedent(np.arange(0, 11, 1), 'doenca_normal')
historico_familiar = ctrl.Antecedent(np.arange(0, 11, 1), 'historico_familiar')


idade_falecimento = ctrl.Consequent(np.arange(0, 121, 1), 'idade_falecimento')

idade['jovem'] = fuzz.trimf(idade.universe, [0, 0, 30])
idade['adulto'] = fuzz.trimf(idade.universe, [25, 50, 75])
idade['idoso'] = fuzz.trimf(idade.universe, [65, 80, 100])


doenca_normal['saudavel'] = fuzz.trimf(doenca_normal.universe, [0, 0, 3])
doenca_normal['moderada'] = fuzz.trimf(doenca_normal.universe, [2, 5, 8])
doenca_normal['grave'] = fuzz.trimf(doenca_normal.universe, [7, 10, 10])


historico_familiar['bom'] = fuzz.trimf(historico_familiar.universe, [0, 0, 4])
historico_familiar['medio'] = fuzz.trimf(historico_familiar.universe, [3, 5, 7])
historico_familiar['ruim'] = fuzz.trimf(historico_familiar.universe, [6, 10, 10])


idade_falecimento['precoce'] = fuzz.trimf(idade_falecimento.universe, [0, 45, 65])
idade_falecimento['normal'] = fuzz.trimf(idade_falecimento.universe, [60, 75, 90])
idade_falecimento['tardia'] = fuzz.trimf(idade_falecimento.universe, [85, 100, 120])

regra1 = ctrl.Rule(idade['jovem'] & doenca_normal['saudavel'] & historico_familiar['bom'], idade_falecimento['tardia'])
regra2 = ctrl.Rule(idade['adulto'] & doenca_normal['moderada'] & historico_familiar['medio'], idade_falecimento['normal'])
regra3 = ctrl.Rule(idade['idoso'] & doenca_normal['grave'] & historico_familiar['ruim'], idade_falecimento['precoce'])
regra4 = ctrl.Rule(idade['jovem'] & doenca_normal['grave'], idade_falecimento['normal'])
regra5 = ctrl.Rule(idade['idoso'] & doenca_normal['saudavel'] & historico_familiar['bom'], idade_falecimento['tardia'])
regra6 = ctrl.Rule(doenca_normal['grave'] | historico_familiar['ruim'], idade_falecimento['precoce'])
regra7 = ctrl.Rule(idade['adulto'] & doenca_normal['saudavel'] & historico_familiar['bom'], idade_falecimento['tardia'])
regra8 = ctrl.Rule(idade['jovem'] & doenca_normal['moderada'] & historico_familiar['medio'], idade_falecimento['normal'])
regra9 = ctrl.Rule(idade['adulto'] & doenca_normal['grave'] & historico_familiar['ruim'], idade_falecimento['precoce'])
regra10 = ctrl.Rule(idade['idoso'] & doenca_normal['moderada'] & historico_familiar['medio'], idade_falecimento['normal'])


sistema_ctrl = ctrl.ControlSystem([regra1, regra2, regra3, regra4, regra5, regra6, regra7, regra8, regra9, regra10])

def calcular_idade_falecimento(idade_input, doenca_input, historico_input):
    sistema = ctrl.ControlSystemSimulation(sistema_ctrl)
    sistema.input['idade'] = idade_input
    sistema.input['doenca_normal'] = doenca_input
    sistema.input['historico_familiar'] = historico_input

    try:
        sistema.compute()
        idade_falecimento_output = sistema.output['idade_falecimento']
        idade_falecimento_output = max(idade_falecimento_output, idade_input)
        if historico_input >= 7:
            idade_falecimento_output += 20  
    except KeyError:
        print("Erro: Não foi possível calcular a idade de falecimento.")
        idade_falecimento_output = None

    return idade_falecimento_output

def calcular_valor_plano(idade_atual, idade_falecimento_prevista, taxa_anual=5000):
    anos_restantes = max(0, idade_falecimento_prevista - idade_atual)

    if anos_restantes < 1:
        valor_mensal = (taxa_anual * 1.7) / 12
    else:
        valor_total = taxa_anual * anos_restantes
        valor_mensal = valor_total / (anos_restantes * 12)

    return valor_mensal

def plotar_pertinencia_idade_falecimento(idade_prevista):
    y_precoce = fuzz.interp_membership(idade_falecimento.universe, idade_falecimento['precoce'].mf, idade_prevista)
    y_normal = fuzz.interp_membership(idade_falecimento.universe, idade_falecimento['normal'].mf, idade_prevista)
    y_tardia = fuzz.interp_membership(idade_falecimento.universe, idade_falecimento['tardia'].mf, idade_prevista)

    plt.figure(figsize=(12, 6))
    plt.plot(idade_falecimento.universe, idade_falecimento['precoce'].mf, 'b', linewidth=2, label='Precoce')
    plt.plot(idade_falecimento.universe, idade_falecimento['normal'].mf, 'g', linewidth=2, label='Normal')
    plt.plot(idade_falecimento.universe, idade_falecimento['tardia'].mf, 'r', linewidth=2, label='Tardia')
    plt.vlines(idade_prevista, 0, 1, colors='k', linestyles='dashed', linewidth=2, label='Idade Prevista')
    plt.title('Pertinência da Idade de Falecimento Prevista', fontsize=16)
    plt.xlabel('Idade', fontsize=14)
    plt.ylabel('Pertinência', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)

    print(f"\nPertinência para idade de falecimento prevista ({idade_prevista:.2f} anos):")
    print(f"Precoce: {y_precoce:.2f}")
    print(f"Normal: {y_normal:.2f}")
    print(f"Tardia: {y_tardia:.2f}")

    plt.show()


nome = input("Digite o nome da pessoa: ")

idade_input = obter_entrada_usuario(
    "Qual é a idade da pessoa?",
    0, 100,
    {0: "Recém-nascido",
     5: "Criança (0-10 anos)",
     15: "Adolescente (10-18 anos)",
     22: "Jovem adulto (18-25 anos)",
     35: "Adulto (25-40 anos)",
     50: "Meia-idade (40-60 anos)",
     70: "Idoso (60-80 anos)",
     90: "Muito idoso (80+ anos)",
     100: "Centenário"}
)

saude_input = obter_entrada_usuario(
    "Como você avalia o nível de saúde atual da pessoa?",
    0, 10,
    {0: "Muito ruim - Múltiplas condições de saúde graves", 
     3: "Ruim - Algumas condições de saúde significativas", 
     5: "Regular - Condições de saúde controladas", 
     7: "Bom - Poucas preocupações de saúde", 
     10: "Excelente - Sem problemas de saúde significativos"}
)

historico_input = obter_entrada_usuario(
    "Como você avalia o histórico familiar de saúde da pessoa?",
    0, 10,
    {0: "Muito ruim - Múltiplas doenças hereditárias graves", 
     3: "Ruim - Histórico significativo de doenças", 
     5: "Regular - Algumas doenças na família", 
     7: "Bom - Poucas doenças familiares", 
     10: "Excelente - Sem doenças hereditárias significativas"}
)


idade_falecimento_prevista = calcular_idade_falecimento(idade_input, saude_input, historico_input)
print(f"\n{nome}, a idade de falecimento prevista é de {idade_falecimento_prevista:.2f} anos.")


valor_plano_mensal = calcular_valor_plano(idade_input, idade_falecimento_prevista)
print(f"O valor mensal do plano funerário será de R${valor_plano_mensal:.2f} por mês.")


plotar_pertinencia_idade_falecimento(idade_falecimento_prevista)
