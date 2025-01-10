import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

idade = ctrl.Antecedent(np.arange(0, 101, 1), 'idade')
doenca_normal = ctrl.Antecedent(np.arange(0, 11, 1), 'doenca_normal')
historico_familiar = ctrl.Antecedent(np.arange(0, 11, 1), 'historico_familiar')

idade_falecimento = ctrl.Consequent(np.arange(0, 121, 1), 'idade_falecimento')

idade['jovem'] = fuzz.trimf(idade.universe, [0, 0, 35])
idade['adulto'] = fuzz.trimf(idade.universe, [25, 50, 75])
idade['idoso'] = fuzz.trimf(idade.universe, [65, 100, 100])

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
    except KeyError:
        print("Erro: Não foi possível calcular a idade de falecimento.")
        idade_falecimento_output = None

    return idade_falecimento_output

def calcular_valor_plano(idade_atual, idade_falecimento_prevista, taxa_anual=5000):
    anos_restantes = max(0, idade_falecimento_prevista - idade_atual)

    if anos_restantes < 1:  
        valor_mensal = (taxa_anual * 1.7) / 12  
    elif anos_restantes > 0:
        valor_total = taxa_anual * anos_restantes
        valor_mensal = valor_total / (anos_restantes * 12)
    else:
        valor_mensal = taxa_anual / 12  
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
idade_input = float(input("Digite a idade da pessoa: "))
doenca_input = float(input("Digite o nível de doença (0-10, onde 0 é saudável e 10 é muito doente): "))
historico_input = float(input("Digite o histórico familiar de doenças (0-10, onde 0 é excelente e 10 é ruim): "))

idade_prevista = calcular_idade_falecimento(idade_input, doenca_input, historico_input)

if idade_prevista:
    valor_mensal_plano = calcular_valor_plano(idade_input, idade_prevista)
    print(f"\nResultados para {nome}:")
    print(f"Idade atual: {idade_input:.0f} anos")
    print(f"Idade prevista de falecimento: {idade_prevista:.2f} anos")
    print(f"Valor mensal estimado do plano funerário: R${valor_mensal_plano:.2f}")

    if idade_prevista - idade_input < 1:
        print("AVISO: O valor do plano funerário foi aumentado em 70% devido à previsão de falecimento no mesmo ano. E o restante do valor será parcelado mensalmente")

    plotar_pertinencia_idade_falecimento(idade_prevista)
else:
    print("Erro no cálculo da idade de falecimento. Verifique as entradas.")
