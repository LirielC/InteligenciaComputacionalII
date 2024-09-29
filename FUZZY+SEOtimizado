import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

class SistemaPrevisaoFalecimento:
    def __init__(self):
        self.criar_variaveis()
        self.criar_regras()
        self.criar_sistema_controle()
        self.pesos = {'idade': 1, 'doenca': 1, 'historico': 0.5}  # Peso do histórico é metade do peso da doença

    def criar_variaveis(self):
        # Variáveis de entrada
        self.idade = ctrl.Antecedent(np.arange(0, 101, 1), 'idade')
        self.doenca = ctrl.Antecedent(np.arange(0, 11, 1), 'doenca')
        self.historico = ctrl.Antecedent(np.arange(0, 11, 1), 'historico')

        # Variável de saída
        self.idade_falecimento = ctrl.Consequent(np.arange(0, 121, 1), 'idade_falecimento')

        # Funções de pertinência
        self.idade['jovem'] = fuzz.trimf(self.idade.universe, [0, 0, 30])
        self.idade['adulto'] = fuzz.trimf(self.idade.universe, [25, 50, 75])
        self.idade['idoso'] = fuzz.trimf(self.idade.universe, [65, 80, 100])

        self.doenca['saudavel'] = fuzz.trimf(self.doenca.universe, [0, 0, 3])
        self.doenca['moderada'] = fuzz.trimf(self.doenca.universe, [2, 5, 8])
        self.doenca['grave'] = fuzz.trimf(self.doenca.universe, [7, 10, 10])

        self.historico['bom'] = fuzz.trimf(self.historico.universe, [0, 0, 4])
        self.historico['medio'] = fuzz.trimf(self.historico.universe, [3, 5, 7])
        self.historico['ruim'] = fuzz.trimf(self.historico.universe, [6, 10, 10])

        self.idade_falecimento['precoce'] = fuzz.trimf(self.idade_falecimento.universe, [0, 45, 65])
        self.idade_falecimento['normal'] = fuzz.trimf(self.idade_falecimento.universe, [60, 75, 90])
        self.idade_falecimento['tardia'] = fuzz.trimf(self.idade_falecimento.universe, [85, 100, 120])

    def criar_regras(self):
        self.regras = [
            ctrl.Rule(self.idade['jovem'] & self.doenca['saudavel'], self.idade_falecimento['tardia']),
            ctrl.Rule(self.idade['adulto'] & self.doenca['moderada'], self.idade_falecimento['normal']),
            ctrl.Rule(self.idade['idoso'] & self.doenca['grave'], self.idade_falecimento['precoce']),
            ctrl.Rule(self.idade['jovem'] & self.doenca['grave'], self.idade_falecimento['normal']),
            ctrl.Rule(self.idade['idoso'] & self.doenca['saudavel'], self.idade_falecimento['tardia']),
            ctrl.Rule(self.doenca['grave'], self.idade_falecimento['precoce']),
            ctrl.Rule(self.idade['adulto'] & self.doenca['saudavel'], self.idade_falecimento['tardia']),
            ctrl.Rule(self.idade['jovem'] & self.doenca['moderada'], self.idade_falecimento['normal']),
            ctrl.Rule(self.idade['adulto'] & self.doenca['grave'], self.idade_falecimento['precoce']),
            ctrl.Rule(self.idade['idoso'] & self.doenca['moderada'], self.idade_falecimento['normal']),
            ctrl.Rule(self.historico['ruim'] & self.doenca['moderada'], self.idade_falecimento['normal']),
            ctrl.Rule(self.historico['bom'] & self.doenca['saudavel'], self.idade_falecimento['tardia'])
        ]

    def criar_sistema_controle(self):
        self.sistema_ctrl = ctrl.ControlSystem(self.regras)

    def calcular_idade_falecimento(self, idade_input, doenca_input, historico_input):
        sistema = ctrl.ControlSystemSimulation(self.sistema_ctrl)
        sistema.input['idade'] = idade_input
        sistema.input['doenca'] = doenca_input
        sistema.input['historico'] = historico_input

        try:
            sistema.compute()
            idade_falecimento_base = sistema.output['idade_falecimento']

            # Aplicar os pesos para ajustar a idade de falecimento
            ajuste_doenca = (doenca_input - 5) * self.pesos['doenca']  # 5 é o valor médio da escala de doença
            ajuste_historico = (historico_input - 5) * self.pesos['historico']  # 5 é o valor médio da escala de histórico

            idade_falecimento_ajustada = idade_falecimento_base - ajuste_doenca - ajuste_historico

            return max(idade_falecimento_ajustada, idade_input)
        except Exception as e:
            print(f"Erro no cálculo: {e}")
            return None

    def calcular_pesos_relativos(self):
        total = sum(self.pesos.values())
        return {k: (v / total) * 100 for k, v in self.pesos.items()}

    def plotar_pertinencia(self, idade_prevista):
        y_precoce = fuzz.interp_membership(self.idade_falecimento.universe, self.idade_falecimento['precoce'].mf, idade_prevista)
        y_normal = fuzz.interp_membership(self.idade_falecimento.universe, self.idade_falecimento['normal'].mf, idade_prevista)
        y_tardia = fuzz.interp_membership(self.idade_falecimento.universe, self.idade_falecimento['tardia'].mf, idade_prevista)

        plt.figure(figsize=(12, 6))
        plt.plot(self.idade_falecimento.universe, self.idade_falecimento['precoce'].mf, 'b', linewidth=2, label='Precoce')
        plt.plot(self.idade_falecimento.universe, self.idade_falecimento['normal'].mf, 'g', linewidth=2, label='Normal')
        plt.plot(self.idade_falecimento.universe, self.idade_falecimento['tardia'].mf, 'r', linewidth=2, label='Tardia')
        plt.vlines(idade_prevista, 0, 1, colors='k', linestyles='dashed', linewidth=2, label='Idade Prevista')
        plt.title('Pertinência da Idade de Falecimento Prevista', fontsize=16)
        plt.xlabel('Idade', fontsize=14)
        plt.ylabel('Pertinência', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.show()

        print(f"\nPertinência para idade de falecimento prevista ({idade_prevista:.2f} anos):")
        print(f"Precoce: {y_precoce:.2f}")
        print(f"Normal: {y_normal:.2f}")
        print(f"Tardia: {y_tardia:.2f}")

class CalculadoraPlanoFunerario:
    def __init__(self, taxa_anual=5000):
        self.taxa_anual = taxa_anual

    def calcular_valor_plano(self, idade_atual, idade_falecimento_prevista):
        anos_restantes = max(0, idade_falecimento_prevista - idade_atual)

        if anos_restantes < 1:
            valor_mensal = (self.taxa_anual * 1.7) / 12  # Aumento de 70%
        elif anos_restantes > 0:
            valor_total = self.taxa_anual * anos_restantes
            valor_mensal = valor_total / (anos_restantes * 12)
        else:
            valor_mensal = self.taxa_anual / 12

        return valor_mensal

def main():
    sistema = SistemaPrevisaoFalecimento()
    calculadora = CalculadoraPlanoFunerario()

    print("Bem-vindo ao Sistema de Previsão de Idade de Falecimento e Cálculo de Plano Funerário")

    pesos = sistema.calcular_pesos_relativos()
    print("\nPeso relativo de cada entrada no sistema:")
    for var, peso in pesos.items():
        print(f"{var.capitalize()}: {peso:.2f}%")

    nome = input("\nDigite o nome da pessoa: ")
    idade_input = float(input("Digite a idade da pessoa: "))
    doenca_input = float(input("Digite o nível de doença (0-10, onde 0 é saudável e 10 é muito doente): "))
    historico_input = float(input("Digite o histórico familiar de doenças (0-10, onde 0 é excelente e 10 é ruim): "))

    idade_prevista = sistema.calcular_idade_falecimento(idade_input, doenca_input, historico_input)

    if idade_prevista:
        valor_mensal_plano = calculadora.calcular_valor_plano(idade_input, idade_prevista)
        print(f"\nResultados para {nome}:")
        print(f"Idade atual: {idade_input:.0f} anos")
        print(f"Idade prevista de falecimento: {idade_prevista:.2f} anos")
        print(f"Valor mensal estimado do plano funerário: R${valor_mensal_plano:.2f}")

        if idade_prevista - idade_input < 1:
            print("AVISO: O valor do plano funerário foi aumentado em 70% devido à previsão de falecimento no mesmo ano.")

        sistema.plotar_pertinencia(idade_prevista)
    else:
        print("Erro no cálculo da idade de falecimento. Verifique as entradas.")

if __name__ == "__main__":
    main()
