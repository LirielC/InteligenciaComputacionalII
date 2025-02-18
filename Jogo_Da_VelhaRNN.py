import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import tkinter as tk
from tkinter import messagebox

def checar_vencedor(tabu):
    posicoes_vencedoras = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                           (0, 3, 6), (1, 4, 7), (2, 5, 8),
                           (0, 4, 8), (2, 4, 6)]
    for a, b, c in posicoes_vencedoras:
        if tabu[a] == tabu[b] == tabu[c] != 0:
            return tabu[a]
    return 0

def gerar_treinamento_dados(num_games=5000):
    X_train, y_train = [], []
    for _ in range(num_games):
        tabu = np.zeros(9)
        player_vez = 1
        game_history = []
        move_history = []
        while True:
            if player_vez == 1:
                move = np.random.choice([i for i in range(9) if tabu[i] == 0])
            else:
                move = np.random.choice([i for i in range(9) if tabu[i] == 0])

            tabu[move] = player_vez
            game_history.append(tabu.copy())
            move_history.append((move, player_vez))
            
            winner = checar_vencedor(tabu)
            if winner != 0 or not 0 in tabu:
                break
            player_vez *= -1
        
        if winner == 1:  
            for i in range(len(game_history) - 1):
                X_train.append(game_history[i] / 2)  
                y_train.append(game_history[i + 1] / 2)
    return np.array(X_train), np.array(y_train)

def modelo_de_construcao():
    modelo = Sequential([
        SimpleRNN(128, input_shape=(1, 9), activation='relu', return_sequences=True),
        Dropout(0.3),
        LSTM(128, activation='relu', return_sequences=True),
        Dropout(0.3),
        LSTM(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(9, activation='linear')  
    ])
    modelo.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')  
    return modelo


X_train, y_train = gerar_treinamento_dados()
X_train = X_train.reshape((-1, 1, 9))
y_train = y_train.reshape((-1, 9))


modelo = modelo_de_construcao()
modelo.fit(X_train, y_train, epochs=150, batch_size=64)


def fazer_movimento(tabu):
    tabu_input = (tabu / 2).reshape((1, 1, 9))  
    prediction = modelo.predict(tabu_input)
    movimento = np.argmax(prediction)
    while tabu[movimento] != 0:  
        prediction[0][movimento] = -1
        movimento = np.argmax(prediction)
    return movimento


class Jogo_da_VelhaAPP:
    def __init__(self, root):
        self.root = root
        self.root.title("Jogo da Velha - IA com RNN e LSTM")
        self.tabu = np.zeros(9)
        self.buttons = []
        self.create_tabu()

    def create_tabu(self):
        for i in range(9):
            button = tk.Button(self.root, text="", font=("Arial", 24), width=5, height=2,
                               command=lambda i=i: self.movimento_usuario(i))
            button.grid(row=i // 3, column=i % 3)
            self.buttons.append(button)

    def movimento_usuario(self, index):
        if self.tabu[index] == 0:
            self.tabu[index] = -1
            self.buttons[index].config(text="X", state="disabled")
            winner = checar_vencedor(self.tabu)
            if winner == -1:
                messagebox.showinfo("Fim do Jogo", "Você venceu!")
                self.reset_tabu()
            elif not 0 in self.tabu:
                messagebox.showinfo("Fim do Jogo", "Empate!")
                self.reset_tabu()
            else:
                self.ai_movimento()

    def ai_movimento(self):
        movimento = fazer_movimento(self.tabu)
        self.tabu[movimento] = 1
        self.buttons[movimento].config(text="O", state="disabled")
        winner = checar_vencedor(self.tabu)
        if winner == 1:
            messagebox.showinfo("Fim do Jogo", "A IA venceu!")
            self.reset_tabu()
        elif not 0 in self.tabu:
            messagebox.showinfo("Fim do Jogo", "Empate!")
            self.reset_tabu()

    def reset_tabu(self):
        self.tabu = np.zeros(9)
        for button in self.buttons:
            button.config(text="", state="normal")

root = tk.Tk()
app = Jogo_da_VelhaAPP(root)
root.mainloop()

   
