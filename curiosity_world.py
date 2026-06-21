import math

from robo import Robo

import pyray as rl

class Curiosity_world():
    def __init__(self, handler):
        self.handler = handler
        self.terreno = self.build_world() # Lista de objetos do terreno
        self.agente = Robo(self, "cerebro/") # Agente do mundo, para interagir com o ambiente
        self.escala = 50
        self.ticks = 0
        self.tempo_simulacao = 25611

    def update(self, dt):
        self.agente.update(dt)
        self.ticks += 1
        if self.ticks >= self.tempo_simulacao:
            self.ticks = 0
            self.agente = Robo(self, "cerebro/") # Reinicia o agente para testar o aprendizado ao longo do tempo

    def render_terreno(self):
        for i, linha in enumerate(self.terreno):
            for j, char in enumerate(linha):
                match char:
                    case "#":
                        # Renderizar um obstáculo: quadrado azul
                        rl.draw_rectangle(j * self.escala, i * self.escala, self.escala, self.escala, rl.BLUE)
                    case "G":
                        # Renderizar um obstáculo: quadrado azul
                        rl.draw_rectangle(j * self.escala, i * self.escala, self.escala, self.escala, rl.GREEN)
                    case "B":
                        # Renderizar um obstáculo: quadrado azul
                        rl.draw_rectangle(j * self.escala, i * self.escala, self.escala, self.escala, rl.RED)
                    case "C":
                        # Renderizar uma haste rotacionando e mudando de cor
                        rl.draw_rectangle(j * self.escala, i * self.escala, self.escala, self.escala, rl.GRAY)
                        # DrawLineEx(Vector2 startPos, Vector2 endPos, float thick, Color color); 
                        start_pos = rl.Vector2(j * self.escala + self.escala // 2 - self.escala//2 * math.cos(self.ticks * 0.05), i * self.escala + self.escala // 2 - self.escala//2 * math.sin(self.ticks * 0.05))     
                        end_pos = rl.Vector2(j * self.escala + self.escala // 2 + self.escala//2 * math.cos(self.ticks * 0.05), i * self.escala + self.escala // 2 + self.escala//2 * math.sin(self.ticks * 0.05))
                        color = rl.Color(255, int(127 + 127 * math.sin(self.ticks * 0.1)), int(127 - 127 * math.sin(self.ticks * 0.1)), 255)
                        rl.draw_line_ex(start_pos, end_pos, 5.0, color)

                    case ".":
                        # Renderizar um chão vazio: quadrado cinza
                        rl.draw_rectangle(j * self.escala, i * self.escala, self.escala, self.escala, rl.GRAY)

    def render_hud(self):
        # Renderizar o HUD, com informações sobre o agente, como posição, velocidade, recompensa acumulada, etc.
        rl.draw_text(f"Posição: ({self.agente.pos.x:.2f}, {self.agente.pos.y:.2f})", 1010, 10, 20, rl.WHITE)
        rl.draw_text(f"Velocidade: ({self.agente.vel.x:.2f}, {self.agente.vel.y:.2f})", 1010, 40, 20, rl.WHITE)
        rl.draw_text(f"Aceleração: ({self.agente.acc.x:.2f}, {self.agente.acc.y:.2f})", 1010, 70, 20, rl.WHITE)
        rl.draw_text(f"Recompensa acumulada (média): {self.agente.brain.get_moving_average_reward():.2f}", 1010, 100, 20, rl.WHITE)
        rl.draw_text(f"Reconstruction Loss: {self.agente.brain.get_reconstruction_loss():.4f}", 1010, 130, 20, rl.WHITE)
        if self.agente.last_action is not None:
            # Ao invés de texto, colocar duas barras verticais cinzas, uma para cada ação, com altura proporcional ao valor da ação, e um número indicando o valor da ação
            # Ela pode variar de -1 a 1, então a barra deve ser centralizada em 0, com altura máxima de 50 pixels. se 1, 25 para cima, se -1, 25 para baixo, com valores intermediario. Uma linha branca deve estar no fim da barra
            rl.draw_text(f"Última ação: ({self.agente.last_action[0].item():.2f}, {self.agente.last_action[1].item():.2f})", 1010, 160, 20, rl.WHITE)
            rl.draw_rectangle(1270, 155, 20, 50, rl.GRAY)
            rl.draw_rectangle(1270, 180-max(int(self.agente.last_action[0].item() * 25), 0), 20, abs(int(self.agente.last_action[0].item() * 25)), rl.BLUE)
            rl.draw_rectangle(1300, 155, 20, 50, rl.GRAY)
            rl.draw_rectangle(1300, 180-max(int(self.agente.last_action[1].item() * 25), 0), 20, abs(int(self.agente.last_action[1].item() * 25)), rl.BLUE)
        if len(self.agente.brain.memory) < self.agente.brain.warm_up_steps:
            rl.draw_text(f"WARM UP: {len(self.agente.brain.memory)}/{self.agente.brain.warm_up_steps}", 1010, 185, 20, rl.RED)

        # Renderizar a imagem reconstruída, se disponível
        if self.agente.imagem_reconstruida is not None:
            rl.draw_texture(rl.load_texture_from_image(rl.Image(self.agente.imagem_reconstruida.astype("uint8"))), 1010, 210, rl.WHITE)

    def render(self):
        self.render_terreno()
        self.agente.render()
        self.render_hud()

    def test_robot_colision_with_terrain(self, raio, pos_alvo, caracter = "#"):
        # Testa se a posição alvo do robô colide com o terreno, retornando True se não colidir e False se colidir
        # Para isso, vamos verificar os quatro cantos do robô, considerando seu raio, para ver se algum deles colide com um obstáculo
        colisions = {"N": 0, "S": 0, "E": 0, "W": 0}
        colidiu = False
        ordem = ["E", "N", "S", "W"]
        idx = 0
        for i in range(-1, 2, 1):
            for j in range(-1, 2, 1):
                if abs(i) == abs(j):
                    continue
                x = int((pos_alvo.x + i * raio) // self.escala)
                y = int((pos_alvo.y + j * raio) // self.escala)
                if self.terreno[y][x] == caracter:
                    # salva quanto cada sensor foi penetrado no obstáculo
                    # Computa com base no lado a ser penetrado. Se é esqueda, é o quanto o x do robô ultrapassa o limite direito do obstáculo, se é direita, é o quanto o x do robô ultrapassa o limite esquerdo do obstáculo, se é cima, é o quanto o y do robô ultrapassa o limite inferior do obstáculo, se é baixo, é o quanto o y do robô ultrapassa o limite superior do obstáculo
                    penetracao = 0
                    match ordem[idx]:
                        case "E":
                            penetracao = raio - (pos_alvo.x - (x * self.escala + self.escala))
                        case "W":
                            penetracao = raio - (x * self.escala - pos_alvo.x)
                        case "N":
                            penetracao = raio - (pos_alvo.y - (y * self.escala + self.escala))
                        case "S":
                            penetracao = raio - (y * self.escala - pos_alvo.y)
                    colisions[ordem[idx]] = penetracao
                    colidiu = True
                idx += 1
        return colidiu, colisions


    def build_world(self):
        # Lógica para construir o mundo, como criar objetos do terreno ou definir obstáculos

        # Ler o terreno a partir de um arquivo
        with open("mapa.txt", "r") as f:
            terreno = []
            for line in f:
                row = []
                for char in line:
                    row.append(char)
                terreno.append(row)
        return terreno