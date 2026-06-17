from robo import Robo

import pyray as rl

class Curiosity_world():
    def __init__(self):
        self.terreno = self.build_world() # Lista de objetos do terreno
        self.agente = Robo(self) # Agente do mundo, para interagir com o ambiente
        self.escala = 50

    def update(self, dt):
        self.agente.update(dt)

    def render_terreno(self):
        for i, linha in enumerate(self.terreno):
            for j, char in enumerate(linha):
                match char:
                    case "#":
                        # Renderizar um obstáculo: quadrado azul
                        rl.draw_rectangle(j * self.escala, i * self.escala, self.escala, self.escala, rl.BLUE)
                    case ".":
                        # Renderizar um chão vazio: quadrado cinza
                        rl.draw_rectangle(j * self.escala, i * self.escala, self.escala, self.escala, rl.GRAY)

    def render(self):
        self.render_terreno()
        self.agente.render()

    def test_robot_colision_with_terrain(self, raio, pos_alvo):
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
                if self.terreno[y][x] == "#":
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