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

    def test_robot_colision_with_terrain(self, robo, pos_alvo):
        # Testa se a posição alvo do robô colide com o terreno, retornando True se não colidir e False se colidir
        # Para isso, vamos verificar os quatro cantos do robô, considerando seu raio, para ver se algum deles colide com um obstáculo
        for i in range(-1, 2, 2):
            for j in range(-1, 2, 2):
                x = int((pos_alvo.x + i * robo.raio) // self.escala)
                y = int((pos_alvo.y + j * robo.raio) // self.escala)
                if self.terreno[y][x] == "#":
                    return True

        return False


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