import math

import pyray as rl

class Robo():
    def __init__(self, world=None):
        self.pos = rl.Vector2(500, 500)
        self.pos_alvo = rl.Vector2(500, 500)
        self.vel = rl.Vector2(0, 0)
        self.acc = rl.Vector2(0, 0)
        self.atrito = 0.1
        self.limiar_atrito = 0.01
        self.acc_max = 0.5
        self.angulo = 0
        self.vel_angular = 0
        self.acc_angular = 0
        self.atrito_angular = 0.1
        self.limiar_atrito_angular = 0.001
        self.acc_max_angular = 0.01 # Em radianos por frame
        self.world = world
        self.raio = 20
        self.raio_visao = 80
        self.raio_tato = 30

        self.tipos_controle = ["absoluto", "rotacional", "rodas"]
        self.controle_atual = 0

        self.tipos_sensores = ["visao", "tato"]
        self.sensores_ativos = [True, True]
        self.tato = {"N": 0, "S": 0, "E": 0, "W": 0}

    def controles(self):
        match self.tipos_controle[self.controle_atual]:
            case "absoluto":

                # Se precionar as setas, o robô se move na direção correspondente
                if (rl.is_key_down(rl.KEY_RIGHT) or rl.is_key_down(rl.KEY_D)) and (rl.is_key_down(rl.KEY_LEFT) or rl.is_key_down(rl.KEY_A)):
                    self.acc.x = 0
                elif rl.is_key_down(rl.KEY_RIGHT) or rl.is_key_down(rl.KEY_D):
                    self.acc.x = self.acc_max
                elif rl.is_key_down(rl.KEY_LEFT) or rl.is_key_down(rl.KEY_A):
                    self.acc.x = -self.acc_max
                else:
                    self.acc.x = 0

                if (rl.is_key_down(rl.KEY_DOWN) or rl.is_key_down(rl.KEY_S)) and (rl.is_key_down(rl.KEY_UP) or rl.is_key_down(rl.KEY_W)):
                    self.acc.y = 0
                elif rl.is_key_down(rl.KEY_DOWN) or rl.is_key_down(rl.KEY_S):
                    self.acc.y = self.acc_max
                elif rl.is_key_down(rl.KEY_UP) or rl.is_key_down(rl.KEY_W):
                    self.acc.y = -self.acc_max
                else:            
                    self.acc.y = 0

            case "rotacional":
                # Setas para os lados rotacionam o robô, setas para cima e para baixo movem o robô para frente e para trás na direção que ele está virado
                if (rl.is_key_down(rl.KEY_RIGHT) or rl.is_key_down(rl.KEY_D)) and (rl.is_key_down(rl.KEY_LEFT) or rl.is_key_down(rl.KEY_A)):
                    self.acc_angular = 0
                elif rl.is_key_down(rl.KEY_RIGHT) or rl.is_key_down(rl.KEY_D):
                    self.acc_angular = self.acc_max_angular
                elif rl.is_key_down(rl.KEY_LEFT) or rl.is_key_down(rl.KEY_A):
                    self.acc_angular = -self.acc_max_angular
                else:
                    self.acc_angular = 0

                if (rl.is_key_down(rl.KEY_DOWN) or rl.is_key_down(rl.KEY_S)) and (rl.is_key_down(rl.KEY_UP) or rl.is_key_down(rl.KEY_W)):
                    self.acc.x = 0
                    self.acc.y = 0
                elif rl.is_key_down(rl.KEY_DOWN) or rl.is_key_down(rl.KEY_S):
                    self.acc.x = self.acc_max * math.cos(self.angulo)
                    self.acc.y = self.acc_max * math.sin(self.angulo)
                elif rl.is_key_down(rl.KEY_UP) or rl.is_key_down(rl.KEY_W):
                    self.acc.x = -self.acc_max * math.cos(self.angulo)
                    self.acc.y = -self.acc_max * math.sin(self.angulo)
                else:            
                    self.acc.x = 0
                    self.acc.y = 0

            case "rodas":
                # Usa seta cima e baixo para controlar a roda direita, e W e S para controlar a roda esquerda, o robô se move na direção correspondente a diferença de velocidade entre as rodas
                if (rl.is_key_down(rl.KEY_W)) and (rl.is_key_down(rl.KEY_S)):
                    acc_roda_esquerda = 0
                elif rl.is_key_down(rl.KEY_W):
                    acc_roda_esquerda = -self.acc_max
                elif rl.is_key_down(rl.KEY_S):
                    acc_roda_esquerda = self.acc_max
                else:
                    acc_roda_esquerda = 0

                if (rl.is_key_down(rl.KEY_UP)) and (rl.is_key_down(rl.KEY_DOWN)):
                    acc_roda_direita = 0
                elif rl.is_key_down(rl.KEY_UP):
                    acc_roda_direita = -self.acc_max
                elif rl.is_key_down(rl.KEY_DOWN):
                    acc_roda_direita = self.acc_max
                else:
                    acc_roda_direita = 0

                self.acc.x = (acc_roda_esquerda + acc_roda_direita) * math.cos(self.angulo) / 2
                self.acc.y = (acc_roda_esquerda + acc_roda_direita) * math.sin(self.angulo) / 2
                self.acc_angular = (acc_roda_direita - acc_roda_esquerda) * self.acc_max_angular




    def phisics(self, dt): # Precisa adicionar o dt, mas agora vai ser mt dificil entao n to usando ele
        # Atualiza a velocidade e a posição do robô com base na aceleração
        self.vel.x += self.acc.x 
        self.vel.y += self.acc.y
        self.pos_alvo.x = self.pos.x + self.vel.x
        if self.world.test_robot_colision_with_terrain(self.raio, self.pos_alvo)[0]: # primeiro o x
            self.pos_alvo.x = self.pos.x
        else:
            self.pos.x = self.pos_alvo.x
        self.pos_alvo.y = self.pos.y + self.vel.y
        if self.world.test_robot_colision_with_terrain(self.raio, self.pos_alvo)[0]: # depois o y, assim o robô pode deslizar nas paredes
            self.pos_alvo.y = self.pos.y
        else:
            self.pos.y = self.pos_alvo.y

        # Aplica atrito para reduzir a velocidade ao longo do tempo
        self.vel.x *= (1 - self.atrito)
        self.vel.y *= (1 - self.atrito)
        if self.vel.x < self.limiar_atrito and self.vel.x > -self.limiar_atrito:
            self.vel.x = 0
        if self.vel.y < self.limiar_atrito and self.vel.y > -self.limiar_atrito:
            self.vel.y = 0

        # Atualiza a velocidade angular e o ângulo do robô com base na aceleração angular
        self.vel_angular += self.acc_angular
        self.angulo += self.vel_angular
        # Aplica atrito angular para reduzir a velocidade angular ao longo do tempo
        self.vel_angular *= (1 - self.atrito_angular)
        if self.vel_angular < self.limiar_atrito_angular and self.vel_angular > -self.limiar_atrito_angular:
            self.vel_angular = 0
        

    def update(self, dt:float) -> None:
        self.controles()
        self.phisics(dt)
        if self.sensores_ativos[self.tipos_sensores.index("tato")]:
            self.sensor_tato()

    def sensor_tato(self):
        # Detecta para cada 
        self.tato = self.world.test_robot_colision_with_terrain(self.raio_tato, self.pos_alvo)[1]
        print("Sensor de tato:", self.tato)
        

    def render(self):
        if self.sensores_ativos[self.tipos_sensores.index("visao")]: # Se o sensor de visão estiver ativo, desenha um círculo ao redor do robô representando seu campo de visão
            rl.draw_circle(int(self.pos.x), int(self.pos.y), self.raio_visao, rl.Color(255, 0, 0, 100))

        if self.sensores_ativos[self.tipos_sensores.index("tato")]: # Se o sensor de tato estiver ativo, desenha um círculo ao redor do robô representando seu campo de tato
            rl.draw_circle(int(self.pos.x), int(self.pos.y), self.raio_tato, rl.Color(0, 255, 0, 100))

        match self.tipos_controle[self.controle_atual]:
            case "absoluto":
                # Desenha um círculo representando o robô na posição atual
                rl.draw_circle(int(self.pos.x), int(self.pos.y), self.raio, rl.BLUE)
            case "rotacional":
                # Colisao
                rl.draw_circle(int(self.pos.x), int(self.pos.y), self.raio, rl.Color(20,100,100,255))
                # Desenha um triangulo isoceles representando o robô na posição atual
                rl.draw_poly(self.pos, 3, self.raio, (self.angulo+math.pi) * 180 / math.pi, rl.BLUE)
                pos_ponta = rl.Vector2(self.pos.x + self.raio//2 * math.cos(self.angulo), self.pos.y + self.raio//2 * math.sin(self.angulo))
                rl.draw_poly(pos_ponta, 3, self.raio//2, (self.angulo+math.pi) * 180 / math.pi, rl.RED)

            case "rodas":
                # Colisao
                rl.draw_circle(int(self.pos.x), int(self.pos.y), self.raio, rl.Color(20,100,100,255))
                # Desenha um triangulo isoceles representando o robô na posição atual
                rl.draw_poly(self.pos, 3, self.raio, (self.angulo+math.pi) * 180 / math.pi, rl.BLUE)
                pos_ponta = rl.Vector2(self.pos.x + self.raio//2 * math.cos(self.angulo), self.pos.y + self.raio//2 * math.sin(self.angulo))
                rl.draw_poly(pos_ponta, 3, self.raio//2, (self.angulo+math.pi) * 180 / math.pi, rl.RED)
                # Com duas rodas atras desenhadas como círculos, uma para cada roda
                pos_roda_esquerda = rl.Vector2(self.pos.x - self.raio//2 * math.cos(self.angulo + math.pi / 2), self.pos.y - self.raio//2 * math.sin(self.angulo + math.pi / 2))
                pos_roda_direita = rl.Vector2(self.pos.x - self.raio//2 * math.cos(self.angulo - math.pi / 2), self.pos.y - self.raio//2 * math.sin(self.angulo - math.pi / 2))
                rl.draw_circle(int(pos_roda_esquerda.x), int(pos_roda_esquerda.y), self.raio//4, rl.GREEN)
                rl.draw_circle(int(pos_roda_direita.x), int(pos_roda_direita.y), self.raio//4, rl.GREEN)

