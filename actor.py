import math
import torch

class Actor():
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.action = None

    def get_action(self, state, target):
        """
            Função para obter a ação a ser tomada pelo ator, dado o estado atual do ambiente.
            A acao é o movimento das rodas, -1 a 1 para a roda esquerda e -1 a 1 para a roda direita, onde -1 é para trás, 0 é parado e 1 é para frente.
            state: (x,y,theta) posição e orientação do robô
            target: (x,y) posição do alvo
        """
        left_wheel = 0.0
        right_wheel = 0.0

        # Correcao angular
        delta_x = target[0] - state[0]+ 0.00001 # Adiciona um pequeno valor para evitar divisão por zero
        delta_y = target[1] - state[1]
        angle_to_target = math.atan2(delta_y, delta_x)

        angle_to_correct = angle_to_target - (state[2]-math.pi)
        # se o angulo for maior que 180 graus, vamos subtrair 360 graus para ficar entre -180 e 180 graus
        if angle_to_correct > math.pi:
            angle_to_correct -= 2 * math.pi
        elif angle_to_correct < -math.pi:
            angle_to_correct += 2 * math.pi

        # Se o abs do angulo for menor que 10 graus, vamos considerar que o robo esta apontando para o alvo, e vamos seguir em frente
        if abs(angle_to_correct) > math.radians(10):
            # Se o angulo for positivo, vamos girar para a esquerda, se for negativo, vamos girar para a direita
            power = min(max(abs(angle_to_correct) / math.pi, 0.5), 1.0)
            if angle_to_correct < 0:
                left_wheel = -1.0 * power
                right_wheel = 1.0 * power
            else:
                left_wheel = 1.0 * power
                right_wheel = -1.0 * power
        else:
            # Se o robo estiver apontando para o alvo, vamos seguir em frente
            distance_to_target = math.sqrt(delta_x**2 + delta_y**2)
            power = min(distance_to_target / 10.0, 1.0)
            left_wheel = 1.0 * power
            right_wheel = 1.0 * power

        return torch.tensor([left_wheel, right_wheel])

        