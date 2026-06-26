class Critic():
    def __init__(self):
        self.reset_intrinsic_value()

    def reset_intrinsic_value(self):
        self.intrinsic_value = 0.0

    def get_arrived(self, state, target, target_range):
        """
            Função para verificar se o robô chegou ao alvo definido.
            A função calcula a distância entre o estado atual do robô (representado pelo espaço latente) e o alvo.
            Se a distância for menor ou igual à faixa de tolerância, considera-se que o robô chegou ao alvo.
        """
        distance = ((state[0] - target[0]) ** 2 + (state[1] - target[1]) ** 2) ** 0.5
        if distance <= target_range:
            return True
        else:
            return False