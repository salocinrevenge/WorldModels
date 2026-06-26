from critic import Critic
from actor import Actor

class Brain():
    def __init__(self, robo, num_actions):
        self.robo = robo
        # Estrutura da arquitetura cognitiva
        self.modelo_de_mundo = None
        self.actor = Actor(num_actions)
        self.critic = Critic()

        # Sensores
        # Expected sensors: Vision, GPS, Compass, Accelerometer, Gyroscope, Touch, Time
        self.value_sensors = dict()
        self.order_sensors = list()
        self.latent_space = None

        # Debug do cerebro
        self.latent_reconstructed = None

        # Action
        self.action = None

        # Paciencia maxima para o robo atingir o alvo
        self.paciencia_alvo_maxima = 200
        self.paciencia_alvo_atual = 0

        # Target
        self.set_target()

        # Rewards
        self.reward=0

    def percept(self):
        pass

    def update(self):
        self.paciencia_alvo_atual += 1
        if self.critic.get_arrived(self.value_sensors["gps"], self.target, self.target_range):
            self.paciencia_alvo_atual = 0
            self.set_target()
            self.add_reward(1)
        else:
            if self.paciencia_alvo_atual > self.paciencia_alvo_maxima:
                self.set_target()
                self.paciencia_alvo_atual = 0

        self.latent_space = self.latent_reconstructed
        self.action = self.actor.get_action((*self.value_sensors["gps"], self.value_sensors["compass"]), self.target)
        return self.action


    def add_reward(self, reward):
        self.reward += reward

    def set_info_sensor(self, value, sensor, encoder=None):
        if sensor not in self.order_sensors:
            self.order_sensors.append(sensor)
        if encoder is not None:
            self.value_sensors[sensor] = encoder(value)
        else:
            self.value_sensors[sensor] = value

    def set_target(self):
        """
            Função para definir o alvo que o robô deve perseguir e a faixa de tolerância para considerar que o alvo foi atingido.
            Utilizado apenas para a definicao de tarefa simples por agora
        """
        target = self.robo.get_random_pos()
        self.target = (target.x, target.y)
        self.target_range = 10

        