from critic import Critic
from actor import Actor
from worldModel import WorldModel
import torch

class Brain():
    def __init__(self, robo, num_actions, frames_to_act):
        self.robo = robo
        # Estrutura da arquitetura cognitiva
        self.modelo_de_mundo = WorldModel(robo, path_to_save="world_model/")
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
        self.paciencia_alvo_maxima = 300/frames_to_act
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
        # Junta todos os sensores e o target em um único tensor para passar para o modelo de mundo

        state = torch.tensor([], dtype=torch.float32)
        for sensor in self.order_sensors:
            state = torch.cat((state, torch.tensor(self.value_sensors[sensor], dtype=torch.float32)))
        state = torch.cat((state, torch.tensor([self.target[0], self.target[1]], dtype=torch.float32)))
        self.modelo_de_mundo.set_state(state)


        # self.modelo_de_mundo.set_state(torch.tensor((*self.value_sensors["gps"], self.value_sensors["compass"]), dtype=torch.float32))
        self.modelo_de_mundo.set_action(torch.tensor(self.action, dtype=torch.float32))
        self.modelo_de_mundo.update()
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

        
    def render(self):
        self.modelo_de_mundo.render()