from critic import Critic
from actor import Actor
from worldModel import WorldModel
import torch
import pyray as rl

class Brain():
    def __init__(self, robo, num_actions, frames_to_act):
        self.robo = robo
        # Estrutura da arquitetura cognitiva
        self.modelo_de_mundo = None
        if False:
            self.modelo_de_mundo = WorldModel(self, path_to_save="world_model/")
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
        self.paciencia_alvo_maxima = 400/frames_to_act
        self.paciencia_alvo_atual = 0

        # Target
        self.set_target()

        # Rewards
        self.reward=0

        self.idx_gps = None


    def percept(self):
        pass

    def update(self):
        self.paciencia_alvo_atual += 1
        real_pos = (self.robo.encoders["gps"].decode(self.value_sensors["gps"][0]), self.robo.encoders["gps"].decode(self.value_sensors["gps"][1]))
        if self.critic.get_arrived(real_pos, self.target, self.target_range):
            self.paciencia_alvo_atual = 0
            self.set_target()
            self.add_reward(1)
        else:
            if self.paciencia_alvo_atual > self.paciencia_alvo_maxima:
                self.set_target()
                self.paciencia_alvo_atual = 0

        self.latent_space = self.latent_reconstructed
        self.action = self.actor.get_action((*real_pos, self.value_sensors["compass"]), self.target)
        # Junta todos os sensores e o target em um único tensor para passar para o modelo de mundo

        state = torch.tensor([], dtype=torch.float32)
        for sensor in self.order_sensors:
            if sensor == "gps":
                self.idx_gps = len(state)
            state = torch.cat((state, torch.tensor(self.value_sensors[sensor], dtype=torch.float32)))
        state = torch.cat((state, torch.tensor([self.target[0], self.target[1]], dtype=torch.float32)))
        if self.modelo_de_mundo is not None:
            self.modelo_de_mundo.set_state(state)


        if self.modelo_de_mundo is not None:
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
        if self.modelo_de_mundo is not None:
            self.modelo_de_mundo.render()

            # Mostra onde eu estava e onde eu acho q eu estarei
            if self.idx_gps is not None and self.modelo_de_mundo.current_state is not None and self.modelo_de_mundo.reconstructed_future_state is not None:
                estou_pos_normalized = self.modelo_de_mundo.current_state.detach().numpy()[self.idx_gps:self.idx_gps+2]
                estou_pos = (self.robo.encoders["gps"].decode(estou_pos_normalized[0]).item(), self.robo.encoders["gps"].decode(estou_pos_normalized[1]).item())
                estarei_pos_normalized = self.modelo_de_mundo.reconstructed_future_state.detach().numpy()[self.idx_gps:self.idx_gps+2]
                estarei_pos = (self.robo.encoders["gps"].decode(estarei_pos_normalized[0]).item(), self.robo.encoders["gps"].decode(estarei_pos_normalized[1]).item())
                rl.draw_circle(int(estou_pos[0]), int(estou_pos[1]), 5, rl.RED)
                rl.draw_circle(int(estarei_pos[0]), int(estarei_pos[1]), 5, rl.GREEN)