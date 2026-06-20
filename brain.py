import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
import random
import os

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        state, action, reward, next_state = zip(*random.sample(self.buffer, batch_size))
        return torch.stack(state), torch.stack(action), torch.stack(reward), torch.stack(next_state)

    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, n_actions, space_dims, hidden_dims):
        super(Actor, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(space_dims, hidden_dims),
            nn.ReLU(True),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(True)
        )
        self.mean_linear = nn.Linear(hidden_dims, n_actions)
        self.log_std_linear = nn.Linear(hidden_dims, n_actions)
        
        self.action_scale = torch.tensor(1.0)
        self.action_bias = torch.tensor(0.0)

    def forward(self, state):
        x = self.feature_extractor(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=-20, max=2) # Evita gradientes explosivos
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        
        # Enforcing Action Bound (correção de probabilidade devido ao tanh)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

class DoubleCritic(nn.Module):
    def __init__(self, space_dims, n_actions, hidden_dims):
        super(DoubleCritic, self).__init__()
        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(space_dims + n_actions, hidden_dims),
            nn.ReLU(True),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(True),
            nn.Linear(hidden_dims, 1)
        )
        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(space_dims + n_actions, hidden_dims),
            nn.ReLU(True),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(True),
            nn.Linear(hidden_dims, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

class InverseModel(nn.Module): 
    # Adaptado para prever ações contínuas
    def __init__(self, n_actions, hidden_dims):
        super(InverseModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dims * 2, hidden_dims),
            nn.ReLU(True),
            nn.Linear(hidden_dims, n_actions)
        )

    def forward(self, features):
        action_pred = self.fc(features)
        return action_pred

class ForwardModel(nn.Module): 
    # Adaptado para receber ações contínuas
    def __init__(self, n_actions, hidden_dims):
        super(ForwardModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dims + n_actions, hidden_dims),
            nn.ReLU(True),
            nn.Linear(hidden_dims, hidden_dims)
        )

    def forward(self, action, features):
        x = torch.cat([action, features], dim=-1)
        next_features = self.fc(x)
        return next_features

class FeatureExtractor(nn.Module):
    def __init__(self, space_dims, hidden_dims):
        super(FeatureExtractor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(space_dims, hidden_dims),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.fc(x)

class ConfigArgs:
    beta = 0.2
    eta = 500.0 # Scale factor for intrinsic reward
    discounted_factor = 0.99
    lr = 3e-4 # Taxa de aprendizado padrão do SAC para todas as redes
    batch_size = 256
    buffer_capacity = 100000
    tau = 0.005 # Fator de atualização suave da rede alvo
    alpha_init = 0.2 # Temperatura inicial

class Brain():
    def __init__(self, robo, num_actions, path_to_save_models=None, len_moving_average=100):
        self.Robo = robo
        self.args = ConfigArgs()
        self.num_actions = num_actions
        self.path_to_save_models = path_to_save_models
        os.makedirs(self.path_to_save_models, exist_ok=True) # Garante que o diretório para salvar os modelos exista

        self.total_reward = 0.0
        self.reconstruction_loss = 0.0 
        self.sensors_latent_info = {}
        self.sensors_names = [] 
        
        self.current_state = None 
        self.last_state = None
        self.last_action = None # Agora guardamos a última ação contínua
        
        self.reward = 0. 
        self.need_to_initialize_models = True
        
        self.memory = ReplayBuffer(self.args.buffer_capacity)
        self.n_steps_until_save = self.args.batch_size * 3 # Salva os modelos a cada 3 batches completos
        self.warm_up_steps = 2000 # Número de etapas para aquecer o buffer de replay
        self.get_moving_average_reward_window = deque(maxlen=len_moving_average) # Janela para calcular a média móvel da recompensa

    def save_models(self):
        if self.path_to_save_models is not None:
            torch.save(self.actor.state_dict(), f"{self.path_to_save_models}actor.pth")
            torch.save(self.critic.state_dict(), f"{self.path_to_save_models}critic.pth")
            torch.save(self.feature_extractor.state_dict(), f"{self.path_to_save_models}feature_extractor.pth")
            torch.save(self.forward_model.state_dict(), f"{self.path_to_save_models}forward_model.pth")
            torch.save(self.inverse_model.state_dict(), f"{self.path_to_save_models}inverse_model.pth")
            print(f"Modelos salvos com sucesso em {self.path_to_save_models}!")

    def initialize_models(self, input_length):
        hidden_dims = 128 # Aumentado para lidar melhor com o espaço contínuo

        self.actor = Actor(self.num_actions, input_length, hidden_dims)
        
        self.critic = DoubleCritic(input_length, self.num_actions, hidden_dims)
        self.critic_target = DoubleCritic(input_length, self.num_actions, hidden_dims)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.feature_extractor = FeatureExtractor(input_length, hidden_dims)
        self.forward_model = ForwardModel(self.num_actions, hidden_dims)
        self.inverse_model = InverseModel(self.num_actions, hidden_dims)

        # load if available
        if self.path_to_save_models is not None:
            try:
                self.actor.load_state_dict(torch.load(f"{self.path_to_save_models}actor.pth"))
                self.critic.load_state_dict(torch.load(f"{self.path_to_save_models}critic.pth"))
                self.critic_target.load_state_dict(self.critic.state_dict())
                self.feature_extractor.load_state_dict(torch.load(f"{self.path_to_save_models}feature_extractor.pth"))
                self.forward_model.load_state_dict(torch.load(f"{self.path_to_save_models}forward_model.pth"))
                self.inverse_model.load_state_dict(torch.load(f"{self.path_to_save_models}inverse_model.pth"))
                print(f"Modelos carregados com sucesso de {self.path_to_save_models}!")
            except FileNotFoundError:
                print(f"Modelos não encontrados em {self.path_to_save_models}, iniciando do zero.")

        # Optimizers
        self.a_optim = optim.Adam(self.actor.parameters(), lr=self.args.lr)
        self.c_optim = optim.Adam(self.critic.parameters(), lr=self.args.lr)
        
        self.icm_params = list(self.feature_extractor.parameters()) + list(self.forward_model.parameters()) + list(self.inverse_model.parameters())
        self.icm_optim = optim.Adam(self.icm_params, lr=self.args.lr)

        # Entropia Automática (Auto-tuning Alpha)
        self.target_entropy = -torch.prod(torch.Tensor([self.num_actions])).item()
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.args.lr)
        self.alpha = self.args.alpha_init

        self.mse_loss = nn.MSELoss()

    def get_state(self):
        self.last_state = self.current_state 
        state_tensors = [self.sensors_latent_info[name] for name in self.sensors_names]
        self.current_state = torch.cat(state_tensors).flatten()

        if self.need_to_initialize_models:
            self.initialize_models(input_length=self.current_state.shape[0])
            self.need_to_initialize_models = False

    def set_info_sensor(self, info, sensor_name, encoder):
        latent_info = encoder(info)
        self.sensors_latent_info[sensor_name] = latent_info
        if sensor_name not in self.sensors_names:
            self.sensors_names.append(sensor_name)

    def add_reward(self, reward):
        self.reward += reward

    def get_total_reward(self):
        return self.total_reward

    def get_moving_average_reward(self, window_size=100):
        if len(self.get_moving_average_reward_window) == 0:
            return 0.0
        return sum(self.get_moving_average_reward_window) / len(self.get_moving_average_reward_window)
    
    def get_reconstruction_loss(self):
        return self.reconstruction_loss

    def compute_intrinsic_reward(self, state, action, next_state):
        with torch.no_grad():
            obs_cat = torch.stack([state, next_state], dim=0)
            features = self.feature_extractor(obs_cat)
            
            # Formato esperado: [1, dims]
            est_next_features = self.forward_model(action.unsqueeze(0), features[0:1])
            forward_loss = self.mse_loss(est_next_features, features[1:2])
            
        return self.args.eta * forward_loss.item()

    def update(self):
        self.get_state()
        self.n_steps_until_save -= 1
        if self.n_steps_until_save <= 0:
            self.save_models()
            self.n_steps_until_save = self.args.batch_size * 10

        # Coleta de experiências online
        if self.last_state is not None and self.last_action is not None:
            # Calcula a recompensa intrínseca na hora para armazenar no buffer
            r_int = self.compute_intrinsic_reward(self.last_state, self.last_action, self.current_state)
            self.total_reward = self.reward + r_int
            self.get_moving_average_reward_window.append(self.total_reward)
            
            self.memory.push(self.last_state.detach(), 
                             self.last_action.detach(), 
                             torch.tensor([self.total_reward], dtype=torch.float32), 
                             self.current_state.detach())

        # Seleciona nova ação para o ambiente usando o Ator
        with torch.no_grad():
            # Warm-up: Exploração 100% caótica nos primeiros 2000 passos
            if len(self.memory) < self.warm_up_steps:
                # Gera uma ação aleatória entre -1 e 1
                action = torch.rand(1, self.num_actions) * 2.0 - 1.0 
            else:
                action, _, _ = self.actor.sample(self.current_state.unsqueeze(0))
            
            self.last_action = action.squeeze(0)
        # Treinamento com Batch
        if len(self.memory) > self.args.batch_size:
            states, actions, rewards, next_states = self.memory.sample(self.args.batch_size)

            # ========================
            # Treinamento do ICM
            # ========================
            features_state = self.feature_extractor(states)
            features_next_state = self.feature_extractor(next_states)
            features_cat = torch.cat([features_state, features_next_state], dim=1)

            inverse_action_prob = self.inverse_model(features_cat)
            est_next_features = self.forward_model(actions, features_state)

            forward_loss = self.mse_loss(est_next_features, features_next_state)
            inverse_loss = self.mse_loss(inverse_action_prob, actions)
            
            icm_loss = (1 - self.args.beta) * inverse_loss + self.args.beta * forward_loss
            self.reconstruction_loss = forward_loss.item()

            self.icm_optim.zero_grad()
            icm_loss.backward()
            self.icm_optim.step()

            # ========================
            # Treinamento do SAC Crítico
            # ========================
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = self.actor.sample(next_states)
                qf1_next_target, qf2_next_target = self.critic_target(next_states, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                next_q_value = rewards + self.args.discounted_factor * (min_qf_next_target)

            qf1, qf2 = self.critic(states, actions)
            qf1_loss = self.mse_loss(qf1, next_q_value)
            qf2_loss = self.mse_loss(qf2, next_q_value)
            critic_loss = qf1_loss + qf2_loss

            self.c_optim.zero_grad()
            critic_loss.backward()
            self.c_optim.step()

            # ========================
            # Treinamento do SAC Ator
            # ========================
            pi, log_pi, _ = self.actor.sample(states)
            qf1_pi, qf2_pi = self.critic(states, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            
            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

            self.a_optim.zero_grad()
            policy_loss.backward()
            self.a_optim.step()

            # ========================
            # Atualização do Alpha (Temperatura)
            # ========================
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()

            # ========================
            # Atualização das Target Networks
            # ========================
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)


        
        self.reward = 0. # Reseta o reforço para a próxima iteração

        return self.last_action.cpu().numpy()