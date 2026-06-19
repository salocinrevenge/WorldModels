import numpy as np
import torch
import torch.nn as nn
import torch.optim

class Actor(nn.Module):
    def __init__(self, n_actions, space_dims, hidden_dims):
        super(Actor, self).__init__()
        self.feature_extractor = nn.Sequential( # Encoder
            nn.Linear(space_dims, hidden_dims),
            nn.ReLU(True),
        )
        self.actor = nn.Sequential( # Politica
            nn.Linear(hidden_dims, n_actions),
            nn.Softmax(dim=-1),
        )
        self.current_policy = None

    def forward(self, x):
        features = self.feature_extractor(x)
        policy = self.actor(features)
        self.current_policy = policy # Probabilidade de cada ação ser a melhor ação, dada a observação do ambiente
        self.current_action = np.random.choice(len(policy), 1, p=policy)[0] # Escolhe uma ação com base na distribuição de probabilidade dada pela política
        return self.current_action
    
class Critic(nn.Module):
    def __init__(self, space_dims, hidden_dims):
        super(Critic, self).__init__()
        self.feature_extractor = nn.Sequential( # Encoder
            nn.Linear(space_dims, hidden_dims),
            nn.ReLU(True),
        )
        self.critic = nn.Linear(hidden_dims, 1) # Valor do estado

    def forward(self, x):
        features = self.feature_extractor(x)
        est_reward = self.critic(features)
        return est_reward # estimated reward
    

class InverseModel(nn.Module): # Dado estado atual e próximo estado, prediz a ação tomada
    def __init__(self, n_actions, hidden_dims):
        super(InverseModel, self).__init__()
        self.fc = nn.Linear(hidden_dims*2, n_actions)

    def forward(self, features):
        features = features.view(1, -1) # (1, hidden_dims)
        action = self.fc(features) # (1, n_actions)
        return action

class ForwardModel(nn.Module): # Dado 
    def __init__(self, n_actions, hidden_dims):
        super(ForwardModel, self).__init__()
        self.fc = nn.Linear(hidden_dims+n_actions, hidden_dims)
        self.eye = torch.eye(n_actions)

    def forward(self, action, features):
        x = torch.cat([self.eye[action], features], dim=-1) # (1, n_actions+hidden_dims)
        features = self.fc(x) # (1, hidden_dims)
        return features

class FeatureExtractor(nn.Module):
    def __init__(self, space_dims, hidden_dims):
        super(FeatureExtractor, self).__init__()
        self.fc = nn.Linear(space_dims, hidden_dims)

    def forward(self, x):
        y = torch.tanh(self.fc(x))
        return y



class PGLoss(nn.Module): # policy gradient loss
    def __init__(self):
        super(PGLoss, self).__init__()

    def forward(self, action_prob, reward):
        loss = -torch.mean(torch.log(action_prob+1e-6)*reward)
        return loss
    

class ConfigArgs:
    beta = 0.2
    lamda = 0.1
    eta = 100.0 # scale factor for intrinsic reward
    discounted_factor = 0.99
    lr_critic = 0.005
    lr_actor = 0.001
    lr_icm = 0.001
    max_eps = 1000
    sparse_mode = True



class Brain():
    def __init__(self, robo, num_actions):
        self.Robo = robo
        self.args = ConfigArgs()
        self.num_actions = num_actions

        self.sensors_latent_info = {} # Dicionário para armazenar as informações dos sensores no espaço latente, onde a chave é o nome do sensor e o valor é o tensor do espaço latente correspondente
        self.sensors_names = [] # Lista para armazenar os nomes dos sensores, para manter a ordem e facilitar o acesso aos tensores no espaço latente durante a concatenação
        self.current_state = None # Estado atual concatenado de sensores, que será usado pelo ator e crítico. Será atualizado a cada passo do ambiente chamando o método get_state(), que concatena os espaços latentes dos sensores na ordem definida por self.sensors_names.
        self.last_state = None # Estado anterior concatenado de sensores, usado para calcular a recompensa intrínseca com base na diferença entre o estado atual e o estado anterior no espaço latente. Será atualizado a cada passo do ambiente, após calcular a recompensa intrínseca, para que na próxima iteração o estado atual se torne o estado anterior.
        self.reforco = 0. # Recompensa extrínseca fornecida pelo agente, que será usada para calcular a vantagem e atualizar o crítico e o ator. Será atualizada a cada passo do ambiente chamando o método set_reforco().
        self.need_to_initialize_models = True # Flag para indicar se os modelos do ator, crítico e ICM precisam ser inicializados. Será definida como False após a primeira chamada ao método update(), que é onde os modelos são inicializados com base no tamanho do estado concatenado dos sensores.


    def initialize_models(self, input_length):
        num_actions = self.num_actions
        self.actor = Actor(n_actions=num_actions, space_dims=input_length, hidden_dims=32) # space dims é o tamanho do espaço de observação
        self.critic = Critic(space_dims=input_length, hidden_dims=32)

        self.feature_extractor = FeatureExtractor(input_length, hidden_dims=32)
        self.forward_model = ForwardModel(num_actions,32)
        self.inverse_model = InverseModel(num_actions,32)


        # Actor Critic optimizers
        self.a_optim = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr_actor)
        self.c_optim = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr_critic)

        # icm: intrinsic curiosity module
        # junta os parâmetros do feature extractor, forward model e inverse model em um único otimizador
        self.icm_params = list(self.feature_extractor.parameters())+ list(self.forward_model.parameters()) + list(self.inverse_model.parameters())
        self.icm_optim = torch.optim.Adam(self.icm_params, lr=self.args.lr_icm)

        self.pg_loss = PGLoss()
        self.mse_loss = nn.MSELoss()
        self.xe_loss = nn.CrossEntropyLoss()

    def select_action(self, policy):
        return np.random.choice(len(policy), 1, p=policy)[0] # Escolhe uma ação com base na distribuição de probabilidade dada pela política

    def to_tensor(self, x, dtype=None):
        return torch.tensor(x, dtype=dtype).unsqueeze(0)
    
    def get_state(self):
        """Concatena as informações dos sensores no espaço latente
         para formar o estado final que será usado pelo ator e crítico.
         A ordem de concatenação é determinada pela lista self.sensors_names,
         garantindo que a mesma ordem seja mantida em cada passo do ambiente."""
        self.last_state = self.current_state # Atualiza o estado anterior antes de calcular o novo estado
        state_tensors = [self.sensors_latent_info[name] for name in self.sensors_names]
        self.current_state = torch.stack(state_tensors)

        if self.need_to_initialize_models:
            self.initialize_models(input_length=self.current_state.shape[0]*self.current_state.shape[1]) # O input length é o tamanho do estado concatenado dos sensores, que é o número de sensores vezes o tamanho do espaço latente de cada sensor
            self.need_to_initialize_models = False

    def set_info_sensor(self, info, sensor_name, encoder):
        """
        Deve ser chamado pelo agente para fornecer as informações
        dos sensores ao cérebro.
        Recebe as informações do sensor, o nome do sensor e 
        um encoder para processar os dados do sensor.
        O resultado é um espaço latente de shape (hidden_dims,)
        que representa a observação do ambiente a partir daquele
        sensor específico.
        """

        latent_info = encoder(info)
        self.sensors_latent_info[sensor_name] = latent_info
        if sensor_name not in self.sensors_names:
            self.sensors_names.append(sensor_name)

    def set_reforco(self, reforco):
        """Deve ser chamado pelo agente para fornecer a recompensa extrínseca
        ao cérebro. A recompensa extrínseca é usada para calcular a vantagem
        e atualizar o crítico e o ator."""
        self.reforco = reforco

    def update(self):
        """Faz o ciclo completo"""
        self.get_state() # Atualiza o estado concatenando os espaços latentes dos sensores
        self.actor(self.current_state) # Passa o estado atualizado pelo ator para obter a política de ações

        advantages = torch.zeros_like(self.actor.current_policy) # Critico e actor
        action_tensor_for_icm = torch.tensor([self.actor.current_action], dtype=torch.long)
        last_v = self.critic(self.last_state)[0]
        current_v = self.critic(self.current_state)[0]
        obs_cat = torch.cat([self.last_state, self.current_state], dim = 0)

        # Modelo de mundo
        features = self.feature_extractor(obs_cat) # Estados no espaço latente, concatenados
        inverse_action_prob = self.inverse_model(features) # Acao tomada dado variavel latente. Acao predita
        est_next_features = self.forward_model(action_tensor_for_icm, features[0:1]) # Dado o espaço latente do estado atual e a ação tomada, estima o espaço latente do próximo estado
        forward_loss = self.mse_loss(est_next_features, features[1])

        # Use the correct action tensor as target for inverse_loss
        inverse_loss = self.xe_loss(inverse_action_prob, action_tensor_for_icm.view(-1))
        icm_loss = (1-self.args.beta)*inverse_loss + self.args.beta*forward_loss

        # Recompensas
        intrinsic_reward = self.args.eta*forward_loss.detach()
        immediate_extrinsic_reward = self.reforco
        total_reward_val = immediate_extrinsic_reward + intrinsic_reward

        advantages[0,self.actor.current_action] = total_reward_val + self.args.discounted_factor * current_v - last_v
        c_target = total_reward_val + self.args.discounted_factor * current_v

        # Losses
        actor_loss = self.pg_loss(self.actor.current_policy, advantages.detach())
        critic_loss = self.mse_loss(last_v, c_target.detach())
        loss = actor_loss + critic_loss + icm_loss

        loss.backward()
        self.icm_optim.step()
        self.a_optim.step()
        self.c_optim.step()

        self.action = self.actor.current_action
        return self.action