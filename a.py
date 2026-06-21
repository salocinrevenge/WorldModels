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

    def forward(self, x):
        features = self.feature_extractor(x)
        policy = self.actor(features)
        return policy   # Probabilidade de cada ação ser a melhor ação, dada a observação do ambiente

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
    def __init__(self, robo, input_length, num_actions):
        self.Robo = robo
        self.args = ConfigArgs()
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
    
    def train(self):
        # Treinamento do carpole, mudar para o robo atual
        global_step = 0 # Contador global de passos
        n_eps = 0       # Contador de episódios
        reward_lst = [] # Lista de recompensas por episódio
        mva_lst = []    # Lista de médias móveis das recompensas por episódio
        mva = 0.        # Média móvel das recompensas por episódio
        avg_ireward_lst = []    # Lista de recompensas intrínsecas médias por episódio

        while n_eps < self.args.max_eps:
            n_eps+=1
            next_state = self.to_tensor(env.reset(), dtype=torch.float) ### inicia o ambiente

            done = False    
            score = 0
            ireward_lst = []    # Lista de recompensas intrínsecas por episódio
            while not done: # Passos
                # Reseta estados e gradientes
                state = next_state
                self.a_optim.zero_grad()
                self.c_optim.zero_grad()
                self.icm_optim.zero_grad()

                policy = self.actor(state)

                action = self.select_action(policy.detach().numpy()[0])

                next_state_from_env, score_i, done, info = env.step(action) # Saida do ambiente após tomar a ação, incluindo o próximo estado, a recompensa extrínseca imediata, se o episódio terminou e outras informações relevantes

                next_state = self.to_tensor(next_state_from_env, dtype=torch.float)

                advantages = torch.zeros_like(policy)

                # Correctly create a tensor for the action taken, of type long
                action_tensor_for_icm = torch.tensor([action], dtype=torch.long)

                v = self.critic(state)[0]
                next_v = self.critic(next_state)[0]

                obs_cat = torch.cat([state, next_state], dim = 0)
                features = self.feature_extractor(obs_cat) # Estados no espaço latente, concatenados
                inverse_action_prob = self.inverse_model(features) # Acao tomada dado variavel latente. Acao predita

                # Pass the correct action tensor to forward_model
                est_next_features = self.forward_model(action_tensor_for_icm, features[0:1]) # Dado o espaço latente do estado atual e a ação tomada, estima o espaço latente do próximo estado
                extrinsic_reward = 0.
                forward_loss = self.mse_loss(est_next_features, features[1])

                # Use the correct action tensor as target for inverse_loss
                inverse_loss = self.xe_loss(inverse_action_prob, action_tensor_for_icm.view(-1))
                icm_loss = (1-self.args.beta)*inverse_loss + self.args.beta*forward_loss

                intrinsic_reward = self.args.eta*forward_loss.detach()

                # Immediate extrinsic reward from environment step
                immediate_extrinsic_reward = score_i

                if done:
                    # Adjusted reward logic for terminal states, combining extrinsic and intrinsic
                    if score < 499:
                        reward_for_advantages = torch.tensor([-100.0], dtype=torch.float) + intrinsic_reward
                        c_target = torch.tensor([-100.0], dtype=torch.float) + intrinsic_reward
                    else:
                        reward_for_advantages = torch.tensor([0.0], dtype=torch.float) + intrinsic_reward
                        c_target = torch.tensor([0.0], dtype=torch.float) + intrinsic_reward

                    advantages[0,action] = reward_for_advantages - v
                else:
                    total_reward_val = immediate_extrinsic_reward + intrinsic_reward
                    advantages[0,action] = total_reward_val + self.args.discounted_factor*next_v-v
                    c_target = total_reward_val + self.args.discounted_factor*next_v

                actor_loss = self.pg_loss(policy, advantages.detach())
                critic_loss = self.mse_loss(v, c_target.detach())

                loss = actor_loss + critic_loss + icm_loss

                loss.backward()
                self.icm_optim.step()
                self.a_optim.step()
                self.c_optim.step()

                if not done:
                    score += score_i

                ireward_lst.append(intrinsic_reward.item())

                global_step +=1

            #print das métricas
            avg_intrinsic_reward = sum(ireward_lst)/ len(ireward_lst)

            mva = 0.95*mva + 0.05*score
            reward_lst.append(score)
            avg_ireward_lst.append(avg_intrinsic_reward)
            mva_lst.append(mva)
            print('Episodes: {}, AVG Score: {:.3f}, Score: {} AVG reward i: {:.6f}'.format(n_eps, mva, score, avg_intrinsic_reward))



import torch
from diffusers import AutoencoderKL

# 1. Carrega o modelo pré-treinado
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").eval()

# Simulação da sua imagem original [Batch, Canais, Altura, Largura] na escala 0-255
# (Certifique-se de que o tipo seja float para as operações matemáticas)
imagem_0_255 = torch.randint(0, 256, (1, 3, 160, 160)).float()

# ========================================================
# PREPARAÇÃO (0 a 255  --->  -1 a 1)
# ========================================================
imagem_normalizada = (imagem_0_255 / 255.0) * 2.0 - 1.0


with torch.no_grad():
    # --- CODIFICA ---
    latente_2d = vae.encode(imagem_normalizada).latent_dist.mode()
    print("Shape do Espaço Latente 2D:", latente_2d.shape) # [1, 4, 20, 20]

    # --- DECODIFICA ---
    imagem_reconstruida = vae.decode(latente_2d).sample


# ========================================================
# PÓS-PROCESSAMENTO (-1 a 1  --->  0 a 255)
# ========================================================
# 1. Volta para a escala 0 a 1
imagem_final = (imagem_reconstruida + 1.0) / 2.0

# 2. Corta valores que saíram levemente do limite devido a dízimas decimais
imagem_final = torch.clamp(imagem_final, 0.0, 1.0)

# 3. Volta para a escala 0 a 255
imagem_final_255 = imagem_final * 255.0

# 4. Opcional: Se precisar salvar ou usar no OpenCV, converta de volta para inteiros
imagem_final_255 = imagem_final_255.to(torch.uint8)

print("Shape final:", imagem_final_255.shape) # [1, 3, 160, 160]
print("Valores mínimos e máximos reais:", imagem_final_255.min().item(), "a", imagem_final_255.max().item())