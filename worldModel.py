import torch
import torch.nn as nn

class LatentInverseDynamics(nn.Module): # Dado estado atual e próximo estado, prediz a ação tomada
    def __init__(self, hidden_dims, action_shape, latent_variables_shape):
        super(LatentInverseDynamics, self).__init__()
        self.fc = nn.Linear(hidden_dims*2+action_shape, latent_variables_shape)

    def forward(self, features):
        features = features.view(1, -1) # (1, hidden_dims)
        action = self.fc(features) # (1, latent_variables_shape)
        return action

class Dynamics(nn.Module): # Dado 
    def __init__(self, latent_variables_shape, hidden_dims):
        super(Dynamics, self).__init__()
        self.fc = nn.Linear(hidden_dims+latent_variables_shape, hidden_dims)
        self.eye = torch.eye(latent_variables_shape)

    def forward(self, action, features):
        x = torch.cat([self.eye[action], features], dim=-1) # (1, latent_variables_shape+hidden_dims)
        features = self.fc(x) # (1, hidden_dims)
        return features
    
class LatentPolicy(nn.Module):
    def __init__(self, latent_variables_shape, action_shape, hidden_dims):
        super(LatentPolicy, self).__init__()
        self.fc = nn.Linear(hidden_dims + action_shape, latent_variables_shape)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, features):
        features = features.view(1, -1) # (1, hidden_dims)
        action_probs = self.softmax(self.fc(features)) # (1, latent_variables_shape)
        return action_probs

class Encoder(nn.Module):
    def __init__(self, space_dims, hidden_dims):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(space_dims, hidden_dims)

    def forward(self, x):
        y = torch.tanh(self.fc(x))
        return y


class WorldModel():
    def __init__(self, robo):
        self.robo = robo

        self.last_state = None
        self.current_state = None
        self.last_action = None
        self.predicted_latent_space = None
        self.hidden_dims = 16
        self.latent_variables_shape = 16
        self.lr_icm = 1e-3

    def set_state(self, state):
        self.last_state = self.current_state
        self.current_state = state

    def set_action(self, action):
        self.last_action = action
        
    def init_models(self):
        self.feature_extractor = Encoder(space_dims=self.current_state.size(0), hidden_dims=self.hidden_dims)
        self.dynamics_model = Dynamics(latent_variables_shape=self.latent_variables_shape, hidden_dims=self.hidden_dims)
        self.inverse_model = LatentInverseDynamics(latent_variables_shape=self.latent_variables_shape, action_shape=self.last_action.size(0), hidden_dims=self.hidden_dims)
        self.latent_policy_model = LatentPolicy(hidden_dims=self.hidden_dims, action_shape=self.last_action.size(0), latent_variables_shape=self.latent_variables_shape)
        self.icm_params = list(self.feature_extractor.parameters())+ list(self.dynamics_model.parameters()) + list(self.inverse_model.parameters() + list(self.latent_policy_model.parameters()))
        self.icm_optim = torch.optim.Adam(self.icm_params, lr=self.lr_icm)
        self.mse_loss = nn.MSELoss()

    def update(self):
        if self.last_state is None or self.current_state is None or self.last_action is None:
            return
        
        self.init_models()

        if self.predicted_latent_space is not None and self.mse_inverse_loss is not None:
            mse_state_loss = self.mse_loss(self.predicted_latent_space, self.current_state)
            loss_somadas = mse_state_loss + self.mse_inverse_loss
            loss_somadas.backward()
            self.icm_optim.step()            

        self.icm_optim.zero_grad()
        
        latent_space_last = self.feature_extractor(self.last_state)
        latent_space_current = self.feature_extractor(self.current_state)

        predicted_latent_variable_1 = self.inverse_model(torch.cat([latent_space_last, latent_space_current, self.last_action], dim=-1))
        predicted_latent_variable_2 = self.latent_policy_model(torch.cat([latent_space_last, self.last_action], dim=-1))
        self.mse_inverse_loss = self.mse_loss(predicted_latent_variable_1, predicted_latent_variable_2)
        self.predicted_latent_space = self.dynamics_model(predicted_latent_variable_1, latent_space_last)


