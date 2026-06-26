import torch
import torch.nn as nn

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


class WorldModel():
    def __init__(self, robo):
        self.robo = robo

        self.last_state = None
        self.current_state = None
        self.last_action = None
        self.hidden_dims = 16

    def set_state(self, state):
        self.last_state = self.current_state
        self.current_state = state

    def set_action(self, action):
        self.last_action = action
        
    def init_models(self):
        self.feature_extractor = FeatureExtractor(space_dims=self.current_state.size(0), hidden_dims=self.hidden_dims)
        self.forward_model = ForwardModel(n_actions=self.last_action.size(0), hidden_dims=self.hidden_dims)
        self.inverse_model = InverseModel(n_actions=self.last_action.size(0), hidden_dims=self.hidden_dims)

    def update(self):
        if self.last_state is None or self.current_state is None or self.last_action is None:
            return
        
        self.init_models()
        # icm: intrinsic curiosity module
        # junta os parâmetros do feature extractor, forward model e inverse model em um único otimizador
        self.icm_params = list(self.feature_extractor.parameters())+ list(self.forward_model.parameters()) + list(self.inverse_model.parameters())
        self.icm_optim = torch.optim.Adam(self.icm_params, lr=self.args.lr_icm)
