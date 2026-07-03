import torch
import torch.nn as nn
import math

from math_utils import Rect
from line_graphics import LineGraphics

import os

class LatentInverseDynamics(nn.Module): # Dado estado atual e próximo estado, prediz a ação tomada
    def __init__(self, hidden_dims, action_shape, latent_variables_shape):
        super(LatentInverseDynamics, self).__init__()
        self.fc = nn.Linear(hidden_dims*2+action_shape, latent_variables_shape)

    def forward(self, features):
        action = self.fc(features) # (1, latent_variables_shape)
        return action

class Dynamics(nn.Module): # Dado 
    def __init__(self, latent_variables_shape, hidden_dims):
        super(Dynamics, self).__init__()
        self.fc = nn.Linear(hidden_dims+latent_variables_shape, hidden_dims)
        self.eye = torch.eye(latent_variables_shape)

    def forward(self, latent_variable, features):
        prediction = self.fc(torch.cat([latent_variable, features], dim=-1))
        return prediction
    
class LatentPolicy(nn.Module):
    def __init__(self, latent_variables_shape, action_shape, hidden_dims):
        super(LatentPolicy, self).__init__()
        self.fc = nn.Linear(hidden_dims + action_shape, latent_variables_shape)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, features):
        action_probs = self.softmax(self.fc(features)) # (1, latent_variables_shape)
        return action_probs

class Encoder(nn.Module):
    def __init__(self, space_dims, hidden_dims):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(space_dims, hidden_dims)

    def forward(self, x):
        y = torch.tanh(self.fc(x))
        return y
    
class Decoder(nn.Module):
    def __init__(self, space_dims, hidden_dims):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(hidden_dims, space_dims)

    def forward(self, x):
        y = torch.tanh(self.fc(x))
        return y


class WorldModel():
    def __init__(self, brain, path_to_save=None, iterations_to_save=10, space_to_draw_graphics = Rect(1010, 250, 380, 300)):
        self.brain = brain
        self.path_to_save = path_to_save
        self.iterations_to_save = iterations_to_save
        self.space_to_draw_graphics = space_to_draw_graphics

        self.last_state = None
        self.current_state = None
        self.last_action = None
        self.current_action = None
        self.predicted_latent_space = None
        self.hidden_dims = 16
        self.latent_variables_shape = 16
        self.lr = 1e-3
        self.mse_inverse_loss = None
        self.mse_state_loss = None
        self.complete_loss = None
        self.batch_size = 16
        self.batch_counter = 0
        self.initialized = False
        self.iterations = 0
        self.reconstructed_future_state = None

    def set_state(self, state):
        self.last_state = self.current_state
        self.current_state = state

    def set_action(self, action):
        self.last_action = self.current_action
        self.current_action = action

    def init_models(self):
        self.encoder= Encoder(space_dims=len(self.current_state), hidden_dims=self.hidden_dims)
        self.dynamics_model = Dynamics(latent_variables_shape=self.latent_variables_shape, hidden_dims=self.hidden_dims)
        self.inverse_model = LatentInverseDynamics(latent_variables_shape=self.latent_variables_shape, action_shape=len(self.last_action), hidden_dims=self.hidden_dims)
        self.latent_policy_model = LatentPolicy(hidden_dims=self.hidden_dims, action_shape=len(self.last_action), latent_variables_shape=self.latent_variables_shape)
        self.decoder = Decoder(space_dims=len(self.current_state), hidden_dims=self.hidden_dims)
        self.params = list(self.encoder.parameters())+ list(self.dynamics_model.parameters()) + list(self.inverse_model.parameters()) + list(self.latent_policy_model.parameters()) + list(self.decoder.parameters())
        self.optim = torch.optim.Adam(self.params, lr=self.lr)
        self.mse_loss = nn.MSELoss()
        if self.path_to_save is not None and os.path.exists(self.path_to_save):
            # Para cada rede, carregue os pesos salvos
            try:
                self.encoder.load_state_dict(torch.load(os.path.join(self.path_to_save, 'encoder.pth')))
                self.dynamics_model.load_state_dict(torch.load(os.path.join(self.path_to_save, 'dynamics_model.pth')))
                self.inverse_model.load_state_dict(torch.load(os.path.join(self.path_to_save, 'inverse_model.pth')))
                self.latent_policy_model.load_state_dict(torch.load(os.path.join(self.path_to_save, 'latent_policy_model.pth')))
                self.decoder.load_state_dict(torch.load(os.path.join(self.path_to_save, 'decoder.pth')))
            except Exception as e:
                print(f"Erro ao carregar os pesos salvos: {e}")
        else:
            print("Os pesos não foram carregados, pois o diretório de salvamento não existe ou não foi especificado.")
            if self.path_to_save is not None:
                print("Diretório de salvamento não existe. Criando novo diretório.")
                os.makedirs(self.path_to_save, exist_ok=True)
            

    def update_graphics(self):
        if self.mse_inverse_loss is not None and self.mse_state_loss is not None:
            if not hasattr(self, 'line_graphics'):
                self.line_graphics = LineGraphics(format=self.space_to_draw_graphics, name_x="Iterations", name_y="Loss\n(Log)", title="World Model Losses", colors={"Latent variable loss": (255, 0, 0, 255), "State loss": (0, 0, 255, 255), "Complete loss": (0, 255, 0, 255)})
            self.line_graphics.add_data(math.log10(self.mse_inverse_loss.item()), tag="Latent variable loss")
            self.line_graphics.add_data(math.log10(self.mse_state_loss.item()), tag="State loss")
            self.line_graphics.add_data(math.log10(self.complete_loss.item()), tag="Complete loss")

            

    def update(self):
        """
        Treina com o estado atual e o estado anterior, e a ação tomada entre eles. Prevê o estado futuro.
        
        """
        if self.last_state is None or self.current_state is None or self.last_action is None:
            return
        self.iterations += 1
        
        if not self.initialized:
            self.init_models()
            self.initialized = True

        latent_space_last = self.encoder(self.last_state)
        latent_space_current = self.encoder(self.current_state)

        # Treinamento do decoder
        reconstructed_current_state = self.decoder(latent_space_current.detach())
        erro_reconstructed = self.mse_loss(reconstructed_current_state, self.current_state)
        erro_reconstructed.backward()

        # Treinamento da Predicao
        predicted_latent_variable_1 = self.inverse_model(torch.cat([latent_space_last, latent_space_current, self.last_action], dim=-1))
        predicted_latent_variable_2 = self.latent_policy_model(torch.cat([latent_space_last, self.last_action], dim=-1))
        self.mse_inverse_loss = self.mse_loss(predicted_latent_variable_1, predicted_latent_variable_2)
        self.predicted_latent_space = self.dynamics_model(predicted_latent_variable_1, latent_space_last)
        self.mse_state_loss = self.mse_loss(self.predicted_latent_space, latent_space_current)
        self.complete_loss = self.mse_state_loss + self.mse_inverse_loss
        self.complete_loss.backward()
        self.optim.step()            
        self.optim.zero_grad()

        self.update_graphics()

        # Save model
        if self.iterations % self.iterations_to_save == 0 and self.path_to_save is not None:
            torch.save(self.encoder.state_dict(), os.path.join(self.path_to_save, 'encoder.pth'))
            torch.save(self.dynamics_model.state_dict(), os.path.join(self.path_to_save, 'dynamics_model.pth'))
            torch.save(self.inverse_model.state_dict(), os.path.join(self.path_to_save, 'inverse_model.pth'))
            torch.save(self.latent_policy_model.state_dict(), os.path.join(self.path_to_save, 'latent_policy_model.pth'))
            torch.save(self.decoder.state_dict(), os.path.join(self.path_to_save, 'decoder.pth'))


        # Faz predição do próximo estado com base no estado atual
        self.latent_policy_model.eval()
        self.dynamics_model.eval()
        self.decoder.eval()
        with torch.no_grad():
            predicted_latent_variable_current = self.latent_policy_model(torch.cat([latent_space_current, self.current_action], dim=-1))
            predicted_future_latent_space = self.dynamics_model(predicted_latent_variable_current, latent_space_current)
            self.reconstructed_future_state = self.decoder(predicted_future_latent_space)

        # Religa os gradientes para continuar o treinamento
        self.latent_policy_model.train()
        self.dynamics_model.train()

        return self.reconstructed_future_state



    def render(self):
        if hasattr(self, 'line_graphics'):
            self.line_graphics.render()

