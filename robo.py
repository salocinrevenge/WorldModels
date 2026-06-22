import math

import pyray as rl
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import os

from brain import Brain

class Robo():
    def __init__(self, world=None, path_to_save_models = None):
        self.pos = rl.Vector2(500, 500)
        self.pos_alvo = rl.Vector2(500, 500)
        self.vel = rl.Vector2(0, 0)
        self.path_to_save_models = path_to_save_models
        self.acc = rl.Vector2(0, 0)
        self.atrito = 0.1
        self.limiar_atrito = 0.01
        self.acc_max = 0.5
        self.angulo = 0
        self.vel_angular = 0
        self.acc_angular = 0
        self.atrito_angular = 0.1
        self.limiar_atrito_angular = 0.001
        self.acc_max_angular = 0.01 # Em radianos por frame
        self.world = world
        self.raio = 20
        self.raio_visao = 80
        self.tamanho_quadrado_visao = self.raio_visao * 2
        self.raio_tato = 30

        self.tipos_controle = ["absoluto", "rotacional", "rodas"]
        self.controle_atual = 2

        self.tipos_sensores = ["visao", "tato", "acc", "gyro", "gps"]
        self.sensores_ativos = [True, True, True, True, True]
        self.tato = {"N": 0, "S": 0, "E": 0, "W": 0}
        self.ordem_tato = ["N", "S", "E", "W"]

        # Cria um diretório para salvar os frames capturados, se não existir
        if not os.path.exists("capturas_visao"):
            os.makedirs("capturas_visao")
        self.frame_counter = 0

        self.delay_visao = 0 # segundos de delay para o sensor de visão, para testar o delay na percepção do robô
        self.delay_visao_atual = 0

        self.export_vision = False # Se True, exporta a visão como PNG e Tensor, se False, só gera o Tensor sem salvar o PNG (para evitar poluir a pasta de capturas durante testes)

        # NOVA TEXTURA FIXA NA GPU: Aloca apenas uma vez na inicialização
        self.vision_tex = rl.load_render_texture(self.tamanho_quadrado_visao, self.tamanho_quadrado_visao)

        self.last_pos = [rl.Vector2(self.pos.x, self.pos.y), rl.Vector2(self.pos.x, self.pos.y), rl.Vector2(self.pos.x, self.pos.y)]
        self.last_angulo = [self.angulo, self.angulo]

        self.brain = Brain(self, num_actions=2, path_to_save_models=self.path_to_save_models)
        self.create_encoders()
        self.autonomous = True # Se True, o robô é controlado pelo cérebro, se False, é controlado pelo usuário
        self.recompensas_terreno = {"G": 1, "B": -10}
        self.limiar_dor = 5 # Se o tato for maior que isso, recebe recompensa negativa igual a quanto ele está mais que o limiar
        self.last_action = None
        self.imagem_reconstruida = None

    def create_encoders(self):

        class LightMobileDecoder(nn.Module):
            def __init__(self, latent_dim=960, output_channels=3):
                super(LightMobileDecoder, self).__init__()
                
                # 1. Projeta o vetor latente para o mapa inicial de 5x5
                # Reduzimos drasticamente a quantidade de canais de 960 para 256 para economizar parâmetros
                self.fc = nn.Sequential(
                    nn.Linear(latent_dim, 256 * 5 * 5),
                    nn.ReLU(inplace=True)
                )
                
                # 2. Camadas de Upsampling Ultra Leves
                # Passando de 5x5 -> 10x10 -> 20x20 -> 40x40 -> 80x80 -> 160x160
                # Mantendo os canais baixos para ser rápido e economizar memória
                self.decoder = nn.Sequential(
                    # Entrada: [Batch, 256, 5, 5]
                    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # -> [Batch, 128, 10, 10]
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    
                    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> [Batch, 64, 20, 20]
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    
                    nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # -> [Batch, 32, 40, 40]
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    
                    nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),   # -> [Batch, 16, 80, 80]
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True),
                    
                    nn.ConvTranspose2d(16, output_channels, kernel_size=4, stride=2, padding=1) # -> [Batch, 3, 160, 160]
                    # Nota: Tiramos o Sigmoid/Tanh daqui porque suas imagens são 0-255. 
                    # Faremos a restrição de valores de forma mais eficiente no bloco forward.
                )

            def forward(self, latent_space):
                # Garante a dimensão correta para batches
                if len(latent_space.shape) == 1:
                    latent_space = latent_space.unsqueeze(0)
                    
                # Transforma o vetor de 960 no mapa de 256 canais x 5x5
                x = self.fc(latent_space)
                x = x.view(-1, 256, 5, 5)
                
                # Executa as convoluções de upsampling
                x = self.decoder(x)
                
                # Como suas imagens originais são 0-255:
                # Usamos o torch.clamp para garantir que nenhum pixel reconstruído saia desse limite
                decoded_image = torch.clamp(x, 0.0, 255.0)
                
                return decoded_image

        class MobileEncoder: # Só vou usar isso aqui, acho q ta ok
            def __init__(self, train_decode_while_encoding = True):
                self.model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
                self.feature_extractor = self.model.features
                self.avgpool = self.model.avgpool
                self.decoder = LightMobileDecoder(latent_dim=960, output_channels=3)
                self.train_decode_while_encoding = train_decode_while_encoding
                # MSE loss e otimizador para treinar o decoder
                self.mse_loss = nn.MSELoss()
                self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=1e-4)

            def __call__(self, x):
                orig_x = x.clone() # Mantém uma cópia da entrada original para o treinamento do decoder, se necessário
                with torch.no_grad():
                    if len(x.shape)==3:
                        # se tem 4 canais, remove o canal alpha
                        if x.shape[0] == 4:
                            x = x[0:3, :, :]
                        # Adiciona uma dimensão de batch
                        x = x.unsqueeze(0)
                    x = self.feature_extractor(x)
                    latent_space = torch.squeeze(self.avgpool(x))
                
                if self.train_decode_while_encoding:
                    # Zera o gradiente:
                    self.decoder.zero_grad()
                    # Treina o decoder para reconstruir a imagem original a partir do espaço latente, usando a entrada original como alvo
                    reconstructed_image = self.decode(latent_space)
                    loss_fn = nn.MSELoss()
                    loss = loss_fn(reconstructed_image, orig_x.unsqueeze(0))
                    loss.backward() # Isso atualiza os pesos do decoder para melhorar a reconstrução ao longo do tempo
                    self.decoder_optimizer.step()
                    self.decoder_optimizer.zero_grad()
                return latent_space
            
            def decode(self, latent_vector):
                # Usa o decoder para reconstruir a imagem a partir do espaço latente
                reconstructed_image = self.decoder(latent_vector)
                # adiciona um canal alpha cheio de 1.0 para garantir que a imagem reconstruída tenha o mesmo formato da imagem original (4 canais RGBA)
                reconstructed_image = torch.cat([reconstructed_image, torch.ones_like(reconstructed_image[:, :1, :, :])], dim=1)
                return reconstructed_image

        class IdentityEncoder:
            def __init__(self, ordem = None):
                self.ordem = ordem 

            def __call__(self, x):
                if isinstance(x, torch.Tensor):
                    return x.flatten() # Achata o tensor para garantir que seja unidimensional
                if hasattr(x, 'x') and hasattr(x, 'y'):
                    return torch.tensor([x.x, x.y])
                if isinstance(x, dict):
                    inputs = []
                    for key in self.ordem:
                        inputs.append(x[key])
                    return torch.tensor(inputs)
                return torch.tensor([x]) # Converte para tensor, caso não seja nenhum dos tipos acima

        self.encoders = {
            "visao": MobileEncoder(),
            "tato": IdentityEncoder(ordem=self.ordem_tato),
            "acc": IdentityEncoder(),
            "gyro": IdentityEncoder(),
            "gps": IdentityEncoder()
        }
        

    def controles(self):
        match self.tipos_controle[self.controle_atual]:
            case "absoluto":

                if self.autonomous:
                    brain_output = self.brain.update()
                    self.last_action = brain_output
                    self.acc.x = brain_output[0].item() * self.acc_max
                    self.acc.y = brain_output[1].item() * self.acc_max
                    
                else:
                    # Se precionar as setas, o robô se move na direção correspondente
                    if (rl.is_key_down(rl.KEY_RIGHT) or rl.is_key_down(rl.KEY_D)) and (rl.is_key_down(rl.KEY_LEFT) or rl.is_key_down(rl.KEY_A)):
                        self.acc.x = 0
                    elif rl.is_key_down(rl.KEY_RIGHT) or rl.is_key_down(rl.KEY_D):
                        self.acc.x = self.acc_max
                    elif rl.is_key_down(rl.KEY_LEFT) or rl.is_key_down(rl.KEY_A):
                        self.acc.x = -self.acc_max
                    else:
                        self.acc.x = 0

                    if (rl.is_key_down(rl.KEY_DOWN) or rl.is_key_down(rl.KEY_S)) and (rl.is_key_down(rl.KEY_UP) or rl.is_key_down(rl.KEY_W)):
                        self.acc.y = 0
                    elif rl.is_key_down(rl.KEY_DOWN) or rl.is_key_down(rl.KEY_S):
                        self.acc.y = self.acc_max
                    elif rl.is_key_down(rl.KEY_UP) or rl.is_key_down(rl.KEY_W):
                        self.acc.y = -self.acc_max
                    else:            
                        self.acc.y = 0

            case "rotacional":
                if self.autonomous:
                    brain_output = self.brain.update()
                    self.last_action = brain_output
                    self.acc.x = self.acc_max * math.cos(self.angulo) * brain_output[0].item()
                    self.acc.y = self.acc_max * math.sin(self.angulo) * brain_output[0].item()
                    self.acc_angular = brain_output[1].item() * self.acc_max_angular
                else:
                    # Setas para os lados rotacionam o robô, setas para cima e para baixo movem o robô para frente e para trás na direção que ele está virado
                    if (rl.is_key_down(rl.KEY_RIGHT) or rl.is_key_down(rl.KEY_D)) and (rl.is_key_down(rl.KEY_LEFT) or rl.is_key_down(rl.KEY_A)):
                        self.acc_angular = 0
                    elif rl.is_key_down(rl.KEY_RIGHT) or rl.is_key_down(rl.KEY_D):
                        self.acc_angular = self.acc_max_angular
                    elif rl.is_key_down(rl.KEY_LEFT) or rl.is_key_down(rl.KEY_A):
                        self.acc_angular = -self.acc_max_angular
                    else:
                        self.acc_angular = 0

                    if (rl.is_key_down(rl.KEY_DOWN) or rl.is_key_down(rl.KEY_S)) and (rl.is_key_down(rl.KEY_UP) or rl.is_key_down(rl.KEY_W)):
                        self.acc.x = 0
                        self.acc.y = 0
                    elif rl.is_key_down(rl.KEY_DOWN) or rl.is_key_down(rl.KEY_S):
                        self.acc.x = self.acc_max * math.cos(self.angulo)
                        self.acc.y = self.acc_max * math.sin(self.angulo)
                    elif rl.is_key_down(rl.KEY_UP) or rl.is_key_down(rl.KEY_W):
                        self.acc.x = -self.acc_max * math.cos(self.angulo)
                        self.acc.y = -self.acc_max * math.sin(self.angulo)
                    else:            
                        self.acc.x = 0
                        self.acc.y = 0

            case "rodas":
                if self.autonomous:
                    brain_output = self.brain.update()
                    self.last_action = brain_output
                    acc_roda_esquerda = -brain_output[0].item() * self.acc_max
                    acc_roda_direita = -brain_output[1].item() * self.acc_max
                else:
                    # Usa seta cima e baixo para controlar a roda direita, e W e S para controlar a roda esquerda, o robô se move na direção correspondente a diferença de velocidade entre as rodas
                    if (rl.is_key_down(rl.KEY_W)) and (rl.is_key_down(rl.KEY_S)):
                        acc_roda_esquerda = 0
                    elif rl.is_key_down(rl.KEY_W):
                        acc_roda_esquerda = -self.acc_max
                    elif rl.is_key_down(rl.KEY_S):
                        acc_roda_esquerda = self.acc_max
                    else:
                        acc_roda_esquerda = 0

                    if (rl.is_key_down(rl.KEY_UP)) and (rl.is_key_down(rl.KEY_DOWN)):
                        acc_roda_direita = 0
                    elif rl.is_key_down(rl.KEY_UP):
                        acc_roda_direita = -self.acc_max
                    elif rl.is_key_down(rl.KEY_DOWN):
                        acc_roda_direita = self.acc_max
                    else:
                        acc_roda_direita = 0

                self.acc.x = (acc_roda_esquerda + acc_roda_direita) * math.cos(self.angulo) / 2
                self.acc.y = (acc_roda_esquerda + acc_roda_direita) * math.sin(self.angulo) / 2
                self.acc_angular = (acc_roda_direita - acc_roda_esquerda) * self.acc_max_angular




    def phisics(self, dt): # Precisa adicionar o dt, mas agora vai ser mt dificil entao n to usando ele
        # Atualiza a velocidade e a posição do robô com base na aceleração
        self.vel.x += self.acc.x 
        self.vel.y += self.acc.y

        self.pos_alvo.x = self.pos.x + self.vel.x
        if self.world.test_robot_colision_with_terrain(self.raio, self.pos_alvo)[0]: # primeiro o x
            self.pos_alvo.x = self.pos.x
        else:
            self.pos.x = self.pos_alvo.x
        self.pos_alvo.y = self.pos.y + self.vel.y
        if self.world.test_robot_colision_with_terrain(self.raio, self.pos_alvo)[0]: # depois o y, assim o robô pode deslizar nas paredes
            self.pos_alvo.y = self.pos.y
        else:
            self.pos.y = self.pos_alvo.y

        self.last_pos.pop(0)
        self.last_pos.append(rl.Vector2(self.pos.x, self.pos.y))

        # Reforcos
        for type_rew in self.recompensas_terreno.keys():
            # print("Hoi")
            if self.world.test_robot_colision_with_terrain(self.raio, self.pos, type_rew)[0]:
                self.brain.add_reward(self.recompensas_terreno[type_rew])
                # print(f"Recompensa: {self.recompensas_terreno[type_rew]} | Total: {self.brain.get_total_reward():.2f}")

        # Aplica atrito para reduzir a velocidade ao longo do tempo
        self.vel.x *= (1 - self.atrito)
        self.vel.y *= (1 - self.atrito)
        if self.vel.x < self.limiar_atrito and self.vel.x > -self.limiar_atrito:
            self.vel.x = 0
        if self.vel.y < self.limiar_atrito and self.vel.y > -self.limiar_atrito:
            self.vel.y = 0

        # Atualiza a velocidade angular e o ângulo do robô com base na aceleração angular
        self.vel_angular += self.acc_angular
        self.angulo += self.vel_angular

        self.last_angulo.pop(0)
        self.last_angulo.append(self.angulo)

        # Aplica atrito angular para reduzir a velocidade angular ao longo do tempo
        self.vel_angular *= (1 - self.atrito_angular)
        if self.vel_angular < self.limiar_atrito_angular and self.vel_angular > -self.limiar_atrito_angular:
            self.vel_angular = 0        

    def get_reconstructed_state(self):
        # Retorna um estado reconstruído a partir do espaço latente, para monitorar o desempenho do modelo de mundo
        if self.brain.latent_reconstructed is None:
            return None
        
        
        # Extrai os sensores originais do estado reconstruído, assumindo a ordem definida por self.tipos_sensores
        reconstructed_sensors = self.brain.separate_state_by_sensor(self.brain.latent_reconstructed.cpu().numpy().squeeze())
        

        if self.sensores_ativos[self.tipos_sensores.index("visao")]:
            self.imagem_reconstruida = self.encoders["visao"].decode(torch.tensor(reconstructed_sensors["visao"])).cpu().detach().numpy() * 255.0
            # Salva a imagem reconstruída como PNG para monitoramento
            caminho_png = f"capturas_visao/reconstruido.png"
            #Usando pil
            from PIL import Image
            print(self.imagem_reconstruida.astype('uint8').shape)
            self.imagem_reconstruida = self.imagem_reconstruida.squeeze()  # Remove dimensões desnecessárias, se houver
            # Remove o canal alpha antes de salvar, se existir
            if self.imagem_reconstruida.shape[0] == 4:
                self.imagem_reconstruida = self.imagem_reconstruida[:3, :, :]
            img = Image.fromarray(self.imagem_reconstruida.astype('uint8').transpose(1, 2, 0)) # Transpõe para [Altura, Largura, Canais] antes de salvar
            img.save(caminho_png)
            



    def update(self, dt:float) -> None:
        self.delay_visao_atual += dt
        if self.sensores_ativos[self.tipos_sensores.index("tato")]:
            self.sensor_tato()

        if self.sensores_ativos[self.tipos_sensores.index("visao")]:
            if self.delay_visao_atual >= self.delay_visao:  # Ta na hora de atualizar o sensor de visão, se tiver passado o tempo de delay
                self.delay_visao_atual = 0
                self.sensor_visao()

        if self.sensores_ativos[self.tipos_sensores.index("acc")]:
            self.sensor_accelerometer()
        
        if self.sensores_ativos[self.tipos_sensores.index("gyro")]:
            self.sensor_gyroscope()

        if self.sensores_ativos[self.tipos_sensores.index("gps")]:
            self.sensor_gps()

        self.controles()

        self.get_reconstructed_state()
        
        self.phisics(dt)

    def sensor_gps(self):
        self.brain.set_info_sensor(self.pos, "gps", encoder=self.encoders["gps"])
        return self.pos

    def sensor_accelerometer(self):
        # Simula o sensor de aceleração
        v1 = rl.Vector2(self.last_pos[-2].x - self.last_pos[-3].x, self.last_pos[-2].y - self.last_pos[-3].y)
        v2 = rl.Vector2(self.last_pos[-1].x - self.last_pos[-2].x, self.last_pos[-1].y - self.last_pos[-2].y)

        self.acelerometer = rl.Vector2(v2.x - v1.x, v2.y - v1.y)
        self.brain.set_info_sensor(self.acelerometer, "acc", encoder=self.encoders["acc"])
        return self.acelerometer
    
    def sensor_gyroscope(self):
        # Simula o sensor de giroscópio
        self.gyroscope = self.last_angulo[-1] - self.last_angulo[-2]
        self.brain.set_info_sensor(self.gyroscope, "gyro", encoder=self.encoders["gyro"]) 
        return self.gyroscope

    def sensor_tato(self):
        # Detecta para cada 
        self.tato = self.world.test_robot_colision_with_terrain(self.raio_tato, self.pos_alvo)[1]
        self.brain.set_info_sensor(self.tato, "tato", encoder=self.encoders["tato"])
        if max(self.tato.values()) > self.limiar_dor:
            self.brain.add_reward(- (max(self.tato.values()) - self.limiar_dor)) # Recompensa negativa proporcional a quanto o tato ultrapassa o limiar de dor
        return self.tato

    def sensor_visao(self):
        """
        Recorta na GPU e converte para Tensor de forma nativa.
        """
        textura_mundo = self.world.handler.motor.render_tex if self.world and self.world.handler and self.world.handler.motor else None
        if not textura_mundo:
            return

        # ==========================================
        # ETAPA 1: RECORTE E INVERSÃO DIRETO NA GPU
        # ==========================================
        rl.begin_texture_mode(self.vision_tex)
        rl.clear_background(rl.BLANK)

        # Região de origem (Mundo)
        X_min = self.pos.x - self.raio_visao
        Y_min = (self.world.handler.motor.WINDOW_HEIGHT - self.pos.y) - self.raio_visao
        src_rect = rl.Rectangle(X_min, Y_min, self.tamanho_quadrado_visao, self.tamanho_quadrado_visao)

        # Região de destino (Nossa mini textura)
        # Passamos a altura negativa (-self.tamanho_quadrado_visao) para a GPU inverter verticalmente de graça!
        dest_rect = rl.Rectangle(0, 0, self.tamanho_quadrado_visao, -self.tamanho_quadrado_visao)

        # Copia os pixels de GPU para GPU instantaneamente
        rl.draw_texture_pro(textura_mundo.texture, src_rect, dest_rect, rl.Vector2(0, 0), 0.0, rl.WHITE)
        rl.end_texture_mode()

        # ==========================================
        # ETAPA 2: BAIXAR APENAS A REGIÃO MINI PARA A CPU
        # ==========================================
        imagem_visao = rl.load_image_from_texture(self.vision_tex.texture)

        # Exportação para validação (mantenha desativado durante execução normal)
        if self.export_vision:
            caminho_png = f"capturas_visao/visao_frame_{self.frame_counter}.png"
            rl.export_image(imagem_visao, caminho_png.encode('utf-8'))

        # ==========================================
        # ETAPA 3: CONVERSÃO BRUTA PARA NUMPY (CORRIGIDO PARA CFFI)
        # ==========================================
        # Pegamos o tamanho total em bytes (Largura * Altura * 4 canais RGBA)
        tamanho_bytes = self.tamanho_quadrado_visao * self.tamanho_quadrado_visao * 4
        
        # CORREÇÃO AQUI: Usamos rl.ffi.buffer para ler o ponteiro CFFI diretamente sem ctypes
        buffer_memoria = rl.ffi.buffer(imagem_visao.data, tamanho_bytes)
        
        # Criamos o array Numpy mapeando esse bloco de memória e tirando uma cópia rápida
        dados_np = np.frombuffer(buffer_memoria, dtype=np.uint8).reshape((self.tamanho_quadrado_visao, self.tamanho_quadrado_visao, 4)).copy()

        # Liberamos a memória da imagem da CPU imediatamente para evitar Memory Leak
        rl.unload_image(imagem_visao)

        # ==========================================
        # ETAPA 4: GERAR TENSOR PYTORCH
        # ==========================================
        tensor_torch = torch.from_numpy(dados_np).permute(2, 0, 1).float() / 255.0

        self.frame_counter += 1

        self.brain.set_info_sensor(tensor_torch, "visao", encoder=self.encoders["visao"])
        return tensor_torch
        

    def render(self):
        if self.sensores_ativos[self.tipos_sensores.index("visao")]: # Se o sensor de visão estiver ativo, desenha um círculo ao redor do robô representando seu campo de visão
            rl.draw_rectangle_lines_ex(rl.Rectangle(int(self.pos.x - self.raio_visao), int(self.pos.y - self.raio_visao), self.raio_visao * 2, self.raio_visao * 2), 1, rl.RED)

        if self.sensores_ativos[self.tipos_sensores.index("tato")]: # Se o sensor de tato estiver ativo, desenha um círculo ao redor do robô representando seu campo de tato
            rl.draw_circle(int(self.pos.x), int(self.pos.y), self.raio_tato, rl.Color(0, 255, 0, 50))

        match self.tipos_controle[self.controle_atual]:
            case "absoluto":
                # Desenha um círculo representando o robô na posição atual
                rl.draw_circle(int(self.pos.x), int(self.pos.y), self.raio, rl.BLUE)
            case "rotacional":
                # Colisao
                rl.draw_circle(int(self.pos.x), int(self.pos.y), self.raio, rl.Color(20,100,100,255))
                # Desenha um triangulo isoceles representando o robô na posição atual
                rl.draw_poly(self.pos, 3, self.raio, (self.angulo+math.pi) * 180 / math.pi, rl.BLUE)
                pos_ponta = rl.Vector2(self.pos.x + self.raio//2 * math.cos(self.angulo), self.pos.y + self.raio//2 * math.sin(self.angulo))
                rl.draw_poly(pos_ponta, 3, self.raio//2, (self.angulo+math.pi) * 180 / math.pi, rl.RED)

            case "rodas":
                # Colisao
                rl.draw_circle(int(self.pos.x), int(self.pos.y), self.raio, rl.Color(20,100,100,255))
                # Com duas rodas atras desenhadas como círculos, uma para cada roda
                pos_roda_esquerda = rl.Vector2(self.pos.x - self.raio//2 * math.cos(self.angulo + math.pi / 2), self.pos.y - self.raio//2 * math.sin(self.angulo + math.pi / 2))
                pos_roda_direita = rl.Vector2(self.pos.x - self.raio//2 * math.cos(self.angulo - math.pi / 2), self.pos.y - self.raio//2 * math.sin(self.angulo - math.pi / 2))
                rl.draw_circle(int(pos_roda_esquerda.x), int(pos_roda_esquerda.y), self.raio//4, rl.GREEN)
                rl.draw_circle(int(pos_roda_direita.x), int(pos_roda_direita.y), self.raio//4, rl.GREEN)
                # Desenha um triangulo isoceles representando o robô na posição atual
                rl.draw_poly(self.pos, 3, self.raio, (self.angulo+math.pi) * 180 / math.pi, rl.BLUE)
                pos_ponta = rl.Vector2(self.pos.x + self.raio//2 * math.cos(self.angulo), self.pos.y + self.raio//2 * math.sin(self.angulo))
                rl.draw_poly(pos_ponta, 3, self.raio//2, (self.angulo+math.pi) * 180 / math.pi, rl.RED)

