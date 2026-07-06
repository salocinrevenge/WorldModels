import json
import os
import pyray as rl

class Controlador:
    def __init__(self, mundo, registrar, path_of_save="dataset/", to_save_imgs = False, frame_save_interval = 10):
        self.mundo = mundo
        self.registrar = registrar
        self.sample_index = 0
        self.to_save_imgs = to_save_imgs
        self.frame_save_interval = frame_save_interval
        
        self.time_to_save = 1000 
        self.ticks = 0
        
        base_dir = path_of_save if not path_of_save.endswith(('.json', '/')) else os.path.dirname(path_of_save.rstrip('/'))
        if not base_dir: 
            base_dir = "dataset"

        if self.registrar:
            os.makedirs(base_dir, exist_ok=True)
            existing_files = [f for f in os.listdir(base_dir) if f.endswith('.json')]
            next_number = 1
            if existing_files:
                existing_numbers = [int(f.split('.')[0]) for f in existing_files if f.split('.')[0].isdigit()]
                if existing_numbers:
                    next_number = max(existing_numbers) + 1
            
            self.sample_index = next_number
            self.path_of_save = os.path.join(base_dir, f"{next_number}.json")
            self.imgs_path = os.path.join(base_dir, f"{next_number}_images")
            os.makedirs(self.imgs_path, exist_ok=True)
            self.imgs_counter = 0
        else:
            if path_of_save.endswith('.json'):
                self.path_of_save = path_of_save
            else:
                existing_files = [f for f in os.listdir(base_dir) if f.endswith('.json')]
                if not existing_files:
                    raise FileNotFoundError(f"Nenhum arquivo .json encontrado em {base_dir}")
                existing_numbers = [int(f.split('.')[0]) for f in existing_files if f.split('.')[0].isdigit()]
                if not existing_numbers:
                    raise FileNotFoundError(f"Nenhum arquivo .json válido encontrado em {base_dir}")
                self.sample_index = max(existing_numbers)
                self.path_of_save = os.path.join(base_dir, f"{max(existing_numbers)}.json")

        if self.registrar:
            self.buffer_agente = []
            self.buffer_imagens = []
            with open(self.path_of_save, "w") as f:
                linha_terreno = {"tipo": "terreno", "dados": self.mundo.terreno}
                f.write(json.dumps(linha_terreno) + "\n")
        else:
            self.historico_agente = []
            with open(self.path_of_save, "r") as f:
                for linha in f:
                    if not linha.strip(): continue
                    dados = json.loads(linha)
                    if dados.get("tipo") == "terreno":
                        self.mundo.terreno = dados["dados"]
                    else:
                        self.historico_agente.append(dados)

            self.playback_fps = 60
            self.playback_speed_options = [0.25, 0.5, 1.0, 2.0, 4.0, 16.0, 128.0]
            self.playback_speed_index = 2
            self.playback_speed = self.playback_speed_options[self.playback_speed_index]
            self.playing = True
            self.playback_frame = 0.0
            self.total_frames = max(len(self.historico_agente), 1)
            self.timeline_rect = rl.Rectangle(1010, 650, 360, 18)
            self.controls_y = 680
            self.speed_controls_y = 724
            self.buttons = {
                "pause": rl.Rectangle(1010, self.controls_y, 90, 34),
                "back_10s": rl.Rectangle(1110, self.controls_y, 90, 34),
                "forward_10s": rl.Rectangle(1210, self.controls_y, 90, 34),
                "speed_down": rl.Rectangle(1010, self.speed_controls_y, 90, 34),
                "speed_reset": rl.Rectangle(1110, self.speed_controls_y, 90, 34),
                "speed_up": rl.Rectangle(1210, self.speed_controls_y, 90, 34),
            }
            self.current_history_index = 0
            self.current_history_time = 0
            self._apply_history_index(0)

    def update(self, dt):
        if self.registrar:
            last_action = self.mundo.agente.last_action
            last_action_a = last_action[0].item() if last_action is not None else None
            last_action_b = last_action[1].item() if last_action is not None else None
            estado_atual = {
                "x": self.mundo.agente.pos.x,
                "y": self.mundo.agente.pos.y,
                "angle": self.mundo.agente.angulo,
                "vel_x": self.mundo.agente.vel.x,
                "vel_y": self.mundo.agente.vel.y,
                "acc_x": self.mundo.agente.acc.x,
                "acc_y": self.mundo.agente.acc.y,
                "vel_angular": self.mundo.agente.vel_angular,
                "acc_angular": self.mundo.agente.acc_angular,
                "target_x": self.mundo.agente.brain.target[0],
                "target_y": self.mundo.agente.brain.target[1],
                "ation_a": last_action_a,
                "ation_b": last_action_b,
                "time": self.mundo.agente.update_counter
            }
            self.buffer_agente.append(estado_atual)

            # Captura e processa a imagem do frame baseado no intervalo
            if self.to_save_imgs and (self.ticks % self.frame_save_interval == 0):
                render_tex_obj = self.mundo.handler.motor.render_tex
                
                imagem_visao = rl.load_image_from_texture(render_tex_obj.texture)
                
                rl.image_crop(imagem_visao, rl.Rectangle(0, 0, 1000, 1000))
                rl.image_flip_vertical(imagem_visao)
                
                # Adiciona a cópia tratada ao buffer
                self.buffer_imagens.append(imagem_visao)
            
            # Bloco de salvamento unificado (executado a cada X passos)
            if (self.ticks + 1) % self.time_to_save == 0:
                self.flush_log()
                if self.to_save_imgs:
                    self.flush_images()
        else:
            self._handle_replay_input()

            if self.playing and self.total_frames > 0:
                self.playback_frame += self.playback_speed * self.playback_fps * dt
                if self.playback_frame >= self.total_frames - 1:
                    self.playback_frame = float(self.total_frames - 1)
                    self.playing = False

            self._apply_history_index(int(self.playback_frame))

        self.ticks += 1

    def _point_in_rect(self, point, rect):
        return rect.x <= point.x <= rect.x + rect.width and rect.y <= point.y <= rect.y + rect.height

    def _clamp_history_index(self, index):
        if not self.historico_agente:
            return 0
        return max(0, min(index, len(self.historico_agente) - 1))

    def _seek_seconds(self, seconds):
        if not self.historico_agente:
            return
        self.playback_frame = float(self._clamp_history_index(int(round(self.playback_frame + seconds * self.playback_fps))))
        # self.playing = False
        self._apply_history_index(int(self.playback_frame))

    def _set_playback_speed(self, speed):
        self.playback_speed = max(0.25, min(speed, 128.0))
        if self.playback_speed in self.playback_speed_options:
            self.playback_speed_index = self.playback_speed_options.index(self.playback_speed)
        else:
            self.playback_speed_index = min(range(len(self.playback_speed_options)), key=lambda idx: abs(self.playback_speed_options[idx] - self.playback_speed))

    def _apply_history_index(self, index):
        if not self.historico_agente:
            return

        index = self._clamp_history_index(index)
        estado = self.historico_agente[index]
        self.current_history_index = index
        self.current_history_time = estado["time"]
        self.mundo.ticks = estado["time"]
        self.mundo.agente.pos.x = estado["x"]
        self.mundo.agente.pos.y = estado["y"]
        self.mundo.agente.angulo = estado["angle"]
        self.mundo.agente.vel.x = estado["vel_x"]
        self.mundo.agente.vel.y = estado["vel_y"]
        self.mundo.agente.acc.x = estado["acc_x"]
        self.mundo.agente.acc.y = estado["acc_y"]
        self.mundo.agente.vel_angular = estado["vel_angular"]
        self.mundo.agente.acc_angular = estado["acc_angular"]
        self.mundo.agente.brain.target = (estado["target_x"], estado["target_y"])
        self.mundo.agente.last_action = (estado["ation_a"], estado["ation_b"])
        self.mundo.agente.update_counter = estado["time"]

    def _handle_replay_input(self):
        if not self.historico_agente:
            return

        mouse_pos = rl.get_mouse_position()
        if rl.is_mouse_button_pressed(rl.MOUSE_BUTTON_LEFT):
            if self._point_in_rect(mouse_pos, self.buttons["pause"]):
                self.playing = not self.playing
            elif self._point_in_rect(mouse_pos, self.buttons["back_10s"]):
                self._seek_seconds(-10)
            elif self._point_in_rect(mouse_pos, self.buttons["forward_10s"]):
                self._seek_seconds(10)
            elif self._point_in_rect(mouse_pos, self.buttons["speed_down"]):
                self._set_playback_speed(self.playback_speed / 2)
            elif self._point_in_rect(mouse_pos, self.buttons["speed_reset"]):
                self._set_playback_speed(1.0)
            elif self._point_in_rect(mouse_pos, self.buttons["speed_up"]):
                self._set_playback_speed(self.playback_speed * 2)

            if self._point_in_rect(mouse_pos, self.timeline_rect):
                relative = (mouse_pos.x - self.timeline_rect.x) / self.timeline_rect.width
                self.playback_frame = float(self._clamp_history_index(int(relative * (len(self.historico_agente) - 1))))
                self.playing = False
                self._apply_history_index(int(self.playback_frame))

    def render(self):
        if self.registrar:
            return

        if not self.historico_agente:
            rl.draw_text("Nenhum replay carregado.", 1010, 550, 20, rl.WHITE)
            return

        panel_x = 1000
        panel_w = 400
        rl.draw_rectangle(panel_x, 0, panel_w, 1000, rl.Color(20, 20, 24, 255))
        rl.draw_line(panel_x, 0, panel_x, 1000, rl.GRAY)

        rl.draw_text("Replay", 1010, 510, 24, rl.WHITE)
        rl.draw_text(f"Frame: {self.current_history_index + 1}/{len(self.historico_agente)}", 1010, 540, 18, rl.WHITE)
        rl.draw_text(f"Tempo: {self.current_history_time / self.playback_fps:.1f}s", 1010, 570, 18, rl.WHITE)
        rl.draw_text(f"Velocidade: {self.playback_speed:.2f}x", 1010, 600, 18, rl.WHITE)
        rl.draw_text("Clique na barra para navegar", 1010, 620, 16, rl.GRAY)

        pause_label = "Pausar" if self.playing else "Continuar"
        self._draw_button(self.buttons["pause"], pause_label)
        self._draw_button(self.buttons["back_10s"], "-10s")
        self._draw_button(self.buttons["forward_10s"], "+10s")
        self._draw_button(self.buttons["speed_down"], "-")
        self._draw_button(self.buttons["speed_reset"], "1x")
        self._draw_button(self.buttons["speed_up"], "+")

        rl.draw_rectangle_lines_ex(self.timeline_rect, 2, rl.WHITE)
        progress = 0.0 if len(self.historico_agente) <= 1 else self.current_history_index / (len(self.historico_agente) - 1)
        fill_width = int(self.timeline_rect.width * progress)
        rl.draw_rectangle(int(self.timeline_rect.x), int(self.timeline_rect.y), fill_width, int(self.timeline_rect.height), rl.BLUE)
        handle_x = self.timeline_rect.x + fill_width
        rl.draw_circle(int(handle_x), int(self.timeline_rect.y + self.timeline_rect.height // 2), 8, rl.YELLOW)

    def _draw_button(self, rect, label):
        rl.draw_rectangle_rec(rect, rl.Color(50, 50, 60, 255))
        rl.draw_rectangle_lines_ex(rect, 2, rl.WHITE)
        rl.draw_text(label, int(rect.x + rect.width / 2 - rl.measure_text(label, 16) / 2), int(rect.y + 8), 16, rl.WHITE)

    def flush_log(self):
        if not self.buffer_agente:
            return
        with open(self.path_of_save, "a") as f:
            for estado in self.buffer_agente:
                f.write(json.dumps(estado) + "\n")
        self.buffer_agente.clear()

    def flush_images(self):
        """Grava as imagens no disco e limpa os objetos da memória C/C++"""
        if not self.buffer_imagens:
            return
            
        for img in self.buffer_imagens:
            caminho_png = os.path.join(self.imgs_path, f"frame_{self.imgs_counter}.png")
            
            # Exporta a imagem salva na RAM para o HD
            rl.export_image(img, caminho_png.encode('utf-8'))
            rl.unload_image(img)  # Libera a memória alocada pelo Raylib em C
            self.imgs_counter += 1

        self.buffer_imagens.clear()