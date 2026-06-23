import pyray as rl
import time

from handler import Handler

class Motor():

    def __init__(self) -> None:
        # Criar a janela
        rl.set_config_flags(rl.FLAG_WINDOW_RESIZABLE) # Permite redimensionar a janela
        rl.set_trace_log_level(rl.LOG_NONE) # Desativa os logs do raylib
        self.WINDOW_WIDTH = 1400
        self.WINDOW_HEIGHT = 1000
        self.RENDER_ASPECT = self.WINDOW_WIDTH / self.WINDOW_HEIGHT
        rl.init_window(self.WINDOW_WIDTH, self.WINDOW_HEIGHT, b"World Model")
        rl.set_target_fps(60)
        rl.enable_cursor()

        # =========================================================================
        # CORREÇÃO: O tamanho virtual agora é FIXO em 1400x1000 (proporção do jogo)
        # =========================================================================
        self.VIRTUAL_W, self.VIRTUAL_H = 1400, 1000 
        
        # A textura é criada UMA VEZ com o tamanho fixo e NUNCA mais é recriada
        self.render_tex = rl.load_render_texture(self.VIRTUAL_W, self.VIRTUAL_H) 
        rl.set_texture_filter(self.render_tex.texture, rl.TEXTURE_FILTER_BILINEAR)   
        
        # O retângulo de origem também fica fixo no tamanho virtual
        self.src_rect = rl.Rectangle(0, 0, self.VIRTUAL_W, -self.VIRTUAL_H)  
        
        self.handler = Handler(self)
        self.prev_time = time.time()

    def get_scaled_rect(self) -> rl.Rectangle: 
        # Retorna o retângulo perfeito para onde a textura deve ser esticada na janela atual,
        # mantendo a proporção (aspect ratio) e centralizando-a (com barras pretas se necessário).
        sw, sh = rl.get_screen_width(), rl.get_screen_height()
        scale  = min(sw / self.VIRTUAL_W, sh / self.VIRTUAL_H)
        dw, dh = self.VIRTUAL_W * scale, self.VIRTUAL_H * scale
        return rl.Rectangle((sw - dw) / 2, (sh - dh) / 2, dw, dh)

    def render(self): # Método chamado a cada frame
        rl.begin_drawing()
        
        # Limpa a janela real (caso haja barras pretas nas laterais/cima/baixo)
        rl.clear_background(rl.BLACK) 

        # 1. Ativa a textura virtual de tamanho FIXO e desenha o jogo nela
        rl.begin_texture_mode(self.render_tex)  
        rl.clear_background(rl.BLACK) 
        
        self.handler.render() # Renderiza mapa, robô e HUD no espaço de 1400x1000
        
        rl.end_texture_mode()

        # 2. Desenha a textura virtual esticando-a para o tamanho correto da janela
        rl.draw_texture_pro(
            self.render_tex.texture,
            self.src_rect,
            self.get_scaled_rect(), # Centraliza e escala mantendo o aspect ratio
            rl.Vector2(0, 0),
            0.0,
            rl.WHITE,
        )
        rl.end_drawing()

    def update(self): # Método chamado a cada frame
        self.handler.update(self.dt)

    def run(self):
        while not rl.window_should_close():
            self.now = time.time()
            self.dt = self.now - self.prev_time
            self.prev_time = self.now

            self.render()
            self.update()