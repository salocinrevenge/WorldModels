import pyray as rl
import time

from handler import Handler

class Motor():

    def __init__(self) -> None:

        # Criar a janela
        rl.set_config_flags(rl.FLAG_WINDOW_RESIZABLE) # Permite redimensionar a janela
        rl.set_trace_log_level(rl.LOG_NONE) # Desativa os logs do raylib, para evitar poluição no console
        self.WINDOW_WIDTH = 1400
        self.WINDOW_HEIGHT = 1000
        self.RENDER_ASPECT = self.WINDOW_WIDTH / self.WINDOW_HEIGHT
        rl.init_window(self.WINDOW_WIDTH, self.WINDOW_HEIGHT, b"World Model")
        rl.set_target_fps(60)
        rl.enable_cursor()

        # Desenhar sobre textura reescalada para manter a proporção

        self.VIRTUAL_W, self.VIRTUAL_H = 1000, 1000 # Tamanho virtual da tela, para manter a proporção
        self.render_tex = rl.load_render_texture(self.VIRTUAL_W, self.VIRTUAL_H) # Render texture para desenhar o jogo, depois reescalada para a tela
        rl.set_texture_filter(self.render_tex.texture, rl.TEXTURE_FILTER_BILINEAR)   # Filtro para suavizar a textura quando reescalada
        self.src_rect = rl.Rectangle(0, 0, self.VIRTUAL_W, -self.VIRTUAL_H)  # Retângulo de origem para desenhar a textura, com altura negativa para inverter a textura
        self.handler = Handler(self)
        self.prev_time = time.time()

    def fit_aspect(self, sw: int, sh: int, aspect: float) -> tuple[int, int]:
        """Largest (w, h) of the given aspect ratio that fits inside sw x sh, in
        physical pixels and rounded to even numbers (some GL drivers dislike odd
        render-texture dimensions)."""
        if sw / sh > aspect: # Se a janela é mais larga que a proporção desejada, limitar pela altura
            h = sh
            w = int(round(sh * aspect))
        else:   # Se a janela é mais alta que a proporção desejada, limitar pela largura
            w = sw
            h = int(round(sw / aspect))
        return max(2, w - (w & 1)), max(2, h - (h & 1))

    def update_render_target(self, render_tex, src_rect):
        """ Mantem o render texture redimensionado para a resolução correta, para evitar distorção da imagem.
        Recria apenas quando o tamanho realmente muda, para evitar perda de desempenho ao recriar a textura a cada frame. """
        sw, sh = rl.get_screen_width(), rl.get_screen_height() # Obtem o tamanho da janela, para ajustar o render texture e manter a proporção
        if sw <= 0 or sh <= 0:
            return render_tex, src_rect
        
        vw, vh = self.fit_aspect(sw, sh, self.RENDER_ASPECT) # Atualiza o tamanho virtual da tela para manter a proporção, baseado no tamanho da janela, externo fica preto
        if vw == self.VIRTUAL_W and vh == self.VIRTUAL_H:
            return render_tex, src_rect

        self.VIRTUAL_W, self.VIRTUAL_H = vw, vh
        rl.unload_render_texture(render_tex)
        render_tex = rl.load_render_texture(vw, vh)
        rl.set_texture_filter(render_tex.texture, rl.TEXTURE_FILTER_BILINEAR)
        src_rect = rl.Rectangle(0, 0, vw, -vh)
        return render_tex, src_rect

    def get_scaled_rect(self) -> rl.Rectangle: # Retorna um retângulo centralizado e redimensionado para manter a proporção da cena dentro da janela
        sw, sh = rl.get_screen_width(), rl.get_screen_height()
        scale  = min(sw / self.VIRTUAL_W, sh / self.VIRTUAL_H)
        dw, dh = self.VIRTUAL_W * scale, self.VIRTUAL_H * scale
        return rl.Rectangle((sw - dw) / 2, (sh - dh) / 2, dw, dh)

    def render(self): # metodo chamado a cada frame
        self.render_tex, self.src_rect = self.update_render_target(self.render_tex, self.src_rect)  # Atualiza o render texture para o tamanho correto, se necessário
        rl.begin_drawing()
        rl.begin_texture_mode(self.render_tex)  # Textura de renderização para desenhar o jogo, depois reescalada para a tela
        rl.clear_background(rl.BLACK) # Limpar a tela

        # Renderizar as cenas
        self.handler.render()

        rl.end_texture_mode()
        rl.draw_texture_pro(
            self.render_tex.texture,
            self.src_rect,
            self.get_scaled_rect(),
            rl.Vector2(0, 0),
            0.0,
            rl.WHITE,
        )
        rl.end_drawing()


    def update(self): # metodo chamado a cada frame
        self.handler.update(self.dt)

    def run(self):
        while not rl.window_should_close():
            self.now = time.time()
            self.dt = self.now - self.prev_time
            self.prev_time = self.now

            self.render()
            self.update()