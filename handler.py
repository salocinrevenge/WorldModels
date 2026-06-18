from curiosity_world import Curiosity_world

class Handler():
    def __init__(self, motor):
        self.motor = motor
        self.scenes = [Curiosity_world(self)] # Lista de cenas, para renderizar e atualizar
        self.current_scene = 0 # Cena atual, para renderizar e atualizar

    def update(self, dt):
        self.scenes[self.current_scene].update(dt)

    def render(self):
        self.scenes[self.current_scene].render()