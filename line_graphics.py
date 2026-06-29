from math_utils import Rect

import pyray as rl 

class LineGraphics:
    def __init__(self, format: Rect, name_x: str = None, name_y: str = None, title: str = None, size_x: int = None, size_y: int = None, colors=None):
        self.format = format
        self.name_x = name_x
        self.name_y = name_y
        self.title = title
        self.size_x = size_x # Numero de amostras no eixo x. Se None, mostra todas as amostras. Se definido, mostra apenas as últimas size_x amostras
        self.size_y = size_y # Numero de amostras no eixo y. Se None, mostra todas as amostras. Se definido, mostra apenas as últimas size_y amostras
        self.colors = colors if colors is not None else {1: rl.BLUE}

        self.data = {}

    def add_data(self, y, x= None, tag = 1):
        if x is None:
            x = len(self.data[tag]) if tag in self.data else 0
            if tag not in self.data:
                self.data[tag] = []
            self.data[tag].append((x, y))

    def render(self):


        # Calcula max_y e min_y para todas as tag
        max_y = max([y for tag in self.data for x, y in self.data[tag]]) if any(len(self.data[tag]) > 0 for tag in self.data) else 1
        min_y = min([y for tag in self.data for x, y in self.data[tag]]) if any(len(self.data[tag]) > 0 for tag in self.data) else 0

        # Desenha sobre a textura do raylib segundo o formato definido
        rl.draw_rectangle_lines(self.format.x, self.format.y, self.format.width, self.format.height, rl.WHITE)

        # Legenda do grafico
        if self.title is not None:
            rl.draw_text(self.title, self.format.x + self.format.width // 2 - rl.measure_text(self.title, 20) // 2, self.format.y - 30, 20, rl.WHITE)
        if self.name_x is not None:
            rl.draw_text(self.name_x, self.format.x + self.format.width // 2 - rl.measure_text(self.name_x, 20) // 2, self.format.y + self.format.height + 20, 20, rl.WHITE)
        if self.name_y is not None:
            rl.draw_text(self.name_y, self.format.x - rl.measure_text(self.name_y, 20) - 10, self.format.y + self.format.height + 20, 20, rl.WHITE)

        for tag, color in self.colors.items():

            self.color = color

            if len(self.data[tag]) == 1:
                x, y = self.data[tag][0]
                x = self.format.x + (x / self.size_x) * self.format.width if self.size_x is not None else self.format.x + (x / len(self.data[tag])) * self.format.width
                y = self.format.y + self.format.height - ((y - min_y) / (max_y - min_y)) * self.format.height if max_y != min_y else self.format.y + self.format.height // 2
                rl.draw_circle(int(x), int(y), 2, color)
            else:
                for i in range(1, len(self.data[tag])):
                    x1, y1 = self.data[tag][i - 1]
                    x2, y2 = self.data[tag][i]
                    x1 = self.format.x + (x1 / self.size_x) * self.format.width if self.size_x is not None else self.format.x + (x1 / len(self.data[tag])) * self.format.width
                    y1 = self.format.y + self.format.height - ((y1 - min_y) / (max_y - min_y)) * self.format.height if max_y != min_y else self.format.y + self.format.height // 2
                    x2 = self.format.x + (x2 / self.size_x) * self.format.width if self.size_x is not None else self.format.x + (x2 / len(self.data[tag])) * self.format.width
                    y2 = self.format.y + self.format.height - ((y2 - min_y) / (max_y - min_y)) * self.format.height if max_y != min_y else self.format.y + self.format.height // 2
                    rl.draw_line(int(x1), int(y1), int(x2), int(y2), color)


        # Para cada cor desenha dentro do retangulo definido no formato, a legenda da cor e o valor atual do dado, no canto superior direito do retangulo
        legend_y = self.format.y + 10
        for tag, color in self.colors.items():
            rl.draw_text(f"{tag}", self.format.x + self.format.width - 160, legend_y, 15, color)
            legend_y += 30

        # Desenha números para o eixo x e o eixo y de 20 em 20%
        for i in range(0, 101, 25):
            x = self.format.x + (i / 100) * self.format.width
            y = self.format.y + self.format.height - (i / 100) * self.format.height
            rl.draw_line(int(x), self.format.y + self.format.height, int(x), self.format.y + self.format.height + 5, rl.WHITE)
            rl.draw_text(f"{i/100 * len(self.data[tag]):.0f}", int(x) - 10, self.format.y + self.format.height + 10, 15, rl.WHITE)
            rl.draw_line(self.format.x - 5, int(y), self.format.x, int(y), rl.WHITE)
            # em notacao cientifica, com 2 casas decimais
            rl.draw_text(f"{i/100 * (max_y - min_y) + min_y:.0e}", self.format.x - 60, int(y) - 10, 15, rl.WHITE)

        