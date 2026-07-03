import json

class Controlador:
    def __init__(self, mundo, registrar, path_of_save="log.json"):
        self.mundo = mundo
        self.registrar = registrar
        self.path_of_save = path_of_save
        self.time_to_save = 1000
        self.ticks = 0
        
        if self.registrar:
            self.buffer_agente = []
            
            # Abre em "w" (write) na inicialização para criar/limpar o arquivo 
            # e salvar a informação estática do terreno na primeira linha.
            with open(self.path_of_save, "w") as f:
                linha_terreno = {"tipo": "terreno", "dados": self.mundo.terreno}
                f.write(json.dumps(linha_terreno) + "\n")
        else:
            self.historico_agente = []
            
            # Modo "r" (read) para ler o arquivo linha por linha
            with open(self.path_of_save, "r") as f:
                for linha in f:
                    if not linha.strip(): continue # Ignora linhas vazias
                    
                    dados = json.loads(linha)
                    if dados.get("tipo") == "terreno":
                        self.mundo.terreno = dados["dados"]
                    else:
                        self.historico_agente.append(dados)

    def update(self):
        # Movi o incremento para o final ou ajustei a lógica de índice para não pular 
        # o frame 0 (se tick começa em 0 e já incrementa, ele leria o índice 1 primeiro).
        
        if self.registrar:
            # Reúne todo o estado atual em um único dicionário (sem listas contínuas)
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
                "time": self.mundo.agente.update_counter
            }
            
            self.buffer_agente.append(estado_atual)

            # Salva no disco a cada time_to_save
            if (self.ticks + 1) % self.time_to_save == 0:
                self.flush_log()
        else:
            # Atualiza o estado do agente com base no log
            if self.ticks < len(self.historico_agente):
                estado = self.historico_agente[self.ticks]
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
                self.mundo.agente.update_counter = estado["time"]

        self.ticks += 1

    def flush_log(self):
        """Grava os dados acumulados no arquivo e esvazia o buffer."""
        if not self.buffer_agente:
            return
            
        # Abre em "a" (append) para adicionar as novas linhas ao final do arquivo
        with open(self.path_of_save, "a") as f:
            for estado in self.buffer_agente:
                # json.dumps() converte um dicionário para uma string JSON em linha única
                f.write(json.dumps(estado) + "\n")
                
        # Limpa o buffer para liberar memória RAM
        self.buffer_agente.clear()