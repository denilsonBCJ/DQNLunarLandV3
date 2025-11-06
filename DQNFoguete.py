import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
import pygame
import numpy as np

# Função para verificar falhas de pouso
def is_landing_failure(dones, truncated, reward, obs, velocity_threshold=2.0, height_threshold=0.1, x_threshold_min=-0.2, x_threshold_max=0.2, angle_threshold=0.2):
    if dones or truncated:
        # Posição Y do foguete (quanto mais perto de 0, mais perto do solo)
        pos_y = obs[1]  # O valor da posição Y está na segunda posição da observação (índice 1)
        velocity_y = obs[3]  # Velocidade no eixo Y está na posição 3 da observação
        pos_x = obs[0]  # Posição X
        angle = obs[4]  # Ângulo do foguete (em radianos)

        # Se o foguete atinge o solo com alta velocidade (indicando um pouso falho)
        if pos_y <= height_threshold and abs(velocity_y) > velocity_threshold:
            return True
        # Se o foguete cai fora da área segura no eixo X
        if pos_x < x_threshold_min or pos_x > x_threshold_max:
            return True
        # Se o foguete pousa com inclinação excessiva
        if abs(angle) > angle_threshold:
            return True
    return False

# Função para verificar se o pouso foi bem-sucedido
def is_landing_success(dones, truncated, reward, obs, velocity_threshold=0.5, height_threshold=0.1, x_threshold_min=-0.2, x_threshold_max=0.2, angle_threshold=0.2):
    if dones or truncated:
        pos_y = obs[1]  # Posição Y
        velocity_y = obs[3]  # Velocidade no eixo Y
        pos_x = obs[0]  # Posição X
        velocity_x = obs[2]  # Velocidade no eixo X
        angle = obs[4]  # Ângulo do foguete (em radianos)
        left_leg_contact = obs[6]  # Contato da perna esquerda (índice 6)
        right_leg_contact = obs[7]  # Contato da perna direita (índice 7)

        # Pouso bem-sucedido se:
        # 1. O foguete está no solo (pos_y <= 0)
        # 2. A velocidade é baixa em ambos os eixos
        # 3. Está dentro da área de pouso
        # 4. A inclinação é mínima
        # 5. As pernas estão tocando o solo
        if (pos_y <= 0 and
            abs(velocity_y) < velocity_threshold and
            abs(velocity_x) < velocity_threshold and
            x_threshold_min <= pos_x <= x_threshold_max and
            abs(angle) < angle_threshold and
            (left_leg_contact or right_leg_contact)):  # Pelo menos uma perna está tocando o solo
            return True
    return False

# Configurações do ambiente
env_id = "LunarLander-v3"  # Atualizado para a versão v3
n_envs = 1  # Número de ambientes paralelos
max_episodes = 1000  # Número máximo de episódios de treinamento

# Criar o ambiente vetorizado
vec_env = make_vec_env(env_id, n_envs=n_envs)

# Definir o modelo DQN
model = DQN(
    "MlpPolicy",
    vec_env,
    verbose=0,  # Reduz os logs do treinamento
    learning_rate=1e-5,
    buffer_size=50000,
    batch_size=128,
    gamma=0.99,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
)

# Inicializa o Pygame
pygame.init()

# Criar uma tela com a mesma resolução do ambiente LunarLander
screen = pygame.display.set_mode((600, 400))
pygame.display.set_caption("999")

# Fonte de texto para desenhar na tela
font = pygame.font.SysFont("Arial", 24)

# Criar o ambiente com render_mode="rgb_array" para capturar a imagem
env = gym.make(env_id, render_mode="rgb_array")  # Usando 'rgb_array' para renderizar como imagem
obs, _ = env.reset()

# Variáveis para controle de falhas de pouso e pousos bem-sucedidos
landing_failures = 0
successful_landings = 0
landing_in_progress = False  # Flag para verificar se o pouso está em andamento
landing_delay_counter = 0  # Contador de delay para esperar o pouso

# Variáveis para o controle de combustível
max_fuel = 10  # Segundos de combustível
fuel = max_fuel
fuel_consumption_rate = 0.01  # Taxa de consumo de combustível (ex: a cada passo, consome 0.1 segundos)

# Número de episódios de treinamento
total_timesteps = int(1e6)
timesteps_per_update = 1000
current_timesteps = 0

# Variáveis para controlar o ritmo da atualização da tela
clock = pygame.time.Clock()
fps = 30  # Controlar o número de quadros por segundo

# Limite de passos após o pouso para garantir que a nave está realmente parada
landing_delay_limit = 100  # Aumentamos o delay para garantir que a nave está estabilizada

# Função para verificar se o combustível acabou
def check_fuel():
    return fuel <= 0

# Começar o treinamento
while current_timesteps < total_timesteps:
    # Treinar o modelo por 'timesteps_per_update'
    model.learn(total_timesteps=timesteps_per_update, reset_num_timesteps=False)

    # Renderizar o ambiente durante o treinamento
    for _ in range(timesteps_per_update):
        # Verifica se o combustível acabou
        if check_fuel():
            # Penaliza se o combustível acabou
            action = 0  # Ação 0: "não fazer nada" (válido para o LunarLander)
            fuel = 0  # Força o combustível a ser zero para simular que não há mais combustível
        else:
            # Caso contrário, o agente decide a ação normalmente
            action, _states = model.predict(obs, deterministic=True)

        obs, rewards, dones, truncated, info = env.step(action)

        # Verifica se houve falha de pouso
        if is_landing_failure(dones, truncated, rewards, obs):
            landing_failures += 1
            landing_in_progress = False  # Se houver falha, reseta o status do pouso

        # Verifica se o pouso foi bem-sucedido
        elif is_landing_success(dones, truncated, rewards, obs):
            if not landing_in_progress:
                landing_in_progress = True
                landing_delay_counter = landing_delay_limit  # Inicia o contador de espera para o pouso

        # Se o pouso foi detectado, aguarda um delay antes de contar como bem-sucedido
        if landing_in_progress:
            landing_delay_counter -= 1  # Decrementa o contador de delay
            if landing_delay_counter <= 0:
                # Verifica novamente se o pouso foi bem-sucedido após o delay
                if is_landing_success(dones, truncated, rewards, obs):
                    successful_landings += 1  # Conta o pouso após o delay
                landing_in_progress = False  # Reseta o status do pouso para evitar múltiplas contagens

        # Atualizar combustível
        if not check_fuel():
            fuel -= fuel_consumption_rate
            if fuel < 0:
                fuel = 0  # O combustível não pode ser negativo

        # Verifica se o episódio terminou
        if dones or truncated:
            obs, _ = env.reset()
            fuel = max_fuel  # Resetando o combustível a cada novo episódio

        # Renderiza o ambiente para capturar a imagem
        frame = env.render()  # Agora renderiza em "rgb_array", que retorna uma imagem

        if frame is not None:  # Verifica se a renderização foi bem-sucedida
            # Corrigir a orientação da imagem
            frame = np.transpose(frame, (1, 0, 2))  # Troca largura e altura

            # Atualiza a tela no ritmo de FPS controlado
            screen.fill((0, 0, 0))  # Limpa a tela
            success_text = font.render(f"Pousos Bem-sucedidos: {successful_landings}", True, (255, 255, 255))
            failure_text = font.render(f"Falhas de Pouso: {landing_failures}", True, (255, 0, 0))
            fuel_text = font.render(f"Combustível: {fuel:.2f} seg", True, (0, 255, 0))

            # Exibe as informações na tela
            screen.blit(success_text, (10, 10))
            screen.blit(failure_text, (10, 40))
            screen.blit(fuel_text, (10, 70))

            # Exibe a renderização do ambiente (imagem do foguete)
            frame = pygame.surfarray.make_surface(frame)  # Converte o array RGB em uma imagem que o Pygame pode exibir
            screen.blit(frame, (0, 100))  # Posiciona a imagem na tela

            pygame.display.flip()  # Atualiza a tela

        # Controla o FPS
        clock.tick(fps)

# Salvar o modelo treinado
model.save("lunar_lander_dqn")

# Fechar o ambiente
env.close()

# Fechar o Pygame
pygame.quit()
