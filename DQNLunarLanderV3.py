import os
# 🔧 Correções para a janela do Pygame/Gym abrir corretamente no Windows
os.environ["SDL_VIDEO_CENTERED"] = "1"
os.environ["SDL_VIDEODRIVER"] = "windows"
os.environ["SDL_VIDEO_WINDOW_POS"] = "100,100"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
import pygame
import numpy as np

# Função para verificar falhas de pouso
def is_landing_failure(dones, truncated, reward, obs, velocity_threshold=2.0, height_threshold=0.1, x_threshold_min=-0.2, x_threshold_max=0.2, angle_threshold=0.2):
    if dones or truncated:
        pos_y = obs[1]
        velocity_y = obs[3]
        pos_x = obs[0]
        angle = obs[4]

        if pos_y <= height_threshold and abs(velocity_y) > velocity_threshold:
            return True
        if pos_x < x_threshold_min or pos_x > x_threshold_max:
            return True
        if abs(angle) > angle_threshold:
            return True
    return False

# Função para verificar se o pouso foi bem-sucedido
def is_landing_success(dones, truncated, reward, obs, velocity_threshold=0.5, height_threshold=0.1, x_threshold_min=-0.2, x_threshold_max=0.2, angle_threshold=0.2):
    if dones or truncated:
        pos_y = obs[1]
        velocity_y = obs[3]
        pos_x = obs[0]
        velocity_x = obs[2]
        angle = obs[4]
        left_leg_contact = obs[6]
        right_leg_contact = obs[7]

        if (pos_y <= 0 and
            abs(velocity_y) < velocity_threshold and
            abs(velocity_x) < velocity_threshold and
            x_threshold_min <= pos_x <= x_threshold_max and
            abs(angle) < angle_threshold and
            (left_leg_contact or right_leg_contact)):
            return True
    return False

# Configurações do ambiente
env_id = "LunarLander-v3"
n_envs = 1
max_episodes = 1000

# Criar o ambiente vetorizado
vec_env = make_vec_env(env_id, n_envs=n_envs)

# Definir o modelo DQN
model = DQN(
    "MlpPolicy",
    vec_env,
    verbose=0,
    learning_rate=1e-5,
    buffer_size=50000,
    batch_size=128,
    gamma=0.99,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
)

# Inicializa o Pygame
pygame.display.init()
pygame.init()

# Criar uma tela com a mesma resolução do ambiente LunarLander
screen = pygame.display.set_mode((600, 400))
pygame.display.set_caption("DQNLunarLandV3")

# Fonte de texto para desenhar na tela
font = pygame.font.SysFont("Arial", 24)

# Criar o ambiente com render_mode="rgb_array" para capturar a imagem
env = gym.make(env_id, render_mode="rgb_array")
obs, _ = env.reset()

# Variáveis de controle
landing_failures = 0
successful_landings = 0
landing_in_progress = False
landing_delay_counter = 0
max_fuel = 10
fuel = max_fuel
fuel_consumption_rate = 0.01
total_timesteps = int(1e6)
timesteps_per_update = 1000
current_timesteps = 0
clock = pygame.time.Clock()
fps = 30
landing_delay_limit = 100

def check_fuel():
    return fuel <= 0

# Loop principal
while current_timesteps < total_timesteps:
    model.learn(total_timesteps=timesteps_per_update, reset_num_timesteps=False)

    for _ in range(timesteps_per_update):
        if check_fuel():
            action = 0
            fuel = 0
        else:
            action, _states = model.predict(obs, deterministic=True)

        obs, rewards, dones, truncated, info = env.step(action)

        if is_landing_failure(dones, truncated, rewards, obs):
            landing_failures += 1
            landing_in_progress = False
        elif is_landing_success(dones, truncated, rewards, obs):
            if not landing_in_progress:
                landing_in_progress = True
                landing_delay_counter = landing_delay_limit

        if landing_in_progress:
            landing_delay_counter -= 1
            if landing_delay_counter <= 0:
                if is_landing_success(dones, truncated, rewards, obs):
                    successful_landings += 1
                landing_in_progress = False

        if not check_fuel():
            fuel -= fuel_consumption_rate
            if fuel < 0:
                fuel = 0

        if dones or truncated:
            obs, _ = env.reset()
            fuel = max_fuel

        frame = env.render()

        if frame is not None:
            frame = np.transpose(frame, (1, 0, 2))
            screen.fill((0, 0, 0))
            success_text = font.render(f"Pousos Bem-sucedidos: {successful_landings}", True, (255, 255, 255))
            failure_text = font.render(f"Falhas de Pouso: {landing_failures}", True, (255, 0, 0))
            fuel_text = font.render(f"Combustível: {fuel:.2f} seg", True, (0, 255, 0))
            screen.blit(success_text, (10, 10))
            screen.blit(failure_text, (10, 40))
            screen.blit(fuel_text, (10, 70))
            frame = pygame.surfarray.make_surface(frame)
            screen.blit(frame, (0, 100))
            pygame.display.flip()

        clock.tick(fps)

# Salvar o modelo treinado
model.save("lunar_lander_dqn")

# Fechar o ambiente e o Pygame
env.close()
pygame.quit()
