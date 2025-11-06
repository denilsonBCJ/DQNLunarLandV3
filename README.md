# 🚀 DQN Lunar Lander V3

Este projeto é uma implementação do algoritmo **Deep Q-Network (DQN)** para treinar um agente a pousar uma nave no ambiente **LunarLander-v3** do [Gymnasium](https://gymnasium.farama.org/).

O agente aprende a controlar o foguete utilizando redes neurais e reforço positivo, ajustando seus parâmetros para realizar pousos suaves e eficientes.



## 🎮 Demonstração
O objetivo do agente é pousar o foguete de forma estável na área designada sem tombar ou sair da tela.

O ambiente é totalmente baseado em **Box2D**, simulando a física realista da gravidade, impulso e colisões.



## 🧠 Tecnologias Utilizadas
- 🧩 **Python 3.13**
- 🤖 **Stable Baselines 3** — biblioteca para aprendizado por reforço
- 🌕 **Gymnasium (Box2D)** — ambiente do Lunar Lander
- 🧮 **NumPy** — operações matemáticas e de rede neural
- 🎨 **Pygame** — visualização e renderização da simulação



## ⚙️ Instalação e Execução

### 1️⃣ Clone o repositório
```bash
git clone https://github.com/SEU_USUARIO/DQNLunarLandV3.git
cd DQNLunarLandV3
```

### 2️⃣ Instale as dependências

```bash
pip install -r requirements.txt
```

### 3️⃣ Execute o treinamento ou a simulação

```bash
python DQNFoguete.py
```


## 📊 Resultados Esperados

Durante o treinamento, o agente deve:

* Aprender a equilibrar o foguete em diferentes fases da descida;
* Minimizar danos durante o pouso;
* Atingir pontuações acima de **200 pontos**, indicando um pouso quase perfeito.



## 🧩 Melhorias Futuras

* Adicionar visualização gráfica do treinamento em tempo real;
* Implementar **replay buffer personalizado**;
* Comparar o desempenho com outros algoritmos (PPO, A2C, SAC);
* Criar interface com **Kivy** ou **Pygame** para controle manual do foguete.



## 👨‍💻 Autor

**Denilson Borges**
💡 Dev Python focado em automação e bots.
🚀 Apaixonado por tech e código limpo.
🧠 Explorando APIs e IA.


## 📜 Licença

Este projeto é de uso livre para fins educacionais e de pesquisa.
Sinta-se à vontade para clonar, estudar e melhorar!

