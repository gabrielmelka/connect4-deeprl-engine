# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 20:04:39 2026

@author: melka
"""

import numpy as np


class Connect4:
    """
2 canals : one for the yellow player, one for the red one
each grid has indexes [line : 0->5 ; columns 0->6] with [0,0] for the bottom left
    """

    def __init__(self):
        self.grid = np.zeros((6, 7), dtype=np.float32)
        self.player = 1  # 1 commence, puis -1, puis 1 ...

    "for a new game"
    def reset(self):
        self.grid = np.zeros((6, 7), dtype=np.float32)
        self.player = 1
        return self.get_state()

    "for a new move"
    def step(self, column):
        # we are looking for the index such that gravity plays a role 
        #when we are choosing a column
        # if the column is full, line =-1
        
        line = -1
        
        for l in range(6):
            if self.grid[l, column] == 0:
                line = l
                break
                
        if line == -1:
            raise ValueError("Column {colonne} full !")

        # put the chip 
        self.grid[line, column] = self.player

        # verify win
        if self._check_win(line, column):
            reward = 1.0
            done = True
            state = self.get_state()
            return state, reward, done

        # verify draw (if every culumn is full)
        if np.all(self.grid != 0):
            reward = 0.0
            done = True
            state = self.get_state()
            return state, reward, done

        # if there is not a win or a draw, the game resumes 
        self.player *= -1
        reward = 0.0
        done = False
        state = self.get_state()
        return state, reward, done

    # ------------------------------------------------------------------
    # win : we only check around the last chip placed 
    # ------------------------------------------------------------------
    def _check_win(self, line, column):
        player = self.grid[line, column]

        # 4 directions 
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dl, dc in directions:
            count = 1  # 1 because there is already a chip (the one we just posed)

            # counting in the positive side
            for k in range(1, 4):
                l = line + k * dl
                c = column + k * dc
                if 0 <= l < 6 and 0 <= c < 7 and self.grid[l, c] == player:
                    count += 1
                else:
                    break

            # counting in the negative side
            for k in range(1, 4):
                l = line - k * dl
                c = column - k * dc
                if 0 <= l < 6 and 0 <= c < 7 and self.grid[l, c] == player:
                    count += 1
                else:
                    break

            if count >= 4:
                return True

        return False

    # ------------------------------------------------------------------
    # legalactions = non full columns 
    # ------------------------------------------------------------------
    def get_legal_actions(self):
        return [c for c in range(7) if self.grid[5, c] == 0]

    # ------------------------------------------------------------------
    # State : conversion en 2 canaux pour le réseau
    #   - Canal 0 : pions du joueur courant  (1 là où il a un pion)
    #   - Canal 1 : pions de l'adversaire    (1 là où l'adversaire a un pion)
    #
    # L'inversion de perspective est automatique :
    #   si self.joueur == 1  → canal 0 = (grille == 1),  canal 1 = (grille == -1)
    #   si self.joueur == -1 → canal 0 = (grille == -1), canal 1 = (grille == 1)
    # ------------------------------------------------------------------
    def get_state(self):
        my_canal = (self.grid == self.player).astype(np.float32)
        opponent_canal = (self.grid == -self.player).astype(np.float32)
        # Shape : (2, 6, 7) for the CNN
        return np.stack([my_canal, opponent_canal], axis=0)

    # ------------------------------------------------------------------
    # Display (for débugger)
    # ------------------------------------------------------------------
    def afficher(self):
        symboles = {0: ".", 1: "X", -1: "O"}
        print()
        for line in range(5, -1, -1):
            row = ""
            for col in range(7):
                row += symboles[self.grid[line, col]] + " "
            print(row)
        print("0 1 2 3 4 5 6")
        print(f"Au tour de : {'X' if self.player == 1 else 'O'}")
        print()


# ======================================================================
# Test rapide
# ======================================================================
if __name__ == "__main__":
    env = Connect4()
    state = env.reset()

    print("=== Partie de test ===")
    env.afficher()

    # Simuler quelques coups
    coups = [3, 2, 3, 2, 3, 2, 3]  # Joueur X empile en colonne 3 → victoire verticale

    for coup in coups:
        print(f"Joueur {'X' if env.player == 1 else 'O'} joue colonne {coup}")
        state, reward, done = env.step(coup)
        env.afficher()

        if done:
            if reward == 1.0:
                # Le joueur qui vient de jouer a gagné
                # Attention : après step(), joueur a déjà changé si pas done
                # Mais ici done=True donc joueur n'a PAS changé
                print("Partie terminée ! Victoire !")
            else:
                print("Match nul !")
            break

    print(f"Shape du state : {state.shape}")
    print(f"Canal 'moi' :\n{state[0]}")
    print(f"Canal 'adversaire' :\n{state[1]}")
    print(f"Actions légales : {env.get_legal_actions()}")
    
    
    
    
#%%

import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim



# the NN

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 6 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )

    def forward(self, x):
        return self.net(x)


#Hyperparameters 

gamma = 0.99
alpha = 0.0001

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.99995    # so that epsilon gets smaller : exploration -> exploitation
batch_size = 64 
buffer_max = 10000 #
target_update = 50       # copy the weights every target_updates games 
n_parties = 2000


#Initialize

env = Connect4()

policy_net = DQN()          # the nn that we are training
target_net = DQN()           # a previous copy used to avoid making a copy each iteration

# same weight between the two 
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=alpha)

buffer = []

rewards_history = []
losses_history = []


# some functions

def choose_action(state, legal_actions, epsilon):
    """Epsilon-greedy masking illegal mooves """
    if random.random() < epsilon:
        return random.choice(legal_actions)
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # (1, 2, 6, 7)
            q_values = policy_net(state_tensor).squeeze(0)        # (7,)

            # Masking the full columns 
            for a in range(7):
                if a not in legal_actions:
                    q_values[a] = float('-inf')

            return q_values.argmax().item()


def train_step():
    """one training step on the buffer batch"""
    if len(buffer) < batch_size:
        return  # not enought transitions : we wait

    # Sampling a batch 
    batch = random.sample(buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch) #we cut the list 
    # of lists into 5 lists 

    states      = torch.FloatTensor(np.array(states))        # (64, 2, 6, 7)
    actions     = torch.LongTensor(actions)                   # (64,)
    rewards     = torch.FloatTensor(rewards)                  # (64,)
    next_states = torch.FloatTensor(np.array(next_states))   # (64, 2, 6, 7)
    dones       = torch.FloatTensor(dones)                    # (64,)

    # Q(s, a) predicted by the nn
    q_all  = policy_net(states)                                # (64, 7)
    q_pred = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # (64,)

    # Goal : r + gamma * max Q(s', a')
    with torch.no_grad():
        q_next     = target_net(next_states)          # (64, 7)
        q_next_max = q_next.max(dim=1).values         # (64,)
        target     = rewards + gamma * q_next_max * (1 - dones)

    #Loss and backprop 
    loss = nn.SmoothL1Loss()(q_pred, target)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    optimizer.step()

    losses_history.append(loss.item())

#%%
import os
os.chdir("C:/Users/melka/Downloads/RL_connect4/")
# evaluation : play the network against an opponent

def play_against(agent_net, opponent_net, n_games=50, agent_is_player1=True):
    """
    plays agent_net against opponent_net.
    opponent_net = None means random opponent.
    returns the win rate of agent_net.
    """
    wins = 0
    for _ in range(n_games):
        state = env.reset()
        done = False
        while not done:
            legal = env.get_legal_actions()
            
            is_agent_turn = (env.player == 1 and agent_is_player1) or \
                            (env.player == -1 and not agent_is_player1)
            
            if is_agent_turn:
                # agent plays without exploration (epsilon=0)
                with torch.no_grad():
                    st = torch.FloatTensor(state).unsqueeze(0)
                    qv = agent_net(st).squeeze(0)
                    for a in range(7):
                        if a not in legal:
                            qv[a] = float('-inf')
                    action = qv.argmax().item()
            else:
                if opponent_net is None:
                    # random opponent
                    action = random.choice(legal)
                else:
                    # opponent is another network, no exploration
                    with torch.no_grad():
                        st = torch.FloatTensor(state).unsqueeze(0)
                        qv = opponent_net(st).squeeze(0)
                        for a in range(7):
                            if a not in legal:
                                qv[a] = float('-inf')
                        action = qv.argmax().item()
            
            state, reward, done = env.step(action)
        
        # who won ?
        if reward == 1.0:
            winner = env.player  # step doesn't change the player when done=True
            if (winner == 1 and agent_is_player1) or \
               (winner == -1 and not agent_is_player1):
                wins += 1
    
    return wins / n_games


def evaluate(agent_net, random_net, best_net):
    """
    evaluates agent_net against :
      - random player (random_net = initial version with random weights)
      - last good version (best_net)
    50 games as yellow + 50 as red for each opponent.
    """
    # against random
    wr_rand_y = play_against(agent_net, None, n_games=50, agent_is_player1=True)
    wr_rand_r = play_against(agent_net, None, n_games=50, agent_is_player1=False)
    
    # against last good version
    wr_best_y = play_against(agent_net, best_net, n_games=50, agent_is_player1=True)
    wr_best_r = play_against(agent_net, best_net, n_games=50, agent_is_player1=False)
    
    print(f"  vs random  : yellow={wr_rand_y:.0%}  red={wr_rand_r:.0%}  total={(wr_rand_y+wr_rand_r)/2:.0%}")
    print(f"  vs best    : yellow={wr_best_y:.0%}  red={wr_best_r:.0%}  total={(wr_best_y+wr_best_r)/2:.0%}")
    
    return (wr_rand_y + wr_rand_r) / 2, (wr_best_y + wr_best_r) / 2


# save the versions

import copy

# random version (initial) - never modified
random_net = DQN()
random_net.load_state_dict(policy_net.state_dict())

# last good version - updated every 1000 games
best_net = DQN()
best_net.load_state_dict(policy_net.state_dict())

# evaluation history for plotting
eval_games = []
eval_vs_random = []
eval_vs_best = []


# training loop with two phases + game recording + version saving
# phase 1 : train against random opponent
# phase 2 : self-play

import pickle
import copy

# recording a game : stores the grid at each step
def record_game(env, agent_net, opponent_net=None, agent_is_player1=True):
    """
    plays one game and records all grid states + moves.
    opponent_net = None means random opponent.
    returns a dict with grids, moves, result.
    """
    state = env.reset()
    done = False

    grids = [env.grid.copy()]  # initial empty grid
    moves = []

    while not done:
        legal = env.get_legal_actions()

        is_agent_turn = (env.player == 1 and agent_is_player1) or \
                        (env.player == -1 and not agent_is_player1)

        if is_agent_turn:
            with torch.no_grad():
                st = torch.FloatTensor(state).unsqueeze(0)
                qv = agent_net(st).squeeze(0)
                for a in range(7):
                    if a not in legal:
                        qv[a] = float('-inf')
                action = qv.argmax().item()
        else:
            if opponent_net is None:
                action = random.choice(legal)
            else:
                with torch.no_grad():
                    st = torch.FloatTensor(state).unsqueeze(0)
                    qv = opponent_net(st).squeeze(0)
                    for a in range(7):
                        if a not in legal:
                            qv[a] = float('-inf')
                    action = qv.argmax().item()

        player = env.player
        state, reward, done = env.step(action)
        moves.append((player, action))
        grids.append(env.grid.copy())

    if reward == 1.0:
        result = "yellow wins" if env.player == 1 else "red wins"
    elif reward == 0.0:
        result = "draw"
    else:
        result = "unknown"

    return {"grids": grids, "moves": moves, "result": result}


# phases config
phase1_games = 4000
phase2_games = 10000
n_parties = phase1_games + phase2_games

# version saving
saved_versions = {}  # {game_number: state_dict}
saved_versions["random"] = copy.deepcopy(policy_net.state_dict())  # initial random weights

# game recording
recorded_games_phase1 = []
recorded_games_phase2 = []

step_count = 0

for game in range(n_parties):

    state = env.reset()
    done = False
    pending = {1: None, -1: None}

    in_phase1 = game < phase1_games

    while not done:
        player = env.player
        legal_actions = env.get_legal_actions()

        if in_phase1 and player == -1:
            # phase 1 : opponent plays randomly
            action = random.choice(legal_actions)
        else:
            action = choose_action(state, legal_actions, epsilon)

        next_state, reward, done = env.step(action)

        if done:
            buffer.append((state, action, reward, next_state, 1.0))
            opponent = -player
            if pending[opponent] is not None:
                s, a = pending[opponent]
                buffer.append((s, a, -reward, state, 1.0))
            rewards_history.append(reward if player == 1 else -reward)
        else:
            if pending[player] is not None:
                s, a = pending[player]
                buffer.append((s, a, 0.0, state, 0.0))
            pending[player] = (state, action)
            state = next_state

        if len(buffer) > buffer_max:
            buffer.pop(0)

        step_count += 1
        if step_count % 4 == 0:
            train_step()

    # decrease epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # update the target network
    if game % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # record last 5 games of phase 1 (games 9995 to 9999)
    if phase1_games - 5 <= game < phase1_games:
        g = record_game(env, policy_net, opponent_net=None, agent_is_player1=True)
        g["game_number"] = game
        recorded_games_phase1.append(g)

    # record first 5 games of phase 2 (games 10000 to 10004)
    if phase1_games <= game < phase1_games + 5:
        g = record_game(env, policy_net, opponent_net=None, agent_is_player1=True)
        g["game_number"] = game
        recorded_games_phase2.append(g)

    # save version every 2000 games + evaluation
    if game % 2000 == 0 and game > 0:
        # save version
        saved_versions[game] = copy.deepcopy(policy_net.state_dict())
        torch.save(policy_net.state_dict(), f"connect4_v{game}.pth")

        # evaluate
        phase_str = "phase1 (vs random)" if in_phase1 else "phase2 (self-play)"
        print(f"\n--- evaluation at game {game} ({phase_str}) ---")
        wr_rand, wr_best = evaluate(policy_net, random_net, best_net)
        eval_games.append(game)
        eval_vs_random.append(wr_rand)
        eval_vs_best.append(wr_best)

        best_net.load_state_dict(policy_net.state_dict())
        print(f"  version {game} saved.\n")

    # quick display
    if game % 500 == 0:
        phase_str = "phase1" if in_phase1 else "phase2"
        recent = rewards_history[-200:] if len(rewards_history) >= 200 else rewards_history
        avg = np.mean(recent) if recent else 0
        avg_loss = np.mean(losses_history[-500:]) if losses_history else 0
        print(f"game {game:>5d} | {phase_str} | eps={epsilon:.3f} | "
              f"avg reward={avg:+.3f} | loss={avg_loss:.4f} | "
              f"buffer={len(buffer)}")


# final save
saved_versions["final"] = copy.deepcopy(policy_net.state_dict())
torch.save(policy_net.state_dict(), "connect4_final.pth")

# final evaluation
print(f"\n--- final evaluation (game {n_parties}) ---")
wr_rand, wr_best = evaluate(policy_net, random_net, best_net)
eval_games.append(n_parties)
eval_vs_random.append(wr_rand)
eval_vs_best.append(wr_best)

# save recorded games
with open("games_phase1.pkl", "wb") as f:
    pickle.dump(recorded_games_phase1, f)
with open("games_phase2.pkl", "wb") as f:
    pickle.dump(recorded_games_phase2, f)
print(f"saved {len(recorded_games_phase1)} phase1 games and {len(recorded_games_phase2)} phase2 games")

# save all versions dict
torch.save(saved_versions, "all_versions.pth")
print(f"saved {len(saved_versions)} versions")

print("final model saved!")

# win rate curves
plt.figure(figsize=(10, 5))
plt.plot(eval_games, eval_vs_random, 'o-', label='vs random')
plt.plot(eval_games, eval_vs_best, 's-', label='vs previous version')
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=phase1_games, color='red', linestyle='--', alpha=0.5, label='phase1 -> phase2')
plt.xlabel("game")
plt.ylabel("win rate")
plt.title("agent progression")
plt.legend()
plt.grid(True)
plt.ylim(0, 1)
plt.savefig("winrate_curve.png")
plt.show()

#%%

# display_game.py
# displays saved connect 4 games using pygame
# usage : python display_game.py saved_games.pkl
# controls : RIGHT arrow = next move, LEFT = previous move, UP/DOWN = change game

import pygame
import pickle
import sys

# colors
BLUE = (30, 60, 180)
BLACK = (0, 0, 0)
RED = (220, 40, 40)
YELLOW = (240, 220, 40)
WHITE = (255, 255, 255)
GRAY = (180, 180, 180)

# dimensions
CELL = 100        # size of each cell in pixels
COLS = 7
ROWS = 6
WIDTH = COLS * CELL
HEIGHT = (ROWS + 1) * CELL  # extra row for info at the top
RADIUS = CELL // 2 - 8


def draw_board(screen, grid, move_index, total_moves, game_index, total_games, info_text=""):
    # background
    screen.fill(BLACK)

    # info bar at the top
    font = pygame.font.SysFont("arial", 28)
    text = font.render(f"game {game_index+1}/{total_games}  |  move {move_index}/{total_moves}  {info_text}",
                       True, WHITE)
    screen.blit(text, (20, 20))

    # board (blue rectangle)
    board_y = CELL  # start below info bar
    pygame.draw.rect(screen, BLUE, (0, board_y, WIDTH, ROWS * CELL))

    # pieces
    for row in range(ROWS):
        for col in range(COLS):
            center_x = col * CELL + CELL // 2
            # row 0 = bottom, so we draw it at the bottom of the screen
            center_y = board_y + (ROWS - 1 - row) * CELL + CELL // 2

            val = grid[row, col]
            if val == 1:
                color = YELLOW
            elif val == -1:
                color = RED
            else:
                color = BLACK

            pygame.draw.circle(screen, color, (center_x, center_y), RADIUS)

    # column numbers
    small_font = pygame.font.SysFont("arial", 22)
    for col in range(COLS):
        num = small_font.render(str(col), True, GRAY)
        screen.blit(num, (col * CELL + CELL // 2 - 6, HEIGHT - 30))

    pygame.display.flip()


def replay_games(games):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("connect 4 replay")

    game_index = 0
    move_index = 0

    running = True
    while running:
        game = games[game_index]
        grids = game["grids"]       # list of grid states (numpy arrays)
        moves = game["moves"]       # list of (player, column)
        result = game["result"]     # "yellow wins", "red wins", "draw"

        # clamp move index
        move_index = max(0, min(move_index, len(grids) - 1))

        # info text
        if move_index == len(grids) - 1:
            info = f"  ->  {result}"
        elif move_index > 0:
            player, col = moves[move_index - 1]
            color = "yellow" if player == 1 else "red"
            info = f"  ({color} played col {col})"
        else:
            info = ""

        draw_board(screen, grids[move_index], move_index, len(grids) - 1,
                   game_index, len(games), info)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    move_index += 1
                elif event.key == pygame.K_LEFT:
                    move_index -= 1
                elif event.key == pygame.K_DOWN:
                    game_index = (game_index + 1) % len(games)
                    move_index = 0
                elif event.key == pygame.K_UP:
                    game_index = (game_index - 1) % len(games)
                    move_index = 0
                elif event.key == pygame.K_ESCAPE:
                    running = False

        pygame.time.wait(50)

    pygame.quit()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage : python display_game.py saved_games.pkl")
        sys.exit(1)

    filepath = sys.argv[1]
    with open(filepath, "rb") as f:
        games = pickle.load(f)

    print(f"loaded {len(games)} games")
    print("controls : RIGHT/LEFT = next/prev move, UP/DOWN = change game, ESC = quit")
    replay_games(games)


#%%

# benchmark.py
# play any two saved versions against each other
# usage : python benchmark.py version_a.pth version_b.pth --n_games 100
# use "random" instead of a .pth file for a random player

import argparse
import numpy as np
import random
import torch
import torch.nn as nn

from connect4_env import Connect4


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 6 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )

    def forward(self, x):
        return self.net(x)


def load_agent(path):
    if path == "random":
        return None
    net = DQN()
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    return net


def choose(net, state, legal):
    if net is None:
        return random.choice(legal)
    with torch.no_grad():
        st = torch.FloatTensor(state).unsqueeze(0)
        qv = net(st).squeeze(0)
        for a in range(7):
            if a not in legal:
                qv[a] = float('-inf')
        return qv.argmax().item()


def play_match(agent_a, agent_b, env, n_games=100):
    # agent_a plays yellow (player 1), agent_b plays red (player -1)
    wins_a = 0
    wins_b = 0
    draws = 0

    for _ in range(n_games):
        state = env.reset()
        done = False
        while not done:
            legal = env.get_legal_actions()
            if env.player == 1:
                action = choose(agent_a, state, legal)
            else:
                action = choose(agent_b, state, legal)
            state, reward, done = env.step(action)

        if reward == 0:
            draws += 1
        elif env.player == 1:
            wins_a += 1
        else:
            wins_b += 1

    return wins_a, wins_b, draws


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("version_a", help="path to .pth file or 'random'")
    parser.add_argument("version_b", help="path to .pth file or 'random'")
    parser.add_argument("--n_games", type=int, default=100)
    args = parser.parse_args()

    agent_a = load_agent(args.version_a)
    agent_b = load_agent(args.version_b)
    env = Connect4()

    # agent_a as yellow, agent_b as red
    wa1, wb1, d1 = play_match(agent_a, agent_b, env, args.n_games)

    # swap : agent_a as red, agent_b as yellow
    wb2, wa2, d2 = play_match(agent_b, agent_a, env, args.n_games)

    total_a = wa1 + wa2
    total_b = wb1 + wb2
    total_d = d1 + d2
    total = 2 * args.n_games

    name_a = args.version_a
    name_b = args.version_b

    print(f"\nresults over {total} games ({args.n_games} each side):")
    print(f"  {name_a:>30s} : {total_a} wins ({total_a/total:.0%})")
    print(f"  {name_b:>30s} : {total_b} wins ({total_b/total:.0%})")
    print(f"  {'draws':>30s} : {total_d} ({total_d/total:.0%})")

    print(f"\ndetail :")
    print(f"  {name_a} as yellow vs {name_b} as red : {wa1}W {wb1}L {d1}D")
    print(f"  {name_a} as red vs {name_b} as yellow : {wa2}W {wb2}L {d2}D")
