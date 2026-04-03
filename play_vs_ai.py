# play_vs_ai.py
# play connect 4 against the trained model
# usage: python Downloads\RL_connect4\play_vs_ai.py Downloads\RL_connect4\best_model.pth red 
# click on a column to play, R to restart and  ESC to quit

import pygame
import sys
import numpy as np
import torch
import torch.nn as nn

from connect4_env import Connect4
from dqn_model import DQN, choose_action

# colors
BLUE = (30, 60, 180)
BLACK = (0, 0, 0)
RED = (220, 40, 40)
YELLOW = (240, 220, 40)
WHITE = (255, 255, 255)
GRAY = (180, 180, 180)
GREEN = (40, 200, 40)

CELL = 100
COLS = 7
ROWS = 6
WIDTH = COLS * CELL
HEIGHT = (ROWS + 2) * CELL
RADIUS = CELL // 2 - 8


def draw(screen, env, human_color, message="", hover_col=-1):
    screen.fill(BLACK)

    font = pygame.font.SysFont("arial", 30)

    if env.player == human_color:
        color_name = "yellow" if human_color == 1 else "red"
        turn_text = f"your turn ({color_name})"
    else:
        turn_text = "AI thinking..."

    text = font.render(turn_text, True, WHITE)
    screen.blit(text, (20, 20))

    if message:
        msg = font.render(message, True, GREEN)
        screen.blit(msg, (20, 55))

    # hover preview
    if hover_col >= 0 and env.player == human_color:
        cx = hover_col * CELL + CELL // 2
        preview_color = YELLOW if human_color == 1 else RED
        pygame.draw.circle(screen, preview_color, (cx, CELL + CELL // 2 - 30), RADIUS // 2)

    # board
    board_y = CELL + 20
    pygame.draw.rect(screen, BLUE, (0, board_y, WIDTH, ROWS * CELL))

    for row in range(ROWS):
        for col in range(COLS):
            cx = col * CELL + CELL // 2
            cy = board_y + (ROWS - 1 - row) * CELL + CELL // 2

            val = env.grid[row, col]
            if val == 1:
                color = YELLOW
            elif val == -1:
                color = RED
            else:
                color = BLACK
            pygame.draw.circle(screen, color, (cx, cy), RADIUS)

    # column numbers
    small_font = pygame.font.SysFont("arial", 22)
    for col in range(COLS):
        num = small_font.render(str(col), True, GRAY)
        screen.blit(num, (col * CELL + CELL // 2 - 6, HEIGHT - 30))

    pygame.display.flip()


def ai_play(env, ai_net, state):
    legal = env.get_legal_actions()
    action = choose_action(state, legal, ai_net, epsilon=0)
    state, reward, done = env.step(action)
    return state, reward, done, action


def main():
    if len(sys.argv) < 2:
        print("usage: python play_vs_ai.py best_model.pth [yellow/red]")
        sys.exit(1)

    # load model
    model_path = sys.argv[1]
    ai_net = DQN()
    ai_net.load_state_dict(torch.load(model_path, map_location="cpu"))
    ai_net.eval()
    print(f"loaded model from {model_path}")

    # choose color
    human_color = 1  # yellow by default
    if len(sys.argv) >= 3 and sys.argv[2] == "red":
        human_color = -1
    color_name = "yellow" if human_color == 1 else "red"
    print(f"you are {color_name}")

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("connect 4 vs AI")

    env = Connect4()
    state = env.reset()

    message = f"you are {color_name}. press R to restart."
    game_over = False
    hover_col = -1

    # if human is red, AI plays first
    if human_color == -1:
        state, reward, done, ai_col = ai_play(env, ai_net, state)
        message = f"AI played column {ai_col}"

    draw(screen, env, human_color, message, hover_col)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEMOTION and not game_over:
                hover_col = event.pos[0] // CELL
                draw(screen, env, human_color, message, hover_col)

            elif event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                # only accept clicks when it's human's turn
                if env.player != human_color:
                    continue

                col = event.pos[0] // CELL
                if col < 0 or col > 6:
                    continue

                legal = env.get_legal_actions()
                if col not in legal:
                    message = "column full!"
                    draw(screen, env, human_color, message, hover_col)
                    continue

                # human plays
                state, reward, done = env.step(col)
                message = ""
                draw(screen, env, human_color, message, hover_col)

                if done:
                    if reward == 1.0:
                        message = "you win!"
                    else:
                        message = "draw!"
                    game_over = True
                    draw(screen, env, human_color, message, -1)
                    continue

                # AI plays
                pygame.time.wait(300)
                state, reward, done, ai_col = ai_play(env, ai_net, state)
                message = f"AI played column {ai_col}"

                if done:
                    if reward == 1.0:
                        message = "AI wins!"
                    else:
                        message = "draw!"
                    game_over = True

                draw(screen, env, human_color, message, hover_col)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    env.reset()
                    state = env.get_state()
                    game_over = False
                    message = f"new game! you are {color_name}."
                    # if human is red, AI plays first
                    if human_color == -1:
                        state, reward, done, ai_col = ai_play(env, ai_net, state)
                        message = f"new game! AI played column {ai_col}"
                    draw(screen, env, human_color, message, -1)
                elif event.key == pygame.K_ESCAPE:
                    running = False

        pygame.time.wait(30)

    pygame.quit()


if __name__ == "__main__":
    main()
