# evaluate.py
# benchmark two versions, display games, play against the bot
# usage:
#   python evaluate.py benchmark connect4_v10000.pth random --n 200
#   python evaluate.py benchmark connect4_v10000.pth connect4_v5000.pth --n 200
#   python evaluate.py display sample_games.pkl
#   python evaluate.py play connect4_final.pth
#   python evaluate.py record connect4_final.pth random --n 5 --out games.pkl

import argparse
import numpy as np
import random
import pickle
import sys

import torch
import torch.nn as nn

from connect4_env import Connect4
from dqn_model import DQN, choose_action


# load a network from file, or None for random
def load_agent(path):
    if path == "random":
        return None
    net = DQN()
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    net.eval()
    return net


# pick an action for a given agent (None = random)
def pick_action(net, state, legal):
    if net is None:
        return random.choice(legal)
    return choose_action(state, legal, net, epsilon=0)


# benchmark two agents
def benchmark(agent_a, agent_b, n_games=100):
    env = Connect4()
    results = {"a_wins": 0, "b_wins": 0, "draws": 0,
               "a_yellow_w": 0, "a_yellow_l": 0, "a_yellow_d": 0,
               "a_red_w": 0, "a_red_l": 0, "a_red_d": 0}

    for i in range(n_games * 2):
        a_is_yellow = (i < n_games)
        state = env.reset()
        done = False
        while not done:
            legal = env.get_legal_actions()
            if env.player == 1:
                net = agent_a if a_is_yellow else agent_b
            else:
                net = agent_b if a_is_yellow else agent_a
            action = pick_action(net, state, legal)
            state, reward, done = env.step(action)

        if reward == 0:
            results["draws"] += 1
            if a_is_yellow:
                results["a_yellow_d"] += 1
            else:
                results["a_red_d"] += 1
        else:
            winner_is_p1 = (env.player == 1)
            a_won = (winner_is_p1 and a_is_yellow) or (not winner_is_p1 and not a_is_yellow)
            if a_won:
                results["a_wins"] += 1
                if a_is_yellow:
                    results["a_yellow_w"] += 1
                else:
                    results["a_red_w"] += 1
            else:
                results["b_wins"] += 1
                if a_is_yellow:
                    results["a_yellow_l"] += 1
                else:
                    results["a_red_l"] += 1

    return results


# record games for display
def record_games(agent_a, agent_b, n_games=5):
    env = Connect4()
    games = []
    for i in range(n_games):
        state = env.reset()
        done = False
        grids = [env.grid.copy()]
        moves = []
        while not done:
            legal = env.get_legal_actions()
            if env.player == 1:
                action = pick_action(agent_a, state, legal)
            else:
                action = pick_action(agent_b, state, legal)
            player = env.player
            state, reward, done = env.step(action)
            moves.append((player, action))
            grids.append(env.grid.copy())
        if reward == 1.0:
            result = "yellow wins" if env.player == 1 else "red wins"
        else:
            result = "draw"
        games.append({"grids": grids, "moves": moves, "result": result})
    return games


# display games with pygame
def display_games(games):
    try:
        import pygame
    except ImportError:
        print("pygame not installed. run: pip install pygame")
        return

    CELL = 100
    COLS, ROWS = 7, 6
    WIDTH = COLS * CELL
    HEIGHT = (ROWS + 1) * CELL
    RADIUS = CELL // 2 - 8

    BLUE = (30, 60, 180)
    BLACK = (0, 0, 0)
    RED = (220, 40, 40)
    YELLOW = (240, 220, 40)
    WHITE = (255, 255, 255)
    GRAY = (180, 180, 180)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("connect 4 replay")
    font = pygame.font.SysFont("arial", 26)
    small_font = pygame.font.SysFont("arial", 20)

    gi, mi = 0, 0
    running = True

    while running:
        game = games[gi]
        grids = game["grids"]
        moves = game["moves"]
        result = game["result"]
        mi = max(0, min(mi, len(grids) - 1))

        screen.fill(BLACK)

        # info
        info = f"game {gi+1}/{len(games)} | move {mi}/{len(grids)-1}"
        if mi == len(grids) - 1:
            info += f" | {result}"
        elif mi > 0:
            p, c = moves[mi-1]
            info += f" | {'yellow' if p==1 else 'red'} -> col {c}"
        screen.blit(font.render(info, True, WHITE), (10, 15))

        # board
        by = CELL
        pygame.draw.rect(screen, BLUE, (0, by, WIDTH, ROWS * CELL))
        for row in range(ROWS):
            for col in range(COLS):
                cx = col * CELL + CELL // 2
                cy = by + (ROWS - 1 - row) * CELL + CELL // 2
                val = grids[mi][row, col]
                color = YELLOW if val == 1 else (RED if val == -1 else BLACK)
                pygame.draw.circle(screen, color, (cx, cy), RADIUS)

        for col in range(COLS):
            screen.blit(small_font.render(str(col), True, GRAY),
                       (col * CELL + CELL//2 - 5, HEIGHT - 28))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT: mi += 1
                elif event.key == pygame.K_LEFT: mi -= 1
                elif event.key == pygame.K_DOWN: gi = (gi+1) % len(games); mi = 0
                elif event.key == pygame.K_UP: gi = (gi-1) % len(games); mi = 0
                elif event.key == pygame.K_ESCAPE: running = False

        pygame.time.wait(50)

    pygame.quit()


# play against the bot interactively
def play_interactive(net):
    env = Connect4()
    state = env.reset()
    env.afficher()

    print("you are RED (O). type column number 0-6.")
    print("the bot plays YELLOW (X).\n")

    done = False
    while not done:
        legal = env.get_legal_actions()

        if env.player == 1:
            # bot plays
            action = pick_action(net, state, legal)
            print(f"bot plays column {action}")
        else:
            # human plays
            while True:
                try:
                    action = int(input("your move (0-6): "))
                    if action in legal:
                        break
                    print(f"column {action} not legal. try: {legal}")
                except ValueError:
                    print("enter a number 0-6")

        state, reward, done = env.step(action)
        env.afficher()

    if reward == 1.0:
        winner = "bot" if env.player == 1 else "you"
        print(f"{winner} won!")
    else:
        print("draw!")


# main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    # benchmark
    p_bench = sub.add_parser("benchmark")
    p_bench.add_argument("a", help=".pth file or 'random'")
    p_bench.add_argument("b", help=".pth file or 'random'")
    p_bench.add_argument("--n", type=int, default=100)

    # display
    p_disp = sub.add_parser("display")
    p_disp.add_argument("file", help=".pkl file with recorded games")

    # play
    p_play = sub.add_parser("play")
    p_play.add_argument("model", help=".pth file")

    # record
    p_rec = sub.add_parser("record")
    p_rec.add_argument("a", help=".pth file or 'random'")
    p_rec.add_argument("b", help=".pth file or 'random'")
    p_rec.add_argument("--n", type=int, default=5)
    p_rec.add_argument("--out", default="recorded_games.pkl")

    args = parser.parse_args()

    if args.command == "benchmark":
        a = load_agent(args.a)
        b = load_agent(args.b)
        r = benchmark(a, b, args.n)
        total = 2 * args.n
        print(f"\nresults over {total} games ({args.n} each side):")
        print(f"  {args.a}: {r['a_wins']} wins ({r['a_wins']/total:.0%})")
        print(f"  {args.b}: {r['b_wins']} wins ({r['b_wins']/total:.0%})")
        print(f"  draws: {r['draws']} ({r['draws']/total:.0%})")
        print(f"\ndetail:")
        print(f"  {args.a} as yellow: {r['a_yellow_w']}W {r['a_yellow_l']}L {r['a_yellow_d']}D")
        print(f"  {args.a} as red:    {r['a_red_w']}W {r['a_red_l']}L {r['a_red_d']}D")

    elif args.command == "display":
        with open(args.file, "rb") as f:
            games = pickle.load(f)
        print(f"loaded {len(games)} games. RIGHT/LEFT=move, UP/DOWN=game, ESC=quit")
        display_games(games)

    elif args.command == "play":
        net = load_agent(args.model)
        play_interactive(net)

    elif args.command == "record":
        a = load_agent(args.a)
        b = load_agent(args.b)
        games = record_games(a, b, args.n)
        with open(args.out, "wb") as f:
            pickle.dump(games, f)
        print(f"recorded {args.n} games to {args.out}")
        for i, g in enumerate(games):
            print(f"  game {i+1}: {g['result']} ({len(g['moves'])} moves)")

    else:
        parser.print_help()
