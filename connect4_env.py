# connect4_env.py
# connect 4 environment for RL
# grid[0,0] = bottom left, grid[5,6] = top right

import numpy as np


class Connect4:
    def __init__(self):
        self.grid = np.zeros((6, 7), dtype=np.float32)
        self.player = 1

    def reset(self):
        self.grid = np.zeros((6, 7), dtype=np.float32)
        self.player = 1
        return self.get_state()

    def step(self, column):
        line = -1
        for l in range(6):
            if self.grid[l, column] == 0:
                line = l
                break

        if line == -1:
            raise ValueError(f"column {column} full!")

        self.grid[line, column] = self.player

        if self._check_win(line, column):
            return self.get_state(), 1.0, True

        if np.all(self.grid != 0):
            return self.get_state(), 0.0, True

        self.player *= -1
        return self.get_state(), 0.0, False

    def _check_win(self, line, column):
        p = self.grid[line, column]
        for dl, dc in [(0,1), (1,0), (1,1), (1,-1)]:
            count = 1
            for k in range(1, 4):
                l, c = line + k*dl, column + k*dc
                if 0 <= l < 6 and 0 <= c < 7 and self.grid[l, c] == p:
                    count += 1
                else:
                    break
            for k in range(1, 4):
                l, c = line - k*dl, column - k*dc
                if 0 <= l < 6 and 0 <= c < 7 and self.grid[l, c] == p:
                    count += 1
                else:
                    break
            if count >= 4:
                return True
        return False

    def get_legal_actions(self):
        return [c for c in range(7) if self.grid[5, c] == 0]

    def get_state(self):
        my = (self.grid == self.player).astype(np.float32)
        opp = (self.grid == -self.player).astype(np.float32)
        return np.stack([my, opp], axis=0)

    def copy(self):
        # returns a deep copy of the environment (useful for recording)
        new = Connect4()
        new.grid = self.grid.copy()
        new.player = self.player
        return new

    def afficher(self):
        symboles = {0: ".", 1: "X", -1: "O"}
        print()
        for line in range(5, -1, -1):
            row = ""
            for col in range(7):
                row += symboles[self.grid[line, col]] + " "
            print(row)
        print("0 1 2 3 4 5 6")
        print(f"turn: {'X' if self.player == 1 else 'O'}\n")
