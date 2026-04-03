# connect4-deeprl-engine
this repo is a project I made myself. The goal of it is to create a connect 4 engine that would be able to be really good at the end. 
GOALS : 
This small personal project is a connect4 engine using Deep reinforcment learning and in particular Deep Q Networks.
The goal for me is to have an engine that is good enough to beat me (i'm good like would be a 1200 elo at chess (i'm 1500 elo at chess)). So it needs to see basic tactics and traps but not have a strong theory nor strategic overview of the game. 

The goal is not to have a perfect player, because the game is solved (a perfect player can always win with the yellows pieces) and beacuse it would require to much computing powers and advanced techniques I don't have the time to explore. 

The goal for me is to learn more about RL and in particular about CS technics to optimise thoses kinds of computation. I'm not using anything more than my laptop (and in particular no GPUs).

Files : 
- connect4_env.py is the environment of the connect4, creating the grid, looking at the possible mooves, checking wins and how mooves would retranscribe in the grid.
- dqn_model.py is the class creating the neural network, implementing the loss and learning part
- train.py is the "important file" implementing the training loop, with the games being played here 
- play_vs_ai.py is the file for the visualisation of games, and where you can actually play real games against versions of the agent (go see the section [????????] if you want to play by youreself against some version of it)
- documents (versions of the ai) , training curves, elo comparaisons ...

