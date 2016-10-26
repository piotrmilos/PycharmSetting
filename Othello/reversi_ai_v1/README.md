# reversi_ai
An attempt to build multiple AI agents to play the game of reversi.

The ultimate goal is to create an agent that combines reinforcement learning with Monte Carlo tree search.
So far, the MCTS agent is implemented and working well.  A Q-learning agent is partly implemented and in progress.

To run, use run_game.py, or create your own top-level game runner.
You just have to create a Reversi object and pass it two agents of your choosing.

Example of running using a HumanAgent (controlled by me) and the MonteCarloAgent:

    ~/reversi_ai(master ✗) ./run_game.py BlackAgent=human WhiteAgent=monte_carlo sim_time=1
    About to run 1 games, black as HumanAgent, white as MonteCarloAgent.
    starting game 1 of 1
    7| - - - - - - - - 
    6| - - - - - - - - 
    5| - - - - - - - - 
    4| - - - O X - - - 
    3| - - - X O - - - 
    2| - - - - - - - - 
    1| - - - - - - - - 
    0| - - - - - - - - 
      ----------------
       0 1 2 3 4 5 6 7 
    
    Enter a move x,y: 2,4
    7| - - - - - - - - 
    6| - - - - - - - - 
    5| - - - - - - - - 
    4| - - X X X - - - 
    3| - - - X O - - - 
    2| - - - - - - - - 
    1| - - - - - - - - 
    0| - - - - - - - - 
      ----------------
       0 1 2 3 4 5 6 7 
    Black plays at (2, 4)
    
    (2, 5): (133/190)
    (4, 5): (196/263)
    (2, 3): (222/294)
    747 simulations performed.
    7| - - - - - - - - 
    6| - - - - - - - - 
    5| - - - - - - - - 
    4| - - X X X - - - 
    3| - - O O O - - - 
    2| - - - - - - - - 
    1| - - - - - - - - 
    0| - - - - - - - - 
      ----------------
       0 1 2 3 4 5 6 7 
    White plays at (2, 3)
    
    Enter a move x,y: 2,2
    7| - - - - - - - - 
    6| - - - - - - - - 
    5| - - - - - - - - 
    4| - - X X X - - - 
    3| - - X X O - - - 
    2| - - X - - - - - 
    1| - - - - - - - - 
    0| - - - - - - - - 
      ----------------
       0 1 2 3 4 5 6 7 
    Black plays at (2, 2)
    
    (2, 5): (92/142)
    (1, 3): (186/255)
    (4, 5): (328/422)
    753 simulations performed.
    7| - - - - - - - - 
    6| - - - - - - - - 
    5| - - - - O - - - 
    4| - - X X O - - - 
    3| - - X X O - - - 
    2| - - X - - - - - 
    1| - - - - - - - - 
    0| - - - - - - - - 
      ----------------
       0 1 2 3 4 5 6 7 
    White plays at (4, 5)
    
    Enter a move x,y: 5,3
    7| - - - - - - - - 
    6| - - - - - - - - 
    5| - - - - O - - - 
    4| - - X X O - - - 
    3| - - X X X X - - 
    2| - - X - - - - - 
    1| - - - - - - - - 
    0| - - - - - - - - 
      ----------------
       0 1 2 3 4 5 6 7 
    Black plays at (5, 3)
    
    (4, 2): (214/270)
    (6, 2): (139/186)
    (1, 2): (105/147)
    (1, 1): (88/128)
    (1, 4): (78/115)
    766 simulations performed.
    7| - - - - - - - - 
    6| - - - - - - - - 
    5| - - - - O - - - 
    4| - - X X O - - - 
    3| - - X X O X - - 
    2| - - X - O - - - 
    1| - - - - - - - - 
    0| - - - - - - - - 
      ----------------
       0 1 2 3 4 5 6 7 
    White plays at (4, 2)
