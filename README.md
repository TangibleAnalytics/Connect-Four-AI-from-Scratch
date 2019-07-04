# Connect-Four-AI-from-Scratch

In many real-world AI problems, the AI agent has no prior knowledge about the mechanisms within the environment. To an AI agent, the environment is a black-box system which can yield rewards through interaction. In order to achieve maximal rewards, an AI agent must create a model for the environment and then use that model to tune the internal policy that it uses to decide actions based upon the state of the environment.

This project will serve as a toy example of an AI learning procedure which facilitates a given AI agent to increasingly bring about maximal rewards for itself. Two AI agents will create an adversarial system by playing against each other in a simulation of the classic board game known as Connect Four.

During each move, an AI agent will consult only with its internal policy to map the game board state to an appropriate action. AI agents will NOT attempt, in any fashion, to simulate game moves and evaluate resulting game board states before deciding upon an action. Doing so would require a treatment of the environment as a white-box system and therefore contradict much of the motivation for this project.

Rewards will be assigned only at the end of each round. If an AI agent has won the given round, it will receive a reward value of plus one. For losing, it will receive a reward value of minus one. For rounds ending in a tie, each AI agent will receive a reward value of zero.

The project will be deemed successful if it can be shown that AI agents can use past experiences to gain an advantage over an opponent. The easiest way to test this is by allowing two AI agents to play the game with randomly initialized internal policies, but only allowing one of them to learn from past experiences and update its internal policy on a routine basis. The resulting empirical win/loss ratio vs. games played plot should show increasing values for the AI agent that is subjected to routine learning cycles.
