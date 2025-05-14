import random

class Nim():
    def __init__(self, initial=[4, 4, 4, 4]):
        self.piles = initial.copy()
        self.player = 0  # Player 0 starts
        self.winner = None

    @classmethod
    def available_actions(cls, piles):
        actions = set()
        for i, pile in enumerate(piles):
            for j in range(1, pile + 1):
                actions.add((i, j))
        return actions

    @classmethod
    def other_player(cls, player):
        return 0 if player == 1 else 1

    def switch_player(self):
        self.player = Nim.other_player(self.player)

    def move(self, action):
        pile, count = action
        self.piles[pile] -= count
        self.switch_player()
        if all(pile == 0 for pile in self.piles):
            self.winner = self.player



class NimAI():
    def __init__(self, alpha=0.5, epsilon=0.1):
        self.q = dict()  # Q-value table
        self.q[(0, 0, 0, 2), (3, 2)] = -1 # Test Q-Value 
        self.q[(0, 0, 0, 2), (3, 1)] = 10 # Test Q-Value 
        
        self.alpha = alpha  # Learning rate
        self.epsilon = epsilon  # Exploration rate

    def update(self, old_state, action, new_state, reward):
        old_q = self.get_q_value(old_state, action)
        best_future_q, best_action = self.best_future_reward(new_state)
        self.update_q_value(old_state, action, old_q, reward, best_future_q)

    def get_q_value(self, state, action):
        """
    Return the Q-value for a given state-action pair.
    
    Parameters:
        state (list): The current game state.
        action (tuple): The action being evaluated.

    Returns:
        float: The Q-value associated with the (state, action) pair. 
               Returns 0 if the pair is not yet in the Q-table.
    """
        state = tuple(state)
        for i in self.q:
            if i[0] == state and i[1] == action:
                return self.q[i]
        return 0

    def update_q_value(self, state, action, old_q, reward, future_q):
        """
        Update the Q-value for a state-action pair using the Q-learning formula.
        """
        state = tuple(state)  # Convert state to a tuple
        self.q[(state, action)] = old_q + self.alpha * (reward + future_q - old_q)
                
    def best_future_reward(self, state):
        """
        Determine the highest Q-value among all possible actions in a given state.
        
        Parameters:
            state (list): The state for which to compute the best future reward.
            
        Returns:
            tuple: The highest Q-value among available actions and the corresponding action.
                   Returns (0, None) if no actions are available.
        """
        state = tuple(state)
        actions = Nim.available_actions(state)
        if not actions:
            return 0, None  # Return a tuple instead of a single value
        best_q = float('-inf')
        best_action = None
        for action in actions:
            q_value = self.get_q_value(state, action)
            if q_value > best_q:
                best_q = q_value
                best_action = action
        return best_q, best_action

    def choose_action(self, state, epsilon=True):
        """
    Choose an action for the given state using an epsilon-greedy strategy.
    
    Parameters:
        state (list): The current game state.
        epsilon (bool): If True, use epsilon-greedy exploration; otherwise, choose the best action.
    
    Returns:
        tuple: The chosen action from the available actions.
    """
        if epsilon and random.random() < self.epsilon:
            actions = Nim.available_actions(state)
            return random.choice(list(actions))
        else:
            best_q, action = self.best_future_reward(state)
            if action is None:
                actions = Nim.available_actions(state)
                return random.choice(list(actions))
            return action

def train(n):
    player = NimAI()

    for i in range(n):
        game = Nim([4, 4, 4, 4])
        last_move = {0: {"state": None, "action": None}, 1: {"state": None, "action": None}}

        while True:
            state = game.piles.copy()
            action = player.choose_action(state)
            last_move[game.player]["state"] = state
            last_move[game.player]["action"] = action

            game.move(action)
            new_state = game.piles.copy()

            if game.winner is not None:
                player.update(state, action, new_state, -1)
                player.update(last_move[game.player]["state"], last_move[game.player]["action"], new_state, 1)
                break
            elif last_move[game.player]["state"] is not None:
                player.update(last_move[game.player]["state"], last_move[game.player]["action"], new_state, 0)

    return player



if __name__ == "__main__":
    import play