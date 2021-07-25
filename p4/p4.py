import math
import random
import time

TOKENS = ["O", "X"]

WIDTH = 7
HEIGHT = 6
WIN = 4

class P4():

    def __init__(self):
        """
        Initialize game board.
        Each game board has
            - `piles`: a list of how many elements remain in each pile
            - `player`: 0 or 1 to indicate which player's turn
            - `winner`: None, 0, or 1 to indicate who the winner is
        """
        self.piles = [''] * WIDTH
        self.player = 0
        self.winner = None

    @classmethod
    def available_actions(cls, piles):
        """
        P4.available_actions(piles) takes a `piles` list as input
        and returns all of the available actions `(i, j)` in that state.
        """
        return {i for i, p in enumerate(piles[1:]) if len(p) < HEIGHT}

    @classmethod
    def other_player(cls, player):
        """
        P4.other_player(player) returns the player that is not
        `player`. Assumes `player` is either 0 or 1.
        """
        return 1 if player == 0 else 0

    def switch_player(self):
        """
        Switch the current player to the other player.
        """
        self.player = P4.other_player(self.player)

    def move(self, action):
        """
        Make the move `action` for the current player.
        """
        # Check for errors
        if self.winner is not None:
            raise Exception("Game already won")
        elif action < 0 or action >= WIDTH:
            raise Exception("Invalid pile")
        elif len(self.piles[action]) >= HEIGHT:
            raise Exception("Full pile")

        # Update pile
        token = TOKENS[self.player]
        self.piles[action] += token

        # for p in self.piles: print(str(p))

        # Check if player wins
        def at(piles, i, j):
            if i < 0 or i >= len(piles): return None
            p = piles[i]
            if j < 0 or j >= len(p): return None
            return p[j]
        
        i = action
        j = len(self.piles[i]) - 1
        b = j
        while at(self.piles, i, b - 1) == token: b = b - 1
        e = j
        if (e - b) + 1 >= WIN: self.winner = self.player
        if self.winner is None:
            b = -1
            while at(self.piles, i + b, j) == token: b -= 1
            e = 1
            while at(self.piles, i + e, j) == token: e += 1
            if (e - b) - 1 >= WIN: self.winner = self.player
        if self.winner is None:
            b = -1
            while at(self.piles, i + b, j + b) == token: b -= 1
            e = 1
            while at(self.piles, i + b, j + b) == token: e += 1
            if (e - b) - 1 >= WIN: self.winner = self.player
        if self.winner is None:
            b = -1
            while at(self.piles, i + b, j - b) == token: b -= 1
            e = 1
            while at(self.piles, i + b, j - b) == token: e += 1
            if (e - b) - 1 >= WIN: self.winner = self.player

        if all([len(p) >= HEIGHT for p in self.piles]):
            self.winner = -1
        self.switch_player()


class P4AI():

    def __init__(self, alpha=0.5, epsilon=0.1):
        """
        Initialize AI with an empty Q-learning dictionary,
        an alpha (learning) rate, and an epsilon rate.

        The Q-learning dictionary maps `(state, action)`
        pairs to a Q-value (a number).
         - `state` is a tuple of remaining piles, e.g. (1, 1, 4, 4)
         - `action` is a tuple `(i, j)` for an action
        """
        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon

    def update(self, old_state, action, new_state, reward):
        """
        Update Q-learning model, given an old state, an action taken
        in that state, a new resulting state, and the reward received
        from taking that action.
        """
        old = self.get_q_value(old_state, action)
        # best_future = self.best_future_reward(new_state)
        best_future = self.best_future_reward(new_state)
        self.update_q_value(old_state, action, old, reward, best_future)

    def get_q_value(self, state, action):
        """
        Return the Q-value for the state `state` and the action `action`.
        If no Q-value exists yet in `self.q`, return 0.
        """
        k = ('_'.join(state[1]), action)
        return self.q[k] if k in self.q else 0

    def update_q_value(self, state, action, old_q, reward, future_rewards):
        """
        Update the Q-value for the state `state` and the action `action`
        given the previous Q-value `old_q`, a current reward `reward`,
        and an estiamte of future rewards `future_rewards`.

        Use the formula:

        Q(s, a) <- old value estimate
                   + alpha * (new value estimate - old value estimate)

        where `old value estimate` is the previous Q-value,
        `alpha` is the learning rate, and `new value estimate`
        is the sum of the current reward and estimated future rewards.
        """
        # print(f"update_q_value={old_q + self.alpha * ((reward + future_rewards) - old_q)}")
        self.q[('_'.join(state), action)] = old_q + self.alpha * ((reward + future_rewards) - old_q)

    def best_future_reward(self, state):
        """
        Given a state `state`, consider all possible `(state, action)`
        pairs available in that state and return the maximum of all
        of their Q-values.

        Use 0 as the Q-value if a `(state, action)` pair has no
        Q-value in `self.q`. If there are no available actions in
        `state`, return 0.
        """
        return max((self.get_q_value(state, a) for a in P4.available_actions(state)), default=0)

    def choose_action(self, state, epsilon=True):
        """
        Given a state `state`, return an action `(i, j)` to take.

        If `epsilon` is `False`, then return the best action
        available in the state (the one with the highest Q-value,
        using 0 for pairs that have no Q-values).

        If `epsilon` is `True`, then with probability
        `self.epsilon` choose a random available action,
        otherwise choose the best action available.

        If multiple actions have the same Q-value, any of those
        options is an acceptable return value.
        """
        aas = P4.available_actions(state)
        epsilon = epsilon and self.epsilon > random.random()
        # if not epsilon: print(f"action={max(aas, key=lambda a: self.get_q_value(state, a))}")
        return list(aas)[random.randrange(0, len(aas))] if epsilon else max(aas, key=lambda a: self.get_q_value(state, a))


def train(n):
    """
    Train an AI by playing `n` games against itself.
    """

    player = P4AI()

    # Play n games
    for i in range(n):
        print(f"Playing training game {i + 1}")
        game = P4()

        # Keep track of last move made by either player
        last = {
            0: {"state": None, "action": None},
            1: {"state": None, "action": None}
        }

        # Game loop
        while True:

            # Keep track of current state and action
            current_player = game.player
            state = game.piles.copy()
            token = [TOKENS[current_player]]
            action = player.choose_action(token + state)
            # Make move
            game.move(action)
            new_state = game.piles.copy()

            # When game is over, update Q values with rewards
            if game.winner == -1:
                break

            if game.winner is not None:
                if game.winner == current_player:
                    player.update(
                        token + state,
                        action,
                        token + new_state,
                        1
                    )
                else:
                    player.update(
                        token + state,
                        action,
                        token + new_state,
                        -1
                    )
                break
            # If game is continuing, no rewards yet
            elif last[current_player]["state"] is not None:
                player.update(
                    token + state,
                    action,
                    token + new_state,
                    0
                )

    print("Done training")

    # Return the trained AI
    return player


def play(ai, human_player=None):
    """
    Play human game against the AI.
    `human_player` can be set to 0 or 1 to specify whether
    human player moves first or second.
    """

    # If no player order set, choose human's order randomly
    if human_player is None:
        human_player = random.randint(0, 1)
    print(f"human={human_player}")

    # Create new game
    game = P4()

    # Game loop
    while True:

        # Print contents of piles
        print()
        print("Piles:")
        for i, pile in enumerate(game.piles):
            print(f"Pile {i}: {pile}")
        print()

        # Compute available actions
        available_actions = P4.available_actions([''] + game.piles)
        time.sleep(1)

        # Let human make a move
        if game.player == human_player:
            print("Your Turn")
            while True:
                pile = int(input("Choose Pile: "))
                if pile in available_actions:
                    break
                print("Invalid move, try again.")

        # Have AI make a move
        else:
            print("AI's Turn")
            pile = ai.choose_action([TOKENS[P4.other_player(human_player)]] + game.piles, epsilon=False)
            print(f"AI chose pile {pile}.")

        # Make move
        game.move((pile))

        # Check for winner
        if game.winner is not None:
            print()
            print("GAME OVER")
            if game.winner == -1:
                print('Tie')
            else:
                winner = P4.other_player(game.player)
                winner = "Human" if winner == human_player else "AI"
                print(f"Winner is {winner}")
            return
