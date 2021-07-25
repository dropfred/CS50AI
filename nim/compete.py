import random
import nim
# import nim2 as nim2
import nim_c1 as nim2
# import nim_c2 as nim2
# import nim_c3 as nim2

def compete(ai1, ai2):
    # If no player order set, choose human's order randomly
    p = random.randint(0, 1)

    # Create new game
    game = nim.Nim()

    # Game loop
    while True:

        # Print contents of piles
        print()
        print("Piles:")
        for i, pile in enumerate(game.piles):
            print(f"Pile {i}: {pile}")
        print()

        # Compute available actions
        available_actions = nim.Nim.available_actions(game.piles)
        # time.sleep(1)

        # Let human make a move
        if game.player == p:
            print("AI1's Turn")
            pile, count = ai1.choose_action(game.piles, epsilon=False)
            print(f"AI1 chose to take {count} from pile {pile}.")

        # Have AI make a move
        else:
            print("AI2's Turn")
            pile, count = ai2.choose_action(game.piles, epsilon=False)
            print(f"AI2 chose to take {count} from pile {pile}.")

        # Make move
        game.move((pile, count))

        # Check for winner
        if game.winner is not None:
            print()
            print("GAME OVER: " + str(game.winner))
            winner = "AI1" if game.winner == 0 else "AI2"
            print(f"Winner is {winner}")
            return game.winner


scores = [0, 0]
ai1 = nim.train(10000)
ai2 = nim2.train(10000)
for _ in range(10):
    print('----------')
    s = compete(ai1, ai2)
    scores[s] += 1
print("scores : " + str(scores))

# scores = [0, 0]
# for _ in range(20):
#     ai1 = nim.train(10000)
#     ai2 = nim2.train(10000)
#     s = compete(ai1, ai2)
#     scores[s] += 1
# print("scores : " + str(scores))