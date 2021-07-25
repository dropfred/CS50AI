import itertools
import random
import copy


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                        print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        return self.cells if len(self.cells) == self.count else set()

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        return self.cells if self.count == 0 else set()

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        if cell in self.cells:
            self.cells.remove(cell)
            self.count -= 1

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        if cell in self.cells: self.cells.remove(cell)


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge: sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge: sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """
        self.moves_made.add(cell)

        if cell not in self.safes: self.knowledge.append(Sentence([cell], 0))

        ncs = []
        for nc in itertools.product(range(cell[0] - 1, cell[0] + 2), range(cell[1] - 1, cell[1] + 2)):
            if nc[0] >= 0 and nc[0] < self.height and nc[1] >= 0 and nc[1] < self.width and nc != cell:
                if nc in self.mines: count -= 1
                elif nc not in self.safes: ncs.append(nc)
        if len(ncs) > 0: self.knowledge.append(Sentence(ncs, count))

        while True:
            done = True

            # for (s1, s2) in itertools.product(self.knowledge, self.knowledge):
            for (s1, s2) in itertools.permutations(self.knowledge, 2):
                if s1.cells < s2.cells:
                    s2.cells -= s1.cells
                    s2.count -= s1.count
                    done = False

            ss = set()
            ms = set()
            for s in self.knowledge:
                for c in s.known_safes(): ss.add(c)
                for c in s.known_mines(): ms.add(c)
            for c in ss :
                    self.mark_safe(c)
                    done = False
            for c in ms :
                    self.mark_mine(c)
                    done = False
            
            # remove empty sentences
            self.knowledge = [s for s in self.knowledge if len(s.cells) > 0]

            if done: break

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        for c in self.safes:
            if c not in self.moves_made: return c
        return None

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        # ai doesn't know how many mines are in total, would be helpful

        # pick unvisited cell
        vs = set()
        for s in self.knowledge: vs |= s.cells
        vs |= self.moves_made 
        vs |= self.mines
        c = next((c for c in itertools.product(range(self.height), range(self.width)) if c not in vs), None)
        if c is not None: return c

        # pick least mined cell
        def probe_mines(knowledge):
            def probe(knowledge):
                ps = {}
                if len(knowledge) == 0: return (ps, 1)
                for s in knowledge:
                    for c in s.cells: ps[c] = 0
                n = 0
                s = knowledge[-1]
                if len(s.cells) >= s.count:
                    scs = list(itertools.combinations(s.cells, s.count))
                    for cs in scs:
                        sk = copy.deepcopy(knowledge)
                        s = sk.pop(-1)
                        ms = set(cs)
                        ss = s.cells - ms
                        for (s, c) in itertools.product(sk, ms): s.mark_mine(c)
                        for (s, c) in itertools.product(sk, ss): s.mark_safe(c)
                        sk = [s for s in sk if s.count > 0]
                        (sms, sn) = probe(sk)
                        for c in ms: ps[c] += sn
                        for c in sms: ps[c] += sms[c]
                        n += sn
                if n == 0: ps.clear()
                return (ps, n)

            (ms, n) = probe(knowledge)
            for m in ms: ms[m] /= n
            return ms

        ms = probe_mines(self.knowledge)
        if len(ms) > 0: c = min(ms, key=lambda c: ms[c])

        return c
