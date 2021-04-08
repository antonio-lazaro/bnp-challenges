"""
Solutions script for BNP Interview Challenges.
"""

from math import fsum
from collections import deque
from sys import argv
import argparse
import textwrap

CHALLENGE_1 = 1
CHALLENGE_2 = 2
CHALLENGE_3 = 3

# MARK: Challenge 1

def is_balanced(input_str):
    """Checks that the input string is properly balanced.

    Args:
      input_str: The string to evaluate.

    Returns:
      Boolean value indicating if the input string is balanced (true) or
      is not balanced (false).
    """

    if input_str is None or len(input_str) == 0:
        return True

    open_list = ["[", "{", "("]
    close_list = ["]", "}", ")"]
    comment_opened = False
    unmatched_symbol_reg = None
    stack = []
    for i, c in enumerate(input_str):
        if c == '/':
            if unmatched_symbol_reg == '*':
                comment_opened = False
                unmatched_symbol_reg = None
            else:
                unmatched_symbol_reg = c
        elif c == '*':
            if unmatched_symbol_reg == '/':
                comment_opened = True
                unmatched_symbol_reg = None
            else:
                unmatched_symbol_reg = c
        elif c in open_list and not comment_opened:
            unmatched_symbol_reg = None
            stack.append(c)
        elif c in close_list and not comment_opened:
            unmatched_symbol_reg = None
            pos = close_list.index(c)
            if ((len(stack) > 0) and
                (open_list[pos] == stack[len(stack)-1])):
                stack.pop()
            else:
                return False
        else:
            unmatched_symbol_reg = None

    if len(stack) == 0 and not comment_opened:
        return True
    else:
        return False

# MARK: Challenge 2

def get_distribution(cost, weights):
    """Checks that an input string is properly balanced.

    Args:
      cost: The initial total cost.
      weights: List of weights

    Returns:
      Distribution of the cost proportionally to the weights.
    """
    weights_sum = fsum(weights)
    weighted_costs = [w / weights_sum * cost for w in weights]
    rounded_costs = [int(c) for c in weighted_costs]

    while sum(rounded_costs) < cost:
        max_improve = None
        for i, rounded_cost in enumerate(rounded_costs):
            if weights[i] > 0:
                # Get modified rounded cost
                m_rounded_cost = rounded_cost + 1

                # Obtain relative rounded error (rre) change
                rre = abs(weighted_costs[i] - rounded_cost) / weights[i]
                m_rre = abs(weighted_costs[i] - m_rounded_cost) / weights[i]
                diff = rre - m_rre
                if max_improve is None or diff > max_improve[1]:
                    max_improve = (i, diff)

        # Modify
        rounded_costs[max_improve[0]] += 1

    return rounded_costs

# MARK: Challenge 3

def get_optimal_sequence(jugs_capacities, target_volume):
    """Finds a sequence of operations that will yield a state
      where at least one jug contains a target volume of water.

    Args:
      target_volume: The target volume to obtain.
      jugs_capacities: List of jugs' capacities

    Returns:
      Sequence of operations  to reach a state where at least
      one jug has the target volume of water in it.
    """
    m = {}
    m_seq = {}
    is_solvable = False

    q = deque()

    n = len(jugs_capacities)
    initial_state = tuple([0 for _ in range(n)])
    if target_volume in initial_state:
        return []
  
    q.append(initial_state)

    while len(q) > 0:

        # Current state
        s = q.popleft()

        # If the state is already visited
        if s in m:
            continue

        # Initialize reached states array
        m[s] = []
        m_seq[s] = []
    
        # Obtain reached states from current state
        for src_jug in range(n):

            # Fill jug
            if s[src_jug] < jugs_capacities[src_jug]:
                reached_state = list(s)
                reached_state[src_jug] = jugs_capacities[src_jug]
                reached_state = tuple(reached_state)
                if target_volume in reached_state:
                    is_solvable = True
                m[s].append(reached_state)
                m_seq[s].append((-1, src_jug))

            # Empty jug
            if s[src_jug] > 0:
                reached_state = list(s)
                reached_state[src_jug] = 0
                reached_state = tuple(reached_state)
                if target_volume in reached_state:
                    is_solvable = True
                m[s].append(reached_state)
                m_seq[s].append((src_jug, -1))
        
            # Pour jug into another jug
            for dest_jug in range(n):
        
                if s[src_jug] != 0 and \
                    src_jug != dest_jug and \
                    s[dest_jug] < jugs_capacities[dest_jug]:

                    reached_state = list(s)
                    total = reached_state[dest_jug] + reached_state[src_jug]

                    if total > jugs_capacities[dest_jug]:
                        reached_state[dest_jug] = jugs_capacities[dest_jug]
                        reached_state[src_jug] = total - jugs_capacities[dest_jug]
                    else:
                        reached_state[dest_jug] = total
                        reached_state[src_jug] = 0

                    if target_volume in reached_state:
                        is_solvable = True

                    reached_state = tuple(reached_state)
                    m[s].append(reached_state)
                    m_seq[s].append((src_jug, dest_jug))
      
        # Add unvisited states to queue
        for reached_state in m[s]:
            if reached_state not in m:
                q.append(reached_state)
  
    # Unsolvable case
    if not is_solvable:
        return None
  
    # Find best path
    best_path = []
    best_seq = []
  
    def find_best_path(path, seq, best_path, best_seq):
        if target_volume in path[-1] and \
                (len(path) <= len(best_path) or len(best_path) == 0):
            best_path[:] = path[:]
            best_seq[:] = seq[:]
        else:
            for i, reached_state in enumerate(m[path[-1]]):
                if reached_state not in path and \
                        (len(path) <= len(best_path) or len(best_path) == 0):
                    seq.append(m_seq[path[-1]][i])
                    path.append(reached_state)
                    find_best_path(path, seq, best_path, best_seq)
                    seq.pop()
                    path.pop()
    
    find_best_path([initial_state], [], best_path, best_seq)
  
    return best_seq

def pour(src, dest, state, capacities):
    """Return next state (only for valid operations)"""
    next_state = list(state)
    if src == -1:
        next_state[dest] = capacities[dest]
    elif dest == -1:
        next_state[src] = 0
    else:
        total = next_state[dest] + next_state[src]
        if total > capacities[dest]:
            next_state[dest] = capacities[dest]
            next_state[src] = total - capacities[dest]
        else:
            next_state[dest] = total
            next_state[src] = 0
    return tuple(next_state)

def beautiful_print(actions, jugs_capacities):
    """Prints the sequence of operations and the results

    Args:
      actions: Sequence of actions
      jugs_capacities: List of jugs' capacities
    """
    n = len(jugs_capacities)
    state = tuple([0 for _ in range(n)])
    print(state)
    for action in actions:
        state = pour(action[0], action[1], state, jugs_capacities)
        print(action, '->', state)
    print('\n')
    print('Number of actions:', len(actions))
    print('Sequence:', actions)


# MARK: Main

if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser(description='Run challenge solution.')
    parser.add_argument('-n', type=int, dest="n",
                        help='an integer for the challenge number')
    args = parser.parse_args(argv[1:])

    if args.n == CHALLENGE_1:

        print('CHALLENGE 1')
        print('-----------')
        print('Description: Test if the string is balanced.')

        input_str = str(raw_input('Enter the string to test: '))

        print('Balanced' if is_balanced(input_str) else 'Unbalanced')

    elif args.n == CHALLENGE_2:

        print('CHALLENGE 2')
        print('-----------')
        print('Description: Cost prorating.')

        # Inputs
        cost = int(input('Enter the cost: '))
        print(textwrap.dedent("""
            Enter the weights
            (Write each weight press enter. To finish just press enter with an empty input):
            """))
        weights = []
        w = raw_input()
        while len(w) > 0:
            weights.append(float(w))
            w = raw_input()

        # Result
        result = get_distribution(cost, weights)
        print('prorated costs:', result)

    elif args.n == CHALLENGE_3:

        print('CHALLENGE 3')
        print('-----------')
        print('Description: Water Jugs.')

        # Inputs
        target = int(input('Enter the target: '))

        print(textwrap.dedent("""
            Enter the jugs capacities
            (Write each capacity and press enter. To finish just press enter with an empty input):
            """))
        capacities = []
        c = raw_input()
        while len(c) > 0:
            capacities.append(int(c))
            c = raw_input()

        # Result
        actions = get_optimal_sequence(capacities, target)

        # Print
        if actions is None:
            print('Not solvable')
        else:
            beautiful_print(actions, capacities)
