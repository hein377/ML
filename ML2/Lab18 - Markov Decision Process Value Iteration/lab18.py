import sys
import copy

GRID_WORLD_ACTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]                                 # simulates down, up, right, left
ACTION_TUP_TO_CHAR = {(1, 0): 'S', (-1, 0): 'N', (0, 1): 'E', (0, -1): 'W'}

def process_input_file(filename):
    grid = []
    with open(filename) as f:
        grid_size = int(f.readline().strip('\n'))
        gamma = float(f.readline().strip('\n'))
        noise = [float(i) for i in list(f.readline().strip('\n').split(','))]
        f.readline()
        for line in f: grid.append(list(line.strip('\n').split(',')))                   # start of actual grid description
        
    return grid, grid_size, gamma, noise

def find_possible_actions(state_s, grid_size):
    candidate_actions = GRID_WORLD_ACTIONS.copy()
    if(state_s[0] == 0): candidate_actions.remove((-1, 0))                              # state at top row: remove up direction
    elif(state_s[0] == grid_size-1): candidate_actions.remove((1, 0))                   # state at bottom row: remove down direction
    if(state_s[1] == 0): candidate_actions.remove((0, -1))                              # state at leftmost col: remove left direction
    elif(state_s[1] == grid_size-1): candidate_actions.remove((0, 1))                   # state at rightmost col: remove right direction
    return candidate_actions

def add_tuples(tup1, tup2):
    ls1, ls2 = list(tup1), list(tup2)
    return tuple([ls1[i]+ls2[i] for i in range(len(ls1))])

def subtract_tuples(tup1, tup2):
    ls1, ls2 = list(tup1), list(tup2)
    return tuple([ls1[i]-ls2[i] for i in range(len(ls1))])

def calc_transition_prob(cur_state, action, end_state, noise):                          # assumes noise[1] == noise[2]
    if(add_tuples(cur_state, action) == end_state): return noise[0]
    elif(subtract_tuples(cur_state, action) == end_state):
        if(len(noise) == 4): return noise[3]
        else: return 0
    return noise[1]

def calc_reward(grid, state_s):                                                          # assumes rewards are integers
    if((reward:=grid[state_s[0]][state_s[1]]) != 'X'): return int(reward)
    return 0

def calc_action_qvals(grid, grid_size, state_s, possible_actions, noise, gamma, value_funcs_arr):
    #for row in value_funcs_arr: print(row)
    #print()

    #print(f"STATE: {state_s}")
    #print(f"REWARD: {grid[state_s[0]][state_s[1]]}\n")

    if((reward:=grid[state_s[0]][state_s[1]]) != 'X'): return int(reward)                                    # if state_s == reward_state, V(state_s) = 0
    possible_state_s_prime = [add_tuples(state_s, pos_action) for pos_action in possible_actions]
    action_qvals = {}

    #print(f"Possible Next States: {possible_state_s_prime}")

    for action in possible_actions:
        #print(f"\nCandidate Action: {action}\n")
        q_val = 0
        for state_s_prime in possible_state_s_prime:
            #print(f"Possible Next State: {state_s_prime}")
            transition_prob = calc_transition_prob(state_s, action, state_s_prime, noise)
            immediate_reward = calc_reward(grid, state_s)

            #print(f"Transition Prob: {transition_prob}")
            #print(f"Immediate Reward: {immediate_reward}")
            #print(f"Action val: {transition_prob * (immediate_reward + gamma*value_funcs_arr[state_s_prime[0]][state_s_prime[1]])}")
            q_val += (transition_prob * (immediate_reward + gamma*value_funcs_arr[state_s_prime[0]][state_s_prime[1]]))
        
        action_qvals.update({action:q_val})
        
        #print(f"ACTION Q_VAL: {action_qvals}")
    #input()

    return sorted(action_qvals.items(), key=lambda x:x[1], reverse=True)

def val_iteration(grid, grid_size, gamma, noise, value_funcs_arr, new_value_funcs_arr, policy_pi_arr, k, k_max):
    if k == k_max or value_funcs_arr == new_value_funcs_arr: return k, value_funcs_arr, policy_pi_arr
    value_funcs_arr = copy.deepcopy(new_value_funcs_arr)

    for row in range(grid_size):
        for col in range(grid_size):
            sorted_keys_vals = calc_action_qvals(grid, grid_size, (row, col), find_possible_actions((row, col), grid_size), noise, gamma, new_value_funcs_arr)

            if(type(sorted_keys_vals) == list):                                                                                 # if not reward state
                policy_pi_arr[row][col] = ACTION_TUP_TO_CHAR[sorted_keys_vals[0][0]]                                   # optimal action policy
                new_value_funcs_arr[row][col] = round(sorted_keys_vals[0][1], 3)                                                 # optimal action's V value
            else: new_value_funcs_arr[row][col] = round(sorted_keys_vals, 3)

    return val_iteration(grid, grid_size, gamma, noise, value_funcs_arr, new_value_funcs_arr, policy_pi_arr, k+1, k_max)

grid, grid_size, gamma, noise = process_input_file(sys.argv[1])
num_iterations, final_values_func_arr, optimal_policy_arr = val_iteration(grid, grid_size, gamma, noise, [], [[0]*grid_size for _ in range(grid_size)], [['-']*grid_size for _ in range(grid_size)], 0, 100)

print(f"NUM INTERATIONS: {num_iterations}")

print("FINAL VALUES GRID")
for row in final_values_func_arr: print(row)

print("\nOPTIMAL POLICY GRID")
for row in optimal_policy_arr: print(row)