# imports
from scipy.sparse.csgraph import floyd_warshall
from scipy.sparse import csr_matrix
from copy import deepcopy
import time
import json

ids = ["209190172", "322794629"]

# global dictionaries
NUM_TO_PASSENGER, NUM_TO_COORD, COORD_TO_NUM   = {}, {}, {}

# general constant
INFINITY, MAX_STATES, MAX_TIME = 100000, 3000, 298
EMPTY = ''

# points constants
RESET_POINTS, DELIVERED_POINTS, REFUEL_POINTS = -50, 100, -10

# constants for the "check_valid" and "result" functions
ACTION_NAME, TAXI_NAME, NEW_LOCATION, PASSENGER_NAME = 0, 1, 2, 2

# constants for the 'get_max_expected_val_action_tuple' and 'act' functions
VALUE, ACTION = 0, 1

# constant for the 'find_unwanted_taxis_passengers' function
SCORE = 1


def json_to_state(json_object):
    """
    Gets a json object, turns it into a dictionary and returns it.
    """
    dictt = json.loads(json_object)
    return dictt


def state_to_json(dic):
    """
    Gets a dictionary, turns it into a json object and returns it.
    """
    json_object = json.dumps(dic)
    return json_object


def map_num_to_passengers_names(passengers_names):
    dictt = {}

    for i, passenger_name in enumerate(passengers_names):
        dictt[i] = passenger_name

    return dictt


def map_coordinates_to_num(n, m):
    dictt = {}
    cnt = 0

    for i in range(n):
        for j in range(m):
            dictt[(i, j)] = cnt
            cnt += 1

    return dictt


def map_num_to_coordinates(n, m):
    dictt = {}
    cnt = 0

    for i in range(n):
        for j in range(m):
            dictt[cnt] = (i, j)
            cnt += 1

    return dictt


def get_distance_matrix(mapp):
    n, m = len(mapp), len(mapp[0])
    size = n*m

    graph = [[0 for i in range(size)] for j in range(size)]

    for i in range(size):
        (x, y) = NUM_TO_COORD[i]
        if mapp[x][y] == "I":
            continue

        # up
        if x != 0 and mapp[x - 1][y] != "I":
            j = COORD_TO_NUM[(x-1, y)]
            graph[i][j] = 1
            graph[j][i] = 1
        # down
        if x != n - 1 and mapp[x + 1][y] != "I":
            j = COORD_TO_NUM[(x+1, y)]
            graph[i][j] = 1
            graph[j][i] = 1
        # left
        if y != 0 and mapp[x][y - 1] != "I":
            j = COORD_TO_NUM[(x, y-1)]
            graph[i][j] = 1
            graph[j][i] = 1
        # right
        if y != m - 1 and mapp[x][y + 1] != "I":
            j = COORD_TO_NUM[(x, y+1)]
            graph[i][j] = 1
            graph[j][i] = 1

    graph = csr_matrix(graph)
    distance_matrix = floyd_warshall(csgraph=graph, directed=False)

    return distance_matrix.tolist()


def update_distance_matrix_infinity_value(distance_matrix, N):
    for i in range(N):
        for j in range(N):
            if distance_matrix[i][j] == float('inf'):
                distance_matrix[i][j] = INFINITY

    return distance_matrix


def get_gas_locations(mapp):
    gas_locations = []

    for i in range(len(mapp)):
        for j in range(len(mapp[0])):

            if mapp[i][j] == "G":
                gas_locations.append(COORD_TO_NUM[(i, j)])

    return gas_locations


def min_dist_with_refuel(gas_locations, distance_matrix, t_location, dest, fuel):
    min_dist = INFINITY

    for g_location in gas_locations:

        # if the taxi can reach the gas station
        if fuel >= distance_matrix[t_location][g_location]:
            temp_dist = distance_matrix[t_location][g_location] + distance_matrix[g_location][dest]
            min_dist = min(min_dist, temp_dist)

    return min_dist

def get_real_dist(t_location, dest, fuel, gas_locations, distance_matrix):

    if fuel >= distance_matrix[t_location][dest]:
        return distance_matrix[t_location][dest]

    return min_dist_with_refuel(gas_locations, distance_matrix, t_location, dest, fuel)


def convert_action_to_legal_format(action):

    if isinstance(action, str):
        return action

    legal_format_action = []
    for sa in action:

        if sa[ACTION_NAME] == "move":
            new_tup = ("move", sa[TAXI_NAME], tuple(sa[NEW_LOCATION]))
            legal_format_action.append(new_tup)

        else:
            legal_format_action.append(sa)

    return tuple(legal_format_action)


def get_default_action(taxis_dict):
    action = []

    for taxi_name in taxis_dict.keys():
        action.append(('wait', taxi_name))

    return tuple(action)


def initialize_state(state):
    passengers_names, taxis_names = list(state["passengers"].keys()), list(state["taxis"].keys())

    state["reset_penalty"] = False

    for i, passenger_name in enumerate(passengers_names):
        state["passengers"][passenger_name]["delivered_reward"] = False

    for taxi_name in taxis_names:
        state["taxis"][taxi_name]["refuel_penalty"] = False


def remove_unwanted_passengers(state, passengers_to_remove):
    # don't include the passenger in our problem
    for passenger_name in passengers_to_remove:
        del state["passengers"][passenger_name]


def convert_state_to_our_format(state, last_action, passengers_to_remove=()):
    passengers_dict, taxis_dict = state["passengers"], state["taxis"]
    del state["turns to go"]

    initialize_state(state)
    remove_unwanted_passengers(state, passengers_to_remove)

    if last_action is None:
        return state

    if last_action == "reset":
        state["reset_penalty"] = True
        return state

    for sa in last_action:

        if sa[ACTION_NAME] == "drop off":
            passengers_dict[sa[PASSENGER_NAME]]["delivered_reward"] = True

        elif sa[ACTION_NAME] == "refuel":
            taxis_dict[sa[TAXI_NAME]]["refuel_penalty"] = True

    return state


def check_valid(new_sa, sa_list, taxis):
    """
    Gets a new sub-action 'new_sa' (for example, ('pick up', 'taxi 1', 'Yossi')), and a list of other
    sub-actions, and returns whether 'new_sa' is allowed given 'sa_list'.
    """

    for sa in sa_list:

        if new_sa[ACTION_NAME] != "move" and sa[ACTION_NAME] == "move":
            if tuple(taxis[new_sa[TAXI_NAME]]["location"]) == tuple(sa[NEW_LOCATION]):
                return False

        elif new_sa[ACTION_NAME] == "move" and sa[ACTION_NAME] != "move":
            if tuple(new_sa[NEW_LOCATION]) == tuple(taxis[sa[TAXI_NAME]]["location"]):
                return False

        elif new_sa[ACTION_NAME] == "move" and sa[ACTION_NAME] == "move":
            if new_sa[NEW_LOCATION] == sa[NEW_LOCATION]:
                return False

    return True


def find_action_list(i, n, sa_lists, sub_res, res, taxis):
    """
    Finds all possible actions in a given state and appends them to res.
    """

    # sa = sub-action (atomic action of one taxi, out of list of actions)
    if i == n:
        sub_res = tuple(sub_res.copy())
        res.append(sub_res)
        return

    sa_list = sa_lists[i]
    for j in range(len(sa_list)):

        if check_valid(sa_list[j], sub_res, taxis):
            sub_res.append(sa_list[j])
            find_action_list(i + 1, n, sa_lists, sub_res, res, taxis)
            sub_res.pop()


def actions(state, initial, taxis_to_ignore=(), is_optimal=True):
    """Returns all the actions that can be executed in the given
    state. The result should be a tuple (or other iterable) of actions
    as defined in the problem description file"""

    mapp, taxis, passengers = state["map"], state["taxis"], state["passengers"]
    is_passenger_in_dest = False

    # sa = sub action (an action is defined as a series of one sub-action for every taxi)
    sa_per_taxi = {}

    for taxi_name in taxis.keys():
        sa_per_taxi[taxi_name] = []

    for taxi_name, taxi_dict in taxis.items():
        (t_x, t_y) = taxi_dict["location"]

        # wait
        action_tup = ("wait", taxi_name)
        sa_per_taxi[taxi_name].append(action_tup)

        # skip all the other possible sub-actions for the ignored taxis
        # allowing them to do only 'wait'
        if taxi_name in taxis_to_ignore:
            continue

        # fuel
        fuel_capacity = initial["taxis"][taxi_name]["fuel"]
        if mapp[t_x][t_y] == "G" and taxi_dict["fuel"] < fuel_capacity:
            sa_per_taxi[taxi_name].append(("refuel", taxi_name))

        # pick up and drop off
        passengers_in_dest_num = 0
        for passenger_name, passenger_dict in passengers.items():

            # indicates that the passenger arrived to their destination
            # if so, no taxi will pick him up
            if passenger_dict["location"] == passenger_dict["destination"]:
                is_passenger_in_dest = True
                continue

            # pick up
            if taxi_dict["location"] == passenger_dict["location"] and taxi_dict["capacity"] > 0:
                sa_per_taxi[taxi_name].append(("pick up", taxi_name, passenger_name))

            # drop off
            # the taxi is at passenger's destination AND the taxi is the one that picked up the passenger
            if taxi_dict["location"] == passenger_dict["destination"] and taxi_name == passenger_dict["location"]:
                sa_per_taxi[taxi_name].append(("drop off", taxi_name, passenger_name))

        # move

        # if fuel is empty can't move
        if taxi_dict["fuel"] == 0:
            continue

        n, m = len(mapp), len(mapp[0])

        # up
        if t_x != 0 and mapp[t_x - 1][t_y] != "I":
            sa_per_taxi[taxi_name].append(("move", taxi_name, [t_x - 1, t_y]))
        # down
        if t_x != n - 1 and mapp[t_x + 1][t_y] != "I":
            sa_per_taxi[taxi_name].append(("move", taxi_name, [t_x + 1, t_y]))
        # left
        if t_y != 0 and mapp[t_x][t_y - 1] != "I":
            sa_per_taxi[taxi_name].append(("move", taxi_name, [t_x, t_y - 1]))
        # right
        if t_y != m - 1 and mapp[t_x][t_y + 1] != "I":
            sa_per_taxi[taxi_name].append(("move", taxi_name, [t_x, t_y + 1]))

    sa_per_taxi_lists = []

    for sa_per_taxi_list in sa_per_taxi.values():
        sa_per_taxi_lists.append(sa_per_taxi_list)

    actions_list = []
    find_action_list(0, len(sa_per_taxi_lists), sa_per_taxi_lists, [], actions_list, taxis)

    # adding the option to reset and return to the initial state in order to accumulate more points
    # in the optimal agent we are adding 'reset' as an optional action for every state
    # in the non-optimal agent we are adding 'reset' only if there's a passenger in his destination
    if is_optimal or is_passenger_in_dest:
        actions_list.append("reset")

    return tuple(actions_list)


def result(state, action, initial):
    """Return the state that results from executing the given
    action in the given state. The action must be one of
    self.actions(state)."""

    new_state = deepcopy(state)

    if action == "reset":
        reset_state = deepcopy(initial)
        reset_state["reset_penalty"] = True
        return reset_state

    taxis, passengers = new_state["taxis"], new_state["passengers"]

    # sa = sub action (an action is defined as a series of one sub-action for every taxi)
    for sa in action:
        sa_name, taxi_name = sa[ACTION_NAME], sa[TAXI_NAME]

        if sa_name == "wait":
            continue

        elif sa_name == "refuel":
            taxis[taxi_name]["fuel"] = initial["taxis"][taxi_name]["fuel"]
            taxis[taxi_name]["refuel_penalty"] = True

        elif sa_name == "pick up":
            # the passenger we pick up
            passenger_name = sa[PASSENGER_NAME]

            passengers[passenger_name]["location"] = taxi_name
            taxis[taxi_name]["capacity"] -= 1

        elif sa_name == "drop off":
            # the passenger we drop off
            passenger_name = sa[PASSENGER_NAME]

            passengers[passenger_name]["location"] = passengers[passenger_name]["destination"]
            passengers[passenger_name]["delivered_reward"] = True
            taxis[taxi_name]["capacity"] += 1

        elif sa_name == "move":
            taxis[taxi_name]["location"] = sa[NEW_LOCATION]
            taxis[taxi_name]["fuel"] -= 1

    return new_state


def get_new_dest_lists(i, n, passengers_dict, total_p, sub_res, res):

    if i == n:
        sub_res = tuple(sub_res.copy())
        res.append((sub_res, total_p))
        return

    possible_goals = passengers_dict[NUM_TO_PASSENGER[i]]["possible_goals"]
    p = passengers_dict[NUM_TO_PASSENGER[i]]["prob_change_goal"]
    current_dest = passengers_dict[NUM_TO_PASSENGER[i]]["destination"]

    # probability to stay in the same destination
    if current_dest not in possible_goals:
        same_dest_p = 1 - p

    else:
        same_dest_p = (1 - p) + (p / len(possible_goals))

    sub_res.append(current_dest)
    get_new_dest_lists(i + 1, n, passengers_dict, total_p * same_dest_p, sub_res, res)
    sub_res.pop()

    # probability to be in a new destination
    for possible_goal in possible_goals:

        # already dealt with
        if possible_goal == current_dest:
            continue

        new_dest_p = p / len(possible_goals)

        sub_res.append(possible_goal)
        get_new_dest_lists(i + 1, n, passengers_dict, total_p * new_dest_p, sub_res, res)
        sub_res.pop()


def get_stochastic_states(state, action, initial, total_passengers_num):
    main_state = result(state, action, initial)

    # if the action is reset, then there aren't any stochastic state (deterministic)
    if action == "reset":
        return [main_state], [1]

    passengers_dict = main_state["passengers"]
    stochastic_states, p_stochastic_states = [], []

    new_dest_lists = []
    get_new_dest_lists(0, total_passengers_num, passengers_dict, 1, [], new_dest_lists)

    for new_dest_list, p in new_dest_lists:
        p_stochastic_states.append(p)
        stochastic_state = deepcopy(main_state)
        passengers_dict = stochastic_state["passengers"]

        for i, new_dest in enumerate(new_dest_list):
            passengers_dict[NUM_TO_PASSENGER[i]]["destination"] = new_dest

        stochastic_states.append(stochastic_state)

    return stochastic_states, p_stochastic_states


def R(state):

    if state["reset_penalty"]:
        state["reset_penalty"] = False
        return RESET_POINTS

    reward = 0
    passengers_dict, taxis_dict = state["passengers"], state["taxis"]

    for passenger_dict in passengers_dict.values():

        if passenger_dict["delivered_reward"]:
            passenger_dict["delivered_reward"] = False
            reward += DELIVERED_POINTS

    for taxi_dict in taxis_dict.values():

        if taxi_dict["refuel_penalty"]:
            taxi_dict["refuel_penalty"] = False
            reward += REFUEL_POINTS

    return reward

# -----------------------------------------------------

class OptimalTaxiAgent:

    def __init__(self, initial):
        initial_copy = json_to_state(state_to_json(initial))
        self.initial = initial_copy

        self.T = self.initial["turns to go"]
        del self.initial["turns to go"]

        initialize_state(self.initial)

        global NUM_TO_PASSENGER
        NUM_TO_PASSENGER = map_num_to_passengers_names(self.initial["passengers"])

        self.total_passengers_num = len(self.initial["passengers"].keys())
        self.default_action = get_default_action(self.initial["taxis"])
        self.last_action = None

        # initialize a dictionary defined as follows:
        # key - (state (json format), action)
        # value - all the stochastic states that are reachable from 'state' by preforming 'action' and their probabilities
        self.s = {}

        # initialize a dictionary defined as follows:
        # key - state (json format)
        # value - all the actions there are available from 'state'
        self.a = {}

        # initialize a dictionary for each round, the dictionary at round t defined as follows:
        # key - state (json format) with t rounds to go
        # value - optimal expected value at round and the optimal action
        self.v = [{} for i in range(self.T + 1)]

        # find all possible states that are reachable from "initial"
        self.get_all_possible_states()

        # get optimal policy for all reachable state from "initial"
        self.get_optimal_policy()

    def get_all_possible_states(self):

        self.v[self.T][state_to_json(self.initial)] = (0, self.default_action)
        convergence_flag = False

        for t in range(self.T, 0, -1):
            # we reached all the possible states by starting from 'initial' state therefore we just need to duplicate the dictionary
            # turns to go is represented by the index (t) of the dictionary in the dictionary list 'v'
            if convergence_flag:
                self.v[t-1] = deepcopy(self.v[t])
                continue

            for state_json in self.v[t].keys():
                state = json_to_state(state_json)

                # reset rewards/penalties indicators
                R(state)

                # all the actions that we can do from "state"
                if state_json in self.a:
                    actions_lst = self.a[state_json]
                else:
                    actions_lst = actions(state, self.initial)
                    self.a[state_json] = actions_lst

                # find all the stochastic states that are reachable from "state" by doing "action"
                for action in actions_lst:
                    action_tup = convert_action_to_legal_format(action)
                    if (state_json, action_tup) in self.s:
                        stochastic_states, _ = self.s[(state_json, action_tup)]

                    else:
                        stochastic_states, p_stochastic_states = get_stochastic_states(state, action, self.initial, self.total_passengers_num)
                        self.s[(state_json, action_tup)] = (stochastic_states, p_stochastic_states)

                    # initialize all the reachable stochastic stats in the dictionary with t-1 turns
                    for s_prime in stochastic_states:
                        self.v[t-1][state_to_json(s_prime)] = (0, self.default_action)

            # check if we reached all the possible states by starting from 'initial' state
            if len(self.v[t-1].keys()) == len(self.v[t].keys()):
                convergence_flag = True

    def find_v0(self):

        for state_json in self.v[0].keys():
            state = json_to_state(state_json)
            state_reward = R(deepcopy(state))
            self.v[0][state_to_json(state)] = (state_reward, EMPTY)

    def get_max_expected_val_action_tuple(self, state, state_json, actions_lst, t):
        state_reward, expected_val_action_lst = R(state), []

        for action in actions_lst:
            action_tup = convert_action_to_legal_format(action)
            stochastic_states, p_stochastic_states = self.s[(state_json, action_tup)]
            expected_val = state_reward

            for s_prime, p in zip(stochastic_states, p_stochastic_states):
                expected_val += p * self.v[t - 1][state_to_json(s_prime)][VALUE]

            expected_val_action_lst.append((expected_val, action))

        return max(expected_val_action_lst, key=lambda x: x[VALUE])

    def get_optimal_policy(self):
        self.find_v0()

        for t in range(1, self.T + 1):
            for state_json in self.v[t].keys():
                state = json_to_state(state_json)
                actions_lst = self.a[state_json]
                max_expected_val_action = self.get_max_expected_val_action_tuple(deepcopy(state), state_json, actions_lst, t)
                self.v[t][state_to_json(state)] = max_expected_val_action

    def act(self, state):
        t = state["turns to go"]
        updated_state = convert_state_to_our_format(deepcopy(state), self.last_action)

        optimal_action = self.v[t][state_to_json(updated_state)][ACTION]
        optimal_action = convert_action_to_legal_format(optimal_action)

        self.last_action = optimal_action
        return optimal_action

# --------------------------------------------------


class TaxiAgent:

    def update_map_for_taxi(self, mapp, wanted_taxi_name):

        for taxi_name, taxi_dict in self.initial["taxis"].items():
            if taxi_name != wanted_taxi_name:

                x, y = taxi_dict["location"]
                mapp[x][y] = "I"

        return mapp

    def get_score(self, taxi_to_score_dict, passenger_to_score_dict, distance_matrix, gas_locations):
        # now, we'll find the score
        # first, find distance from taxi to passenger
        fuel = taxi_to_score_dict["fuel"]
        t_location = taxi_to_score_dict["location"]
        t_location = COORD_TO_NUM[tuple(t_location)]

        p_location, p_destination = passenger_to_score_dict["location"], passenger_to_score_dict["destination"]
        p_location, p_destination = COORD_TO_NUM[tuple(p_location)], COORD_TO_NUM[tuple(p_destination)]

        t_p_dist = get_real_dist(t_location, p_location, fuel, gas_locations, distance_matrix)

        # second, find the distance from the passenger's location to destination
        possible_goals = passenger_to_score_dict["possible_goals"]
        p_change = passenger_to_score_dict["prob_change_goal"]

        expected_p_d_dist = (1 - p_change) * distance_matrix[p_location][p_destination]

        p_change = p_change / len(possible_goals)
        for possible_goal in possible_goals:
            possible_goal = COORD_TO_NUM[tuple(possible_goal)]
            expected_p_d_dist += p_change * distance_matrix[p_location][possible_goal]

        return t_p_dist + expected_p_d_dist

    def find_unwanted_taxis_passengers(self):

        mapp, taxis_dict, passengers_dict = self.initial["map"], self.initial["taxis"], self.initial["passengers"]
        n, m = len(mapp), len(mapp[0])

        if self.finished or (len(taxis_dict.keys()) == 1 and len(passengers_dict.keys()) == 1):
            return

        global NUM_TO_COORD, COORD_TO_NUM
        NUM_TO_COORD = map_num_to_coordinates(n, m)
        COORD_TO_NUM = map_coordinates_to_num(n, m)

        scores_lst = []
        for taxi_name, taxi_dict in self.initial["taxis"].items():
            # for each taxi, ignore all other taxis (by removing them and setting their location to "impassible")
            updated_map = self.update_map_for_taxi(deepcopy(mapp), taxi_name)
            distance_matrix = update_distance_matrix_infinity_value(get_distance_matrix(updated_map), N=n*m)
            gas_locations = get_gas_locations(updated_map)

            for passenger_name, passenger_dict in self.initial["passengers"].items():

                score = self.get_score(taxi_dict, passenger_dict, distance_matrix, gas_locations)
                scores_lst.append(((taxi_name, passenger_name), score))

        best_taxi, best_passenger = min(scores_lst, key=lambda x: x[SCORE])[0]

        # find all taxis we want to ignore in the problem
        for taxi_name in taxis_dict.keys():

            if taxi_name != best_taxi:
                self.taxis_to_ignore.append(taxi_name)

        # find all the passengers we want to remove from the problem
        for passenger_name in passengers_dict.keys():

            if passenger_name != best_passenger:
                self.passengers_to_remove.append(passenger_name)

    def initialize_agent_parameters(self):
        global NUM_TO_PASSENGER
        NUM_TO_PASSENGER = map_num_to_passengers_names(list(self.initial["passengers"].keys()))

        self.total_passengers_num = len(self.initial["passengers"].keys())
        self.default_action = get_default_action(self.initial["taxis"])

        # initialize a dictionary defined as follows:
        # key - (state (json format), action)
        # value - all the stochastic states that are reachable from 'state' by preforming 'action' and their probabilities
        self.s = {}

        # initialize a dictionary defined as follows:
        # key - state (json format)
        # value - all the actions there are available from 'state'
        self.a = {}

        # initialize a dictionary for each round, the dictionary at round t defined as follows:
        # key - state (json format) with t rounds to go
        # value - optimal expected value at round and the optimal action
        self.v = [{} for i in range(self.T + 1)]

    def __init__(self, initial):
        # start timer since the '__init__' function started
        self.start_time = time.time()

        initial_copy = json_to_state(state_to_json(initial))
        self.initial = initial_copy

        self.T = self.initial["turns to go"]
        del self.initial["turns to go"]

        self.finished = False
        self.last_action = None
        self.taxis_to_ignore = []
        self.passengers_to_remove = []

        self.initialize_agent_parameters()
        initialize_state(self.initial)

        # trying to find all possible states that are reachable from "initial" for the original problem
        self.finished = self.get_all_possible_states(is_test=True)

        if not self.finished:
            # reducing the problem
            self.find_unwanted_taxis_passengers()
            remove_unwanted_passengers(self.initial, self.passengers_to_remove)
            self.initialize_agent_parameters()

            # find all possible states that are reachable from "initial" for the reduced problem
            self.get_all_possible_states()

        # get policy for all reachable state from "initial"
        self.get_policy()

    def get_all_possible_states(self, is_test=False):

        self.v[self.T][state_to_json(self.initial)] = (0, self.default_action)
        convergence_flag = False

        for t in range(self.T, 0, -1):
            state_num = 0

            # we reached all the possible states by starting from 'initial' state therefore we just need to duplicate the dictionary
            # turns to go is represented by the index (t) of the dictionary in the dictionary list 'v'
            if convergence_flag:
                self.v[t-1] = deepcopy(self.v[t])
                continue

            for state_json in self.v[t].keys():
                state = json_to_state(state_json)

                # reset rewards/penalties indicators
                R(state)

                # all the actions that we can do from "state"
                if state_json in self.a:
                    actions_lst = self.a[state_json]

                else:
                    actions_lst = actions(state, self.initial, self.taxis_to_ignore, is_optimal=False)
                    self.a[state_json] = actions_lst

                for action in actions_lst:
                    action_tup = convert_action_to_legal_format(action)
                    if (state_json, action_tup) in self.s:
                        stochastic_states, _ = self.s[(state_json, action_tup)]

                    else:
                        stochastic_states, p_stochastic_states = get_stochastic_states(state, action, self.initial, self.total_passengers_num)
                        self.s[(state_json, action_tup)] = (stochastic_states, p_stochastic_states)

                    # initialize all the reachable stochastic stats in the dictionary with t-1 turns
                    for s_prime in stochastic_states:
                        s_prime_json = state_to_json(s_prime)

                        if is_test and s_prime_json not in self.v[t-1]:
                            state_num += 1

                            if state_num > MAX_STATES:
                                return False

                        self.v[t-1][s_prime_json] = (0, self.default_action)

            # check if we reached all the possible states by starting from 'initial' state
            if len(self.v[t-1].keys()) == len(self.v[t].keys()):
                convergence_flag = True

        return True

    def find_v0(self):

        for state_json in self.v[0].keys():
            state = json_to_state(state_json)
            state_reward = R(deepcopy(state))
            self.v[0][state_to_json(state)] = (state_reward, EMPTY)

    def get_max_expected_val_action_tuple(self, state, state_json, actions_lst, t):
        state_reward, expected_val_action_lst = R(state), []

        for action in actions_lst:
            action_tup = convert_action_to_legal_format(action)
            stochastic_states, p_stochastic_states = self.s[(state_json, action_tup)]
            expected_val = state_reward

            for s_prime, p in zip(stochastic_states, p_stochastic_states):
                expected_val += p * self.v[t - 1][state_to_json(s_prime)][VALUE]

            expected_val_action_lst.append((expected_val, action))

        return max(expected_val_action_lst, key=lambda x: x[VALUE])

    def get_policy(self):
        self.find_v0()

        for t in range(1, self.T + 1):

            for state_json in self.v[t].keys():
                state = json_to_state(state_json)
                actions_lst = self.a[state_json]

                max_expected_val_action = self.get_max_expected_val_action_tuple(deepcopy(state), state_json, actions_lst, t)
                self.v[t][state_to_json(state)] = max_expected_val_action

                if time.time() - self.start_time > MAX_TIME:
                    return

    def act(self, state):
        t = state["turns to go"]
        updated_state = convert_state_to_our_format(deepcopy(state), self.last_action, self.passengers_to_remove)
        optimal_action = self.v[t][state_to_json(updated_state)][ACTION]
        optimal_action = convert_action_to_legal_format(optimal_action)

        self.last_action = optimal_action
        return optimal_action
