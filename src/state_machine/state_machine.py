import numpy as np

from typing import Optional, List, Dict, Tuple
from flatland.core.env import Environment
from flatland.envs.rail_env import RailEnvActions
from flatland.envs.agent_utils import RailAgentStatus
from flatland.core.grid.grid4_utils import get_new_position, get_direction, Grid4TransitionsEnum
from flatland.core.transition_map import GridTransitionMap


# TODO Add check for status
# Only for active agents
def act(args, env, a, state, prediction):
	
	agent = env.agents[a]
	if agent.status == RailAgentStatus.READY_TO_DEPART:
		agent_virtual_position = agent.initial_position
	elif agent.status == RailAgentStatus.ACTIVE:
		agent_virtual_position = agent.position
	elif agent.status == RailAgentStatus.DONE:
		agent_virtual_position = agent.target
	else:
		return 1, 0
	
	# Case 1 - Path is free, go
	if state[0] == 0:
		# Go	
		return 0, 0
	# Case 2 - in front of an overlapping span
	elif state[0] == 1 and state[args.prediction_depth] == 0:
		# Go if has priority, should consider prio in first span? prio like this doesn't work because of wrong conflicting agents TODO 
		if state[args.prediction_depth*4] < state[args.prediction_depth*4 + 1]:
			return 0, 0
		else:
			possible_transitions = env.rail.get_transitions(*agent_virtual_position,
			                                                agent.direction)  # Build list of possible branching directions from cell
			if np.count_nonzero(possible_transitions) > 1:
				actions = find_alternative(env, possible_transitions, agent_virtual_position, agent.direction, prediction)
				# Pick one of those
				return np.random.choice(actions), 1
			else:
				return 1, 0 # Stop, no immediate alternative
			
	# Case 3 
	elif state[0] == 1 and state[args.prediction_depth] == 1:
		
		possible_transitions = env.rail.get_transitions(*agent_virtual_position,
		                                                agent.direction)  # Build list of possible branching directions from cell
		if np.count_nonzero(possible_transitions) > 1:
			actions = find_alternative(env, possible_transitions, agent_virtual_position, agent.direction, prediction)
			# Pick one of those
			return np.random.choice(actions), 1
		else:
			
			# Stop
			return 1, 0
	
def find_alternative(env, possible_transitions, agent_pos, agent_dir, prediction):
	# Approccio naive - se non mi trovo su un fork mi blocco
	# altrimenti si potrebbe far ricalcolare uno shortestpath che non consideri il binario su cui si trova il treno che confligge

		possible_directions = []
		neighbours = []
		for j, branch_direction in enumerate([(agent_dir + j) % 4 for j in range(-1, 3)]):
			if possible_transitions[branch_direction]:
				possible_directions.append(branch_direction)
		for direction in possible_directions:
			neighbour_cell = get_new_position(agent_pos, direction)
			new_direction = get_direction(pos1=agent_pos, pos2=neighbour_cell)
			neighbours.append((neighbour_cell, new_direction))
		# Compute all possible moves except the ones of the shortest path
		next_cell = (prediction[0][0], prediction[0][1])
		neighbours = [n for n in neighbours if next_cell not in n]
		
		actions = [get_action_for_move(
			agent_pos,
			agent_dir,
			n[0],
			n[1],
			env.rail
		) for n in neighbours]
		
		return actions

def get_action_for_move(
		agent_position: Tuple[int, int],
		agent_direction: Grid4TransitionsEnum,
		next_agent_position: Tuple[int, int],
		next_agent_direction: int,
		rail: GridTransitionMap) -> Optional[RailEnvActions]:
	"""
	Get the action (if any) to move from a position and direction to another.
	TODO https://gitlab.aicrowd.com/flatland/flatland/issues/299 The implementation could probably be more efficient
	and more elegant. But given the few calls this has no priority now.
	Parameters
	----------
	agent_position
	agent_direction
	next_agent_position
	next_agent_direction
	rail
	Returns
	-------
	Optional[RailEnvActions]
		the action (if direct transition possible) or None.
	"""
	possible_transitions = rail.get_transitions(*agent_position, agent_direction)
	num_transitions = np.count_nonzero(possible_transitions)
	# Start from the current orientation, and see which transitions are available;
	# organize them as [left, forward, right], relative to the current orientation
	# If only one transition is possible, the forward branch is aligned with it.
	if rail.is_dead_end(agent_position):
		valid_action = RailEnvActions.MOVE_FORWARD
		new_direction = (agent_direction + 2) % 4
		if possible_transitions[new_direction]:
			new_position = get_new_position(agent_position, new_direction)
			if new_position == next_agent_position and new_direction == next_agent_direction:
				return valid_action
	elif num_transitions == 1:
		valid_action = RailEnvActions.MOVE_FORWARD
		for new_direction in [(agent_direction + i) % 4 for i in range(-1, 2)]:
			if possible_transitions[new_direction]:
				new_position = get_new_position(agent_position, new_direction)
				if new_position == next_agent_position and new_direction == next_agent_direction:
					return valid_action
	else:
		for new_direction in [(agent_direction + i) % 4 for i in range(-1, 2)]:
			if possible_transitions[new_direction]:
				if new_direction == agent_direction:
					valid_action = RailEnvActions.MOVE_FORWARD
					new_position = get_new_position(agent_position, new_direction)
					if new_position == next_agent_position and new_direction == next_agent_direction:
						return valid_action
				elif new_direction == (agent_direction + 1) % 4:
					valid_action = RailEnvActions.MOVE_RIGHT
					new_position = get_new_position(agent_position, new_direction)
					if new_position == next_agent_position and new_direction == next_agent_direction:
						return valid_action
				elif new_direction == (agent_direction - 1) % 4:
					valid_action = RailEnvActions.MOVE_LEFT
					new_position = get_new_position(agent_position, new_direction)
					if new_position == next_agent_position and new_direction == next_agent_direction:
						return valid_action