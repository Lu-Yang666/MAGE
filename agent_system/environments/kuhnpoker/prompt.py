from typing import List, Dict, Optional
KUHN_POKER_PLAY_PROMPT = """
You are an expert Kuhn Poker agent.

# Game Rules
- Kuhn Poker is a two-player card game. The deck includes only three
cards: King (K) > Queen (Q) > Jack (J).
- At the start of each game, both player_0 and player_1 place 1 chip
into the pot as a blind ante.
- Each player is dealt a private card, and the third card is set aside
unseen.
- The two players take turns acting, starting with player 0. A player
can choose to:
    - PASS: place no additional chips into the pot.
    - BET: place 1 additional chip into the pot.
- If a player chooses to <PASS> after the other player’s <BET>, the
betting player wins the pot.If both players choose to <PASS> or both players choose to <BET>, the
player with the higher card wins the pot.

# Player Information
- In each of your turns, the game state shows your private card and the betting history. You need to choose an action based on your card and the current
game state.
- You are playing as Player {agent_player_id}, opponent is Player {opponent_player_id}.

# Your Goal
Your goal is to maximize your total chips over the long run by making strategic decisions based on your private card and the betting history.

# Observation
The initial state of the game is:
{init_observation}{past_trajectories_reflections}{current_trajectory}

Now it's your turn to make a move.
- Your should first reason step-by-step about the current situation.
- Then choose ONE action: PASS or BET. Respond with the action name inside <action> </action> tag like <action>PASS</action> or <action>BET</action>.
"""

KUHN_POKER_REFLECT_PROMPT = """
You are an expert Kuhn Poker agent.
You are playing as Player {agent_player_id}, opponent is Player {opponent_player_id}.

# Game Rules
- Kuhn Poker is a two-player card game. The deck includes only three
cards: King (K) > Queen (Q) > Jack (J).
- At the start of each game, both player_0 and player_1 place 1 chip
into the pot as a blind ante.
- Each player is dealt a private card, and the third card is set aside
unseen.
- The two players take turns acting, starting with player 0. A player
can choose to:
    - PASS: place no additional chips into the pot.
    - BET: place 1 additional chip into the pot.
- If a player chooses to <PASS> after the other player’s <BET>, the
betting player wins the pot.If both players choose to <PASS> or both players choose to <BET>, the
player with the higher card wins the pot.

# Player Information
- In each of your turns, the game state shows your private card and the betting history. You need to choose an action based on your card and the current
game state.
- You are playing as Player {agent_player_id}, opponent is Player {opponent_player_id}.

# Your Goal
Your goal is to maximize your total chips over the long run by making strategic decisions based on your private card and the betting history.

# Task
You are given a past trial of your play. Reflect on your actions:
- Identify mistakes or suboptimal moves
- Devise a concise improved plan for your next attempt

# Past Experience
The initial state of the game is:
{init_observation}{current_trajectory}
The task is NOT successfully completed.

Now reflect and propose a new strategy.
- Step-by-step reasoning about mistakes and improvements.
- End your response with your reflection inside <remark> </remark> tags.
"""

KUHN_POKER_REFLECT_PROMPT_WON = """
You are an expert Kuhn Poker agent.
You are playing as Player {agent_player_id}, opponent is Player {opponent_player_id}.

# Game Rules
- Kuhn Poker is a two-player card game. The deck includes only three
cards: King (K) > Queen (Q) > Jack (J).
- At the start of each game, both player_0 and player_1 place 1 chip
into the pot as a blind ante.
- Each player is dealt a private card, and the third card is set aside
unseen.
- The two players take turns acting, starting with player 0. A player
can choose to:
    - PASS: place no additional chips into the pot.
    - BET: place 1 additional chip into the pot.
- If a player chooses to <PASS> after the other player’s <BET>, the
betting player wins the pot.If both players choose to <PASS> or both players choose to <BET>, the
player with the higher card wins the pot.

# Player Information
- In each of your turns, the game state shows your private card and the betting history. You need to choose an action based on your card and the current
game state.
- You are playing as Player {agent_player_id}, opponent is Player {opponent_player_id}.

# Your Goal
Your goal is to maximize your total chips over the long run by making strategic decisions based on your private card and the betting history.

# Task
You are given a past trial of your play. Reflect on your actions:
- Identify mistakes or suboptimal moves
- Devise a concise improved plan for your next attempt

# Past Experience
The initial state of the game is:
{init_observation}{current_trajectory}
The task is successfully completed.

Now reflect on why the strategy worked and how to maintain or improve it in future attempts.
- Step-by-step reasoning about key decisions that led to success.
- Identify any risks, inefficiencies, or opponent responses that could have caused failure.
- Propose refinements or safeguards to make the strategy more robust.
- End your response with your reflection inside <remark> </remark> tags.
"""

# ===============================
# Templates for past trajectories & reflections (lost/won variants)
# ===============================
PAST_TRAJECTORY_AND_REFLECTION_TEMPLATE_LOST = """
On trial #{traj_idx}, you have taken the following actions:
{past_trajectory}
The task is NOT successfully completed. Your reflection is:
{reflection}
"""

PAST_TRAJECTORY_AND_REFLECTION_TEMPLATE_WON = """
On trial #{traj_idx}, you have taken the following actions:
{past_trajectory}
The task is successfully completed. Your reflection is:
{reflection}
"""

HISTORY_ONLY_TEMPLATE_LOST = """
On trial #{traj_idx}, you have taken the following actions:
{past_trajectory}
the task is NOT successfully completed.
"""

HISTORY_ONLY_TEMPLATE_WON = """
On trial #{traj_idx}, you have taken the following actions:
{past_trajectory}
the task is successfully completed.
"""

REFLECTION_ONLY_TEMPLATE_LOST = """
On trial #{traj_idx}, the task is NOT successfully completed. Your reflection is:
{reflection}
"""

REFLECTION_ONLY_TEMPLATE_WON = """
On trial #{traj_idx}, the task is successfully completed. Your reflection is:
{reflection}
"""

def parse_reflection(traj_idx, past_traj, reflection, reflection_type, past_outcomes=None):
    """
    Build the concatenated past trajectories + reflections text.
    - traj_idx: number of past trials to include (int)
    - past_traj: list of past trajectory strings (len >= traj_idx)
    - reflection: list of reflection strings (len >= traj_idx)
    - reflection_type: one of 'history_and_reflection', 'history_only', 'reflection_only'
    - past_outcomes: optional list of booleans indicating success for each past trial (True=won)
    """
    if traj_idx == 0:
        return '\n'
    past_outcomes = past_outcomes or []
    memories = []
    for _idx in range(traj_idx):
        # determine outcome for this past trial
        outcome = False
        if _idx < len(past_outcomes):
            outcome = bool(past_outcomes[_idx])
        # choose templates based on outcome
        if outcome:
            # won
            if reflection_type == 'history_and_reflection':
                memory = PAST_TRAJECTORY_AND_REFLECTION_TEMPLATE_WON.format(
                    traj_idx=_idx + 1,
                    past_trajectory=past_traj[_idx],
                    reflection=reflection[_idx]
                )
            elif reflection_type == 'history_only':
                memory = HISTORY_ONLY_TEMPLATE_WON.format(
                    traj_idx=_idx + 1,
                    past_trajectory=past_traj[_idx],
                )
            elif reflection_type == 'reflection_only':
                memory = REFLECTION_ONLY_TEMPLATE_WON.format(
                    traj_idx=_idx + 1,
                    reflection=reflection[_idx]
                )
            else:
                raise ValueError(f"Unknown reflection_type: {reflection_type}")
        else:
            # lost / not successful
            if reflection_type == 'history_and_reflection':
                memory = PAST_TRAJECTORY_AND_REFLECTION_TEMPLATE_LOST.format(
                    traj_idx=_idx + 1,
                    past_trajectory=past_traj[_idx],
                    reflection=reflection[_idx]
                )
            elif reflection_type == 'history_only':
                memory = HISTORY_ONLY_TEMPLATE_LOST.format(
                    traj_idx=_idx + 1,
                    past_trajectory=past_traj[_idx],
                )
            elif reflection_type == 'reflection_only':
                memory = REFLECTION_ONLY_TEMPLATE_LOST.format(
                    traj_idx=_idx + 1,
                    reflection=reflection[_idx]
                )
            else:
                raise ValueError(f"Unknown reflection_type: {reflection_type}")
        memories.append(memory)
    return ''.join(memories)

# ===============================
# Templates for current trajectory
# ===============================
CURR_TRAJ_AT_TRAJ1 = """
You have already taken the following moves:
{current_trajectory}
"""

CURR_TRAJ_AT_TRAJ2toN = """
Currently on trial #{traj_idx}. Moves taken so far:
{current_trajectory}
"""

TRAJ_2toN_INIT = """
Currently on trial #{traj_idx}, starting from the initial board state.
"""

def parse_current_trajectory(turn_idx, traj_idx, curr_traj):
    if traj_idx == 0:
        if turn_idx == 0:
            return ""
        else:
            return CURR_TRAJ_AT_TRAJ1.format(current_trajectory=curr_traj)
    else:
        if turn_idx == 0:
            return TRAJ_2toN_INIT.format(traj_idx=traj_idx + 1)
        else:
            return CURR_TRAJ_AT_TRAJ2toN.format(
                traj_idx=traj_idx + 1,
                current_trajectory=curr_traj
            )

# ===============================
# Main Prompt Function
# ===============================

def get_kuhn_poker_prompt(phase: str = 'play',
                         turn_idx: int = 0,
                         traj_idx: int = 0,
                         init_observation: str = '',
                         curr_traj: str = '',
                         past_traj: list = None,
                         reflection: list = None,
                         reflection_type: str = 'reflection_only',
                         agent_starts: bool = True,
                         past_outcomes: list = None,
                         reflect_success: bool = False):
    """
    Build prompt for Kuhn Poker.
    init_observation is now a string returned by self.state.information_state_string(...)
    """
    past_traj = past_traj or []
    reflection = reflection or []
    past_outcomes = past_outcomes or []
    assert phase in ["play", "reflect"]

    # determine player id based on agent_starts
    agent_player_id = 0 if agent_starts else 1
    opponent_player_id = 1 - agent_player_id

    if phase == "play":
        past_trajectories_reflections = parse_reflection(
            traj_idx, past_traj, reflection, reflection_type, past_outcomes=past_outcomes
        )
        current_trajectory = parse_current_trajectory(turn_idx, traj_idx, curr_traj)
        return KUHN_POKER_PLAY_PROMPT.format(
            agent_player_id=agent_player_id,
            opponent_player_id=opponent_player_id,
            init_observation=init_observation,
            past_trajectories_reflections=past_trajectories_reflections,
            current_trajectory=current_trajectory
        ).strip()
    else:
        current_trajectory = parse_current_trajectory(turn_idx, traj_idx, curr_traj)
        if reflect_success:
            reflect_template = KUHN_POKER_REFLECT_PROMPT_WON
        else:
            reflect_template = KUHN_POKER_REFLECT_PROMPT
        return reflect_template.format(
            agent_player_id=agent_player_id,
            opponent_player_id=opponent_player_id,
            init_observation=init_observation,
            current_trajectory=current_trajectory
        ).strip()