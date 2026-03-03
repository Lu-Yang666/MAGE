# ===============================
# Play Prompt Template
# ===============================
TICTACTOE_PLAY_PROMPT = """
You are an expert agent playing Tic-Tac-Toe on a {board_size} by {board_size} board.
The rows and columns are indexed from 1 to {board_size}.

# Cell States
- Empty cells (.): cells that are not yet taken
- Player X (X): cells taken by player X
- Player O (O): cells taken by player O

# Game Rules
- A valid action is placing your mark on an empty cell (.).
- Each action is represented as a coordinate (row, col).
- At the INITIAL state, all of the following actions are valid:
   (1,1), (1,2), (1,3),
   (2,1), (2,2), (2,3),
   (3,1), (3,2), (3,3)
- At any later state, valid actions are ONLY those positions that are still empty (.).
- A player WINS if they place THREE of their own marks consecutively in a straight line.
- A straight line can be:
  1. A full ROW  
     Example: (2,1), (2,2), (2,3)
  2. A full COLUMN  
     Example: (1,3), (2,3), (3,3)
  3. A DIAGONAL  
     Examples:
     - Main diagonal: (1,1), (2,2), (3,3)
     - Anti-diagonal: (1,3), (2,2), (3,1)
- The game is a DRAW if:
  - All cells are filled AND neither player has achieved a winning line
- The game continues as long as:
  - No player has won AND there exists at least one empty cell (.)

# Your Goal
Your goal is to place your mark to win the game or prevent the opponent from winning.
You play as {player_symbol} and your opponent plays as {opponent_symbol}.

# Observation
The initial state of the game is:
{init_observation}{past_trajectories_reflections}{current_trajectory}

IMPORTANT: The action history (“moves taken so far”) contains ONLY YOUR actions.
The opponent acts automatically; its actions are NOT shown in the history and appear ONLY in the observation.
Now it's your turn to make a move.
- First reason step-by-step about the current board state and possible threats/opportunities. NOTE: In the given observation, cells marked with {opponent_symbol} are the opponent's moves. You canNOT place your mark there, and these cells do NOT count toward your winning line.
- Then choose ONE EMPTY cell (.) to place your mark.
- Put the index of the cell in the format of "(row, col)" within the <action> </action> tag.
"""

# ===============================
# Reflect Prompt Templates (lost/won)
# ===============================
TICTACTOE_REFLECT_PROMPT = """
You are an expert agent playing Tic-Tac-Toe on a {board_size} by {board_size} board.
The rows and columns are indexed from 1 to {board_size}.

# Cell States
- Empty cells (.): cells that are not yet taken
- Player X (X): cells taken by player X
- Player O (O): cells taken by player O

# Game Rules
- A valid action is placing your mark on an empty cell (.).
- Each action is represented as a coordinate (row, col).
- At the INITIAL state, all of the following actions are valid:
   (1,1), (1,2), (1,3),
   (2,1), (2,2), (2,3),
   (3,1), (3,2), (3,3)
- At ANY LATER state, valid actions are ONLY those positions that are still empty (.).
- A player WINS if they place THREE of their own marks consecutively in a straight line.
- A straight line can be:
  1. A full ROW  
     Example: (2,1), (2,2), (2,3)
  2. A full COLUMN  
     Example: (1,3), (2,3), (3,3)
  3. A DIAGONAL  
     Examples:
     - Main diagonal: (1,1), (2,2), (3,3)
     - Anti-diagonal: (1,3), (2,2), (3,1)
- The game is a DRAW if:
  - All cells are filled AND neither player has achieved a winning line
- The game continues as long as:
  - No player has won AND there exists at least one empty cell (.)
 
# Your Goal
Your goal is to place your mark to win the game or prevent the opponent from winning.
You play as {player_symbol} and your opponent plays as {opponent_symbol}.

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

TICTACTOE_REFLECT_PROMPT_WON = """
You are an expert agent playing Tic-Tac-Toe on a {board_size} by {board_size} board.
The rows and columns are indexed from 1 to {board_size}.

# Cell States
- Empty cells (.): cells that are not yet taken
- Player X (X): cells taken by player X
- Player O (O): cells taken by player O

# Game Rules
- A valid action is placing your mark on an empty cell (.).
- Each action is represented as a coordinate (row, col).
- At the INITIAL state, all of the following actions are valid:
   (1,1), (1,2), (1,3),
   (2,1), (2,2), (2,3),
   (3,1), (3,2), (3,3)
- At ANY LATER state, valid actions are ONLY those positions that are still empty (.).
- A player WINS if they place THREE of their own marks consecutively in a straight line.
- A straight line can be:
  1. A full ROW  
     Example: (2,1), (2,2), (2,3)
  2. A full COLUMN  
     Example: (1,3), (2,3), (3,3)
  3. A DIAGONAL  
     Examples:
     - Main diagonal: (1,1), (2,2), (3,3)
     - Anti-diagonal: (1,3), (2,2), (3,1)
- The game is a DRAW if:
  - All cells are filled AND neither player has achieved a winning line
- The game continues as long as:
  - No player has won AND there exists at least one empty cell (.)
 
# Your Goal
Your goal is to place your mark to win the game or prevent the opponent from winning.
You play as {player_symbol} and your opponent plays as {opponent_symbol}.

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
# Main function to get TicTacToe prompt
# ===============================
def get_tictactoe_prompt(board_size: int,
                         player_symbol: str = 'X',
                         opponent_symbol: str = 'O',
                         phase: str = 'play',
                         turn_idx: int = 0,
                         traj_idx: int = 0,
                         init_observation: str = '',
                         curr_traj: str = '',
                         past_traj: list = None,
                         reflection: list = None,
                         reflection_type: str = 'reflection_only',
                         past_outcomes: list = None,
                         reflect_success: bool = False):
    """
    New parameters:
      - past_outcomes: optional list of booleans for past trials (True=success/won)
      - reflect_success: when phase == 'reflect', choose won/failed reflect prompt
    """
    past_traj = past_traj or []
    reflection = reflection or []
    past_outcomes = past_outcomes or []

    assert phase in ['play', 'reflect']

    if phase == 'play':
        # include past trajectories + reflections using past_outcomes
        past_trajectories_reflections = parse_reflection(
            traj_idx, past_traj, reflection, reflection_type, past_outcomes=past_outcomes
        )
        current_trajectory = parse_current_trajectory(turn_idx, traj_idx, curr_traj)
        prompt = TICTACTOE_PLAY_PROMPT.format(
            board_size=board_size,
            player_symbol=player_symbol,
            opponent_symbol=opponent_symbol,
            init_observation=init_observation,
            past_trajectories_reflections=past_trajectories_reflections,
            current_trajectory=current_trajectory
        )
    else:
        # choose won/lost reflect prompt for the current reflection
        current_trajectory = parse_current_trajectory(turn_idx, traj_idx, curr_traj)
        if reflect_success:
            reflect_template = TICTACTOE_REFLECT_PROMPT_WON
        else:
            reflect_template = TICTACTOE_REFLECT_PROMPT
        prompt = reflect_template.format(
            board_size=board_size,
            player_symbol=player_symbol,
            opponent_symbol=opponent_symbol,
            init_observation=init_observation,
            current_trajectory=current_trajectory
        )
    return prompt.strip()