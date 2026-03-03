import re
import copy
from typing import List, Tuple

def tictactoe_projection(actions: List[str], board_size: int = 3, phase: str = 'play'):
    """
    Parse agent output for Tic-Tac-Toe.
    Args:
        actions: list of agent output strings
        board_size: 3 for normal Tic-Tac-Toe
        phase: 'play' or 'reflect'
    Returns:
        If phase=='play':
            plans: list of extracted plans (optional, can be empty)
            actions: list of (row, col) tuples
            valids: list of 0/1 flags indicating valid actions
        If phase=='reflect':
            reflections: list of extracted reflections
            valids: list of 0/1 flags
    """
    actions = copy.deepcopy(actions)
    if phase == 'play':
        valids = [0] * len(actions)
        plans = [''] * len(actions)
        for i, text in enumerate(actions):
            original_text = text
            # try extract <action>...</action>
            start_idx = text.find("<action>")
            end_idx = text.find("</action>")
            if start_idx == -1 or end_idx == -1:
                # invalid, fallback
                valids[i] = 0
                actions[i] = (-1, -1)
            else:
                extracted_action = text[start_idx + len("<action>"): end_idx].strip()
                # match format (row, col)
                match = re.search(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", extracted_action)
                if match:
                    row, col = match.groups()
                    actions[i] = (int(row), int(col))
                    valids[i] = 1
                else:
                    actions[i] = (-1, -1)
                    valids[i] = 0
            # optional: extract <plan>...</plan>
            plan_start = original_text.rfind("<plan>")
            plan_end = original_text.rfind("</plan>")
            if plan_start == -1 or plan_end == -1:
                plans[i] = ''
            else:
                plans[i] = original_text[plan_start + len("<plan>"): plan_end].strip()
        return plans, actions, valids

    else:  # reflect phase
        valids = [0] * len(actions)
        reflections = [''] * len(actions)
        for i, text in enumerate(actions):
            start_tag = "<remark>"
            end_tag = "</remark>"
            start_idx = text.rfind(start_tag)
            end_idx = text.rfind(end_tag)
            if start_idx == -1 or end_idx == -1:
                reflections[i] = ''
            else:
                reflections[i] = text[start_idx + len(start_tag): end_idx].strip()[:2000]  # max 2000 chars
                valids[i] = 1
        return reflections, valids