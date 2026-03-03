import re
import copy
from typing import List, Tuple

def kuhn_poker_projection(actions: List[str], phase: str = 'play'):
    """
    Parse agent output for Kuhn Poker.
    Args:
        actions: list of agent output strings
        phase: 'play' or 'reflect'
    Returns:
        If phase=='play':
            plans: list of extracted plans (optional, can be empty)
            actions: list of integers indicating the action choice (0: PASS, 1: BET)
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
            # Try to extract <action>...</action>
            start_idx = text.find("<action>")
            end_idx = text.find("</action>")
            if start_idx == -1 or end_idx == -1:
                # Invalid, fallback
                valids[i] = 0
                actions[i] = -1  # -1 indicates invalid action
            else:
                extracted_action = text[start_idx + len("<action>"): end_idx].strip()
                # Match for action (either "PASS" or "BET")
                if extracted_action == "PASS":
                    actions[i] = 0  # PASS action
                    valids[i] = 1
                elif extracted_action == "BET":
                    actions[i] = 1  # BET action
                    valids[i] = 1
                else:
                    actions[i] = -1  # Invalid action
                    valids[i] = 0

            # Optional: extract <plan>...</plan>
            plan_start = original_text.find("<plan>")
            plan_end = original_text.find("</plan>")
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
            start_idx = text.find(start_tag)
            end_idx = text.find(end_tag)
            if start_idx == -1 or end_idx == -1:
                reflections[i] = ''
            else:
                reflections[i] = text[start_idx + len(start_tag): end_idx].strip()[:2000]  # max 2000 chars
                valids[i] = 1

        return reflections, valids
