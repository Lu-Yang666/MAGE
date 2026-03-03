import ray
import gym
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable
import copy
import random
import math
import pyspiel
from open_spiel.python.algorithms.mcts import MCTSBot, RandomRolloutEvaluator

class TicTacToe:
    """
    Tic-Tac-Toe core game.
    - board_size: board_size x board_size
    - players: 1 (agent, X) and 2 (opponent, O)
    - actions passed to step() are 0-based (row, col)
    """
    def __init__(self, board_size: int = 3):
        self.board_size = int(board_size)
        self.reset()

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(int(seed) % (2**31))
            random.seed(int(seed) % (2**31))
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)  # 0 empty, 1 X, 2 O
        self.current_player = 1  # agent always starts as player 1 in this design
        self.done = False
        self.winner: Optional[int] = None
        self.action_seq = []
        return self.render_board(), {"won": False}

    def copy(self):
        new = TicTacToe(self.board_size)
        new.board = self.board.copy()
        new.current_player = self.current_player
        new.done = self.done
        new.winner = self.winner
        new.action_seq = list(self.action_seq)
        return new

    def legal_actions(self) -> List[Tuple[int,int]]:
        coords = list(zip(*np.where(self.board == 0)))
        return [(int(r), int(c)) for (r,c) in coords]

    def check_win(self, player: int) -> bool:
        b = self.board
        n = self.board_size
        # rows / cols
        for i in range(n):
            if np.all(b[i, :] == player) or np.all(b[:, i] == player):
                return True
        # diagonals
        if np.all(np.diag(b) == player) or np.all(np.diag(np.fliplr(b)) == player):
            return True
        return False

    def render_board(self) -> str:
        symbols = {0: ".", 1: "X", 2: "O"}
        lines = []
        for i in range(self.board_size):
            lines.append(" ".join(symbols[int(self.board[i, j])] for j in range(self.board_size)))
        return "\n".join(lines)

def opponent_policy_random(board: np.ndarray) -> Tuple[int,int]:
    empties = list(zip(*np.where(board == 0)))
    return tuple(map(int, random.choice(empties)))

# Preferred positions policy: try positions in priority order
def make_preferred_policy(pref_list: List[Tuple[int,int]]) -> Callable[[np.ndarray], Tuple[int,int]]:
    def policy(board: np.ndarray):
        for pos in pref_list:
            r,c = pos
            if board[r,c] == 0:
                return (r,c)
        return opponent_policy_random(board)
    return policy

# Biased random: weight empty cells by a heuristic (closer to center, or corners)
def make_biased_random_policy(bias: str = 'center', alpha: float = 2.0) -> Callable[[np.ndarray], Tuple[int,int]]:
    def policy(board: np.ndarray):
        n = board.shape[0]
        empties = list(zip(*np.where(board == 0)))
        if not empties:
            raise RuntimeError("no empty cells")
        weights = []
        center = ((n-1)/2.0, (n-1)/2.0)
        for (r,c) in empties:
            if bias == 'center':
                d = math.hypot(r-center[0], c-center[1])
                w = (1.0 / (1.0 + d))**alpha
            elif bias == 'corner':
                corners = [(0,0),(0,n-1),(n-1,0),(n-1,n-1)]
                mind = min(math.hypot(r-cr, c-cc) for (cr,cc) in corners)
                w = (1.0 / (1.0 + mind))**alpha
            else:
                w = 1.0
            weights.append(w)
        # normalize
        s = sum(weights)
        probs = [w/s for w in weights]
        return empties[np.random.choice(len(empties), p=probs)]
    return policy

# mapping name -> func (basic)
OPPONENT_POLICIES_BASIC = {
    "random": opponent_policy_random,
}

@ray.remote(num_cpus=0.1)
class TicTacToeWorker:
    def __init__(self, env_kwargs: Dict[str, Any] = None):
        if env_kwargs is None:
            env_kwargs = {}
        board_size = env_kwargs.get("board_size", 3)
        self.env = TicTacToe(board_size=board_size)
        self.agent_starts = env_kwargs.get("agent_starts", True)

        # choose opponent policy (after board created)
        opp_spec = env_kwargs.get("opponent_policy", "rule")
        if isinstance(opp_spec, list):
            self.opponent_name = opp_spec[0]
        else:
            self.opponent_name = opp_spec
        self.opponent_policy = self._parse_policy_spec(opp_spec, env_kwargs)
        # store copy for restart
        self.env_copy = self.env.copy()

    def _parse_policy_spec(self, spec, env_kwargs):
        """
        Accept various spec forms and return a callable(policy_fn(board)->(r,c)):
          - callable -> used directly
          - string -> map to builtins, supports 'mcts' specially
          - tuple/list -> (name, params) e.g. ('fixed',(0,0)), ('pattern',[(0,0),(0,1)])
        """
        # callable
        if callable(spec):
            return spec
        # string name
        if isinstance(spec, str):
            if spec == 'mcts':
                # build MCTS bot here using env_kwargs
                self.osl_game = pyspiel.load_game("tic_tac_toe")
                evaluator = RandomRolloutEvaluator(1, np.random.RandomState())
                self.mcts_bot = MCTSBot(self.osl_game,
                                        uct_c=1.4,
                                        max_simulations=env_kwargs.get('mcts_sim_num', 100),
                                        evaluator=evaluator,
                                        random_state=np.random.RandomState(),
                                        solve=True,
                                        verbose=False)
                print("mcts_sim_num: ", env_kwargs.get('mcts_sim_num', 100))
                return 'mcts'
            return OPPONENT_POLICIES_BASIC.get(spec, opponent_policy_random)
        # tuple/list spec
        if isinstance(spec, (tuple, list)) and len(spec) >= 1:
            name = spec[0]
            params = spec[1] if len(spec) > 1 else None
            if name == 'preferred' and isinstance(params, list):
                return make_preferred_policy(params)
            if name == 'biased_random' and isinstance(params, dict):
                bias = params.get('bias','center')
                alpha = float(params.get('alpha',2.0))
                return make_biased_random_policy(bias=bias, alpha=alpha)
            # fallback if name maps to basic
            if isinstance(name, str):
                return OPPONENT_POLICIES_BASIC.get(name, opponent_policy_random)
        # fallback
        return opponent_policy_random

    def _opponent_play_first(self):
        """Make the starting (opponent) play one move -- called when agent is configured as second."""
        n = self.env.board_size
        # if using MCTS, build state and call mcts; else use policy callable
        if getattr(self, 'opponent_policy', None) == "mcts":
            state = self.osl_game.new_initial_state()
            osl_action = self.mcts_bot.step(state)
            opp_r = osl_action // n
            opp_c = osl_action % n
        else:
            opp_r, opp_c = self.opponent_policy(self.env.board.copy())

        # safety: fallback if invalid
        if not (0 <= opp_r < n and 0 <= opp_c < n) or self.env.board[opp_r, opp_c] != 0:
            print("warning! random action!")
            empties = list(zip(*np.where(self.env.board == 0)))
            opp_r, opp_c = tuple(map(int, random.choice(empties)))
        # Place opponent (player 2) mark
        self.env.board[opp_r, opp_c] = 2
        self.env.action_seq.append((opp_r, opp_c))

    def step(self, action: Tuple[int,int]):
        """
        Execute agent (player 1) action, then if not terminal simulate opponent move.
        action: (row, col) 0-based indices.
        Returns: obs_str, reward_for_agent (float), done (bool), info (dict)
        info contains zero_sum mapping: info['zero_sum'] = {'agent': r_a, 'opponent': r_b}
        """
        # If game already done, return terminal observation (no further rewards)
        if self.env.done:
            if self.env.check_win(1):
                info = {"won": True, "draw": False, "is_action_valid": True, "opponent": self.opponent_name}
            elif self.env.check_win(2):
                info = {"won": False, "draw": False, "is_action_valid": True, "opponent": self.opponent_name}
            else:
                info = {"won": False, "draw": True, "is_action_valid": True, "opponent": self.opponent_name}
            return self.env.render_board(), 0.0, True, info

        # Validate action is in-bounds and empty
        r, c = action
        r, c = r-1, c-1
        # print(f"action: {action}, (r,c): ({r}, {c})")
        n = self.env.board_size
        info = {"won": False, "draw": False, "is_action_valid": True, "opponent": self.opponent_name}

        if not (0 <= r < n and 0 <= c < n):
            info["is_action_valid"] = False
            return self.env.render_board(), 0.0, False, info
        if self.env.board[r, c] != 0:
            info["is_action_valid"] = False
            return self.env.render_board(), 0.0, False, info

        # Apply agent move (player 1 -> mark 1)
        self.env.board[r, c] = 1
        self.env.action_seq.append((r,c))

        # Check if player (X) wins after their move
        if self.env.check_win(1):
            self.env.done = True
            self.env.winner = 1
            info["won"] = True
            return self.env.render_board(), 10.0, True, info

        # Check if the game ends in a draw after the player's move
        if np.all(self.env.board != 0):  # all cells filled
            self.env.done = True
            self.env.winner = 0  # draw
            info["draw"] = True
            return self.env.render_board(), 0.0, True, info

        # Not terminal: opponent plays one move (fixed policy)
        if getattr(self, 'opponent_policy', None) == "mcts":
            state = self.osl_game.new_initial_state()
            for (rr,cc) in self.env.action_seq:
                action_idx = rr * self.env.board_size + cc
                state.apply_action(action_idx)
            osl_action = self.mcts_bot.step(state)
            opp_r = osl_action // self.env.board_size
            opp_c = osl_action % self.env.board_size
        else:
            opp_r, opp_c = self.opponent_policy(self.env.board.copy())

        # safety: policy should return valid empty, but check
        if not (0 <= opp_r < n and 0 <= opp_c < n) or self.env.board[opp_r, opp_c] != 0:
            print("Warning! random action!")
            empties = list(zip(*np.where(self.env.board == 0)))
            opp_r, opp_c = tuple(map(int, random.choice(empties)))

        self.env.board[opp_r, opp_c] = 2  # opponent marks O
        self.env.action_seq.append((opp_r, opp_c))

        # Check if the opponent (O) wins after their move
        if self.env.check_win(2):
            self.env.done = True
            self.env.winner = 2
            info["won"] = False
            info["draw"] = False
            return self.env.render_board(), -10.0, True, info

        # Check if the game ends in a draw after the opponent's move
        if np.all(self.env.board != 0):  # all cells filled
            self.env.done = True
            self.env.winner = 0  # draw
            info["draw"] = True
            return self.env.render_board(), 0.0, True, info

        # Otherwise game continues: no reward yet
        self.env.current_player = 1  # back to agent
        return self.env.render_board(), 0.0, False, info

    def reset(self, seed_for_reset: Optional[int] = None):
        obs, info = self.env.reset(seed_for_reset)
        # if opponent plays first, do it after policy is set
        if not self.agent_starts:
            print("agent second")
            self._opponent_play_first()
        obs = self.env.render_board()
        info["opponent"] = self.opponent_name
        self.env_copy = self.env.copy()
        return obs, info

    def restart(self):
        self.env = self.env_copy.copy()
        obs = self.env.render_board()
        info = {"won": False, "zero_sum": {"agent": 0.0, "opponent": 0.0}, "opponent": self.opponent_name}
        return obs, info

    def render(self, mode_for_render='board'):
        if mode_for_render == 'board':
            return self.env.render_board()
        else:
            raise ValueError("Unsupported render mode")

class TicTacToeMultiProcessEnv(gym.Env):
    def __init__(self,
                 seed: int = 0,
                 env_num: int = 1,
                 group_n: int = 1,
                 is_train: bool = True,
                 env_kwargs: Dict[str, Any] = None):
        super().__init__()
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        self.is_train = is_train
        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        np.random.seed(seed)
        if env_kwargs is None:
            env_kwargs = {}
        # create per-worker agent_starts
        agent_starts_list = self._build_agent_starts_list(env_kwargs)
        print(f"agent_starts: {agent_starts_list}")
        # create per-worker opponent policies
        opponent_policy_list = self._build_opponent_policies(env_kwargs)
        print(f"opponent list: {opponent_policy_list}")
        self.workers = []
        for i in range(self.num_processes):
            kw = dict(env_kwargs)
            kw["agent_starts"] = agent_starts_list[i]
            kw["opponent_policy"] = opponent_policy_list[i]
            self.workers.append(TicTacToeWorker.remote(kw))

    def _build_agent_starts_list(self, cfg):
        play_mode = cfg.get("play_mode", "first")
        ratio = cfg.get("second_player_ratio", 0.5)
        n = self.num_processes

        if play_mode == "first":
            return [True] * n
        if play_mode == "second":
            return [False] * n
        if play_mode == "mixed":
            num_second = int(round(n * ratio))
            agent_starts = [False] * num_second + [True] * (n - num_second)
            random.shuffle(agent_starts)
            return agent_starts
        raise ValueError(f"Unknown play_mode: {play_mode}")

    def _build_opponent_policies(self, cfg) -> List[Any]:
        n = self.num_processes
        group_n = self.group_n
        opp = cfg.get('opponent_policy', None)
        pool = cfg.get('opponent_policy_pool', None)
        mixed = cfg.get('mixed_group', False)
        if mixed:
            print("==========shuffle! mixed group!")
        if opp is not None:
            return [opp] * n

        if pool is not None:
            final_policies = []
            for policy in pool:
                final_policies.extend([policy] * group_n)
                if len(final_policies) >= n:
                    if mixed:
                        random.shuffle(final_policies)
                    return final_policies[:n]
        
            remainder = n - len(final_policies)
            final_policies.extend(['mcts'] * remainder)
            if mixed:
                random.shuffle(final_policies)
            return final_policies
        return ['mcts'] * n

    def step(self, actions: List[Tuple[int,int]]):
        assert len(actions) == self.num_processes
        futures = [w.step.remote(a) for w, a in zip(self.workers, actions)]
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = zip(*results)
        print(f"action examples: [{actions[0]}, {actions[1]}, {actions[2]}]")
        return list(obs_list), np.array(reward_list), np.array(done_list), list(info_list)

    def reset(self):
        if self.is_train:
            seeds = np.random.randint(0, 2**16-1, size=self.env_num)
        else:
            seeds = np.random.randint(2**16, 2**32-1, size=self.env_num)
        seeds = np.repeat(seeds, self.group_n).tolist()
        futures = [w.reset.remote(s) for w, s in zip(self.workers, seeds)]
        results = ray.get(futures)
        obs_list, info_list = zip(*results)
        return list(obs_list), list(info_list)

    def restart(self):
        futures = [w.restart.remote() for w in self.workers]
        results = ray.get(futures)
        obs_list, info_list = zip(*results)
        return list(obs_list), list(info_list)

    def render(self, mode='board', env_idx: Optional[int] = None):
        if env_idx is not None:
            return ray.get(self.workers[env_idx].render.remote(mode))
        futures = [w.render.remote(mode) for w in self.workers]
        return ray.get(futures)

    def close(self):
        for w in self.workers:
            try:
                ray.kill(w)
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

def build_tictactoe_fixed_opponent_envs(seed=0, env_num=1, group_n=1, is_train=True, env_kwargs=None):
    return TicTacToeMultiProcessEnv(seed=seed, env_num=env_num, group_n=group_n, is_train=is_train, env_kwargs=env_kwargs)
