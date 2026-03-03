import ray
import gym
import numpy as np
import pyspiel
import random
import copy
from typing import Dict, Any, List, Optional, Tuple
import re
import openai

@ray.remote(num_cpus=0.1)
class KuhnPokerWorker:

    def __init__(self, env_kwargs: Dict[str, Any] = None):
        if env_kwargs is None:
            env_kwargs = {}
        self.config = env_kwargs

        self.env = pyspiel.load_game("kuhn_poker")
        self.state = self.env.new_initial_state()
        # initial ante
        self.bets = [1, 1]
        # agent config
        self.agent_starts = env_kwargs.get("agent_starts", True)
        self.opponent_policy = env_kwargs.get("opponent_policy", "random")
        self._build_opponent()
        self.state_copy = copy.deepcopy(self.state)
        self.bets_copy = copy.deepcopy(self.bets)

    def _build_opponent(self):
        if self.opponent_policy == "cfr":
            from open_spiel.python.algorithms import cfr
            import gzip, os, pickle
            self.cfr_solver = cfr.CFRSolver(self.env)
            cfr_checkpoint_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "cfr.pkl.gz"
            )
            if os.path.exists(cfr_checkpoint_path):
                print("loading cfr checkpoint!")
                with gzip.open(cfr_checkpoint_path, "rb") as f:
                    self.cfr_avg_policy = pickle.load(f)
            else:
                for _ in range(self.config.get("cfr_iterations", 1000)):
                    self.cfr_solver.evaluate_and_update_policy()
                self.cfr_avg_policy = self.cfr_solver.average_policy()
            self.opponent_policy = "cfr"
        elif self.opponent_policy == "conservative":
            self.opponent_policy = "conservative"
        elif self.opponent_policy == "aggressive":
            self.opponent_policy = "aggressive"
        elif self.opponent_policy == "intermediate":
            self.opponent_policy = "intermediate"
        else:
            raise ValueError(f"Unknown opponent: {self.opponent_policy}")

    def _get_rule_based_action(self):
        opp_id = 1 if self.agent_starts else 0
        info_state = self.state.information_state_string(opp_id)
        card = int(info_state[0])      
        history = info_state[1:]        

        if self.opponent_policy == "conservative":
            if card == 2:              
                return 1
            return 0                  
        if self.opponent_policy == "aggressive":
            if "b" in history:          
                if card == 0: return 0  
                return 1                
            return 1                  
        if self.opponent_policy == "intermediate":
            if card == 2: return 1      
            if card == 0: return 0      
            if card == 1:              
                return 0 if "b" in history else 1
               
        return 0

    def _opponent_cfr(self):
        opponent_id = 1 if self.agent_starts else 0  # 0 or 1
        action_probs = self.cfr_avg_policy.action_probabilities(
            self.state, opponent_id
        )
        # action_probs: Dict[action_id, prob]
        legal_actions = self.state.legal_actions()
        best = max(legal_actions, key=lambda a: action_probs.get(a, 0.0))
        return best

    def _get_full_obs(self):
        card_map = {0: "Jack", 1: "Queen", 2: "King"}
        hist = self.state.history()
        cards = {
            "player 0 card": card_map[hist[0]] if len(hist) > 0 else None,
            "player 1 card": card_map[hist[1]] if len(hist) > 1 else None,
        }
        return cards

    def _rollout_chance(self):
        np.random.seed(self.config['seed'])
        while self.state.is_chance_node():
            outcomes = self.state.chance_outcomes()
            actions, probs = zip(*outcomes)
            a = random.choices(actions, probs)[0]
            self.state.apply_action(a)

    def get_readable_obs(self, info_str):
        card_map = {"0": "Jack", "1": "Queen", "2": "King"}
        my_card_raw = info_str[0]
        my_card_name = card_map.get(my_card_raw, my_card_raw)
        actions_raw = info_str[1:]
        action_list = []
       
        for i, char in enumerate(actions_raw):
                player_id = i % 2  # 0, 1, 0, 1...
                action_name = "Pass" if char == 'p' else "Bet"
                action_list.append(f"Player {player_id}: {action_name}")
           
        actions_display = " -> ".join(action_list) if action_list else "No actions yet"
        readable_obs = f"Your Card: {my_card_name} | Actions: {actions_display}"
       
        return readable_obs


    def is_winnable_oracle(self):
        card_map = {"Jack": 0, "Queen": 1, "King": 2}
        full_obs = self._get_full_obs()
        p0_card = card_map[full_obs["player 0 card"]]
        p1_card = card_map[full_obs["player 1 card"]]
       
        agent_id = 0 if self.agent_starts else 1
        opp_id = 1 - agent_id
        agent_card = p0_card if agent_id == 0 else p1_card
        opp_card = p0_card if opp_id == 0 else p1_card

        def get_opp_action(history_has_bet, current_opp_card):
            policy = self.opponent_policy
            if policy == "conservative":
                return 1 if current_opp_card == 2 else 0
            elif policy == "aggressive":
                if history_has_bet: return 0 if current_opp_card == 0 else 1
                return 1  
            elif policy == "intermediate":
                if current_opp_card == 2: return 1
                if current_opp_card == 0: return 0
                return 0 if history_has_bet else 1 
            elif policy == "cfr":
                if not history_has_bet:
                    return 1 if current_opp_card == 2 else 0 
                else:
                    return 1 if current_opp_card == 2 else 0 
            return 0

        if self.agent_starts:
            opp_res_a = get_opp_action(False, opp_card)
            if opp_res_a == 0:
                if agent_card > opp_card: return True
            else:
                if agent_card > opp_card: return True
            opp_res_b = get_opp_action(True, opp_card)
            if opp_res_b == 0:
                return True
            else:
                if agent_card > opp_card: return True
        else:
            opp_first_act = get_opp_action(False, opp_card)
            if opp_first_act == 0:
                if agent_card > opp_card: return True
                opp_res_to_bluff = get_opp_action(True, opp_card)
                if opp_res_to_bluff == 0: return True
            else:
                if agent_card > opp_card: return True

        return False

    def reset(self, seed_for_reset: Optional[int] = None):
        if seed_for_reset is not None:
            np.random.seed(seed_for_reset)
            random.seed(seed_for_reset)
        self.state = self.env.new_initial_state()
        self._rollout_chance()
        self.bets = [1, 1]
        # If agent starts = False => opponent plays first as player0
        if not self.agent_starts:
            self._opponent_play()
        self.state_copy = copy.deepcopy(self.state)
        self.bets_copy = copy.deepcopy(self.bets)
        obs = self.state.information_state_string(1 if not self.agent_starts else 0)
        obs = self.get_readable_obs(obs)
        return obs, {
            "opponent": self.opponent_policy,
            "agent_starts": self.agent_starts,
            "won": False,
            "draw": False,
            "score": self.state.returns()[0 if self.agent_starts else 1],
            "full_obs": f"Your observation: {obs}, cards info: {self._get_full_obs()}",
            "is_winnable": self.is_winnable_oracle()
        }

    def restart(self):
        self.state = copy.deepcopy(self.state_copy)
        self.bets = copy.deepcopy(self.bets_copy)
        obs = self.state.information_state_string(1 if not self.agent_starts else 0)
        obs = self.get_readable_obs(obs)
        return obs, {
            "opponent": self.opponent_policy,
            "agent_starts": self.agent_starts,
            "won": False,
            "score": self.state.returns()[0 if self.agent_starts else 1],
            "full_obs": f"Your observation: {obs}, cards info: {self._get_full_obs()}"
        }

    def _opponent_play(self):
        """
        Opponent plays first as player0.
        """
        if self.opponent_policy == "cfr":
            a = self._opponent_cfr()
        elif self.opponent_policy in ["conservative", "aggressive", "intermediate"]:
            a = self._get_rule_based_action()
        else:
            raise ValueError("Unknown opponent policy")
        self.state.apply_action(a)

    def step(self, action: int):
        """
        Agent acts (player1 if agent_starts=False else player0).
        Then opponent plays.
        """
        # 1) if terminate, return
        if self.state.is_terminal():
            agent_player_id = 0 if self.agent_starts else 1
            obs = self.state.information_state_string(agent_player_id)
            obs = self.get_readable_obs(obs)
            score = self.state.returns()[0 if self.agent_starts else 1]
            print("terminal score:", score, "terminal reward:", score*5)
            return obs, 5*score, True, {
                "opponent": self.opponent_policy,
                "won": score == 2,
                "draw": False,
                "score": score,
                "is_action_valid": True,
                "full_obs": f"Your observation: {obs}, cards info: {self._get_full_obs()}"
            }

        agent_player_id = 0 if self.agent_starts else 1
        opponent_player_id = 1 - agent_player_id

        # 2) check whether agent action is legal
        legal_actions = self.state.legal_actions()
        if action not in legal_actions:
            obs = self.state.information_state_string(agent_player_id)
            obs = self.get_readable_obs(obs)
            return obs, 0.0, False, {
                "opponent": self.opponent_policy,
                "won": False,
                "draw": False,
                "score": self.state.returns()[0 if self.agent_starts else 1],
                "is_action_valid": False,
                "full_obs": f"Your observation: {obs}, cards info: {self._get_full_obs()}"
            }

        # 3) agent action
        self.state.apply_action(action)

        # 4) terminate after agent action
        if self.state.is_terminal():
            payoff = self.state.returns()[agent_player_id]
            reward = 5 * payoff
            # reward = 10 if payoff > 0 else (-10 if payoff < 0 else 0)
            obs = self.state.information_state_string(agent_player_id)
            obs = self.get_readable_obs(obs)
            return obs, reward, True, {
                "opponent": self.opponent_policy,
                "won": payoff > 0,
                "score": payoff,
                "draw": payoff == 0 or abs(payoff) < 1e-6,
                "is_action_valid": True,
                "full_obs": f"Your observation: {obs}, cards info: {self._get_full_obs()}"
            }

        # 5) opponent plays
        self._opponent_play()

        # 6) terminate after oppnent action
        if self.state.is_terminal():
            payoff = self.state.returns()[agent_player_id]
            reward = 5 * payoff
            # reward = 10 if payoff > 0 else (-10 if payoff < 0 else 0)
            obs = self.state.information_state_string(agent_player_id)
            obs = self.get_readable_obs(obs)
            return obs, reward, True, {
                "opponent": self.opponent_policy,
                "won": payoff > 0,
                "draw": payoff == 0 or abs(payoff) < 1e-6,
                "score": payoff,
                "is_action_valid": True,
                "full_obs": f"Your observation: {obs}, cards info: {self._get_full_obs()}"
            }

        # 7) game continues
        obs = self.state.information_state_string(agent_player_id)
        obs = self.get_readable_obs(obs)
        print("continue score", self.state.returns()[agent_player_id])
        return obs, 0.0, False, {
            "opponent": self.opponent_policy,
            "won": False,
            "draw": False,
            "score": self.state.returns()[agent_player_id],
            "is_action_valid": True,
            "full_obs": f"Your observation: {obs}, cards info: {self._get_full_obs()}"
        }

    def render(self, mode='text'):
        if mode != 'text':
            raise ValueError("Only 'text' render supported")
        return str(self.state)

class KuhnPokerMultiProcessEnv(gym.Env):
    """
    Ray wrapper for Kuhn Poker with opponent.
    """

    def __init__(
        self,
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 1,
        is_train: bool = True,
        env_kwargs: Dict[str, Any] = None
    ):
        super().__init__()
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        self.env_num = env_num
        self.group_n = group_n
        self.num_processes = env_num * group_n
        self.is_train = is_train
        self.seed = seed
        np.random.seed(seed)

        if env_kwargs is None:
            env_kwargs = {}

        # opponent list support
        self.opponent_list = self.generate_opponent_list(
            opponent_policy=env_kwargs.get("opponent_policy", None),
            opponent_policy_pool=env_kwargs.get("opponent_policy_pool", None)
        )
        print(f"opponent list: {self.opponent_list}")
        # agent_starts list support
        self.agent_starts_list = self._build_agent_starts_list(env_kwargs)
        print(f"agent_starts: {self.agent_starts_list}")

        self.workers = []
        for i in range(self.num_processes):
            kw = dict(env_kwargs)
            kw['seed'] = self.seed
            kw["opponent_policy"] = self.opponent_list[i]
            kw["agent_starts"] = self.agent_starts_list[i]
            self.workers.append(KuhnPokerWorker.remote(kw))

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
            random.seed(self.seed)
            random.shuffle(agent_starts)
            return agent_starts
        raise ValueError(f"Unknown play_mode: {play_mode}")

    def generate_opponent_list(self, opponent_policy, opponent_policy_pool):
        """
        Generate the list of opponent policies based on the given pool and ratios.
       
        Args:
            opponent_policy: str or list/tuple, default None. The policy to be used when opponent_policy_pool is not provided.
            opponent_policy_pool: List of dicts with keys {"policy": str, "ratio": float}, specifying the policies and their ratios.
       
        Returns:
            opponent_list: List of policies with length equal to num_processes.
        """
        n = self.num_processes
        group_n = self.group_n
        env_num = self.env_num
        if opponent_policy is not None:
            # If opponent_policy is provided, use it directly
            opponent_list = [opponent_policy] * n
        elif opponent_policy_pool is not None:
            # opponent_policy_pool: list of dicts {"policy": str, "ratio": float}
            total = self.num_processes
            policy_counts = []

            for i, item in enumerate(opponent_policy_pool):
                policy = item["policy"]
                ratio = float(item.get("ratio", 0.0))
                env_count = int(round(ratio * env_num))
                policy_counts.append((policy, env_count * group_n))
            # Generate opponent_list
            opponent_list = []
            for policy, count in policy_counts:
                opponent_list += [policy] * count
            # If the rounding error causes too few/many, fix it
            if len(opponent_list) < total:
                opponent_list += ["intermediate"] * (total - len(opponent_list))
            elif len(opponent_list) > total:
                opponent_list = opponent_list[:total]
        else:
            # If neither opponent_policy nor opponent_policy_pool is provided, default to "random"
            opponent_list = ["intermediate"] * self.num_processes
       
        return opponent_list
     
    def reset(self):
        if self.is_train:
            seeds = np.random.randint(0, 2**16-1, size=self.env_num)
        else:
            seeds = np.random.randint(2**16, 2**32-1, size=self.env_num)
        seeds = np.repeat(seeds, self.group_n).tolist()
        futures = [w.reset.remote(s) for w, s in zip(self.workers, seeds)]
        results = ray.get(futures)
        obs_list, info_list = zip(*results)
        winnable = 0
        for w in list(info_list):
            if w['is_winnable']:
                winnable += 1
        print(f"winnable envs: {winnable}, total envs: {len(list(info_list))}, upper bound: {winnable/len(list(info_list))}")
        return list(obs_list), list(info_list)

    def restart(self):
        futures = [w.restart.remote() for w in self.workers]
        results = ray.get(futures)
        obs_list, info_list = zip(*results)
        return list(obs_list), list(info_list)

    def step(self, actions: List[int]):
        futures = [w.step.remote(a) for w, a in zip(self.workers, actions)]
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = zip(*results)
        print(f"action example: [{actions[0]}, {actions[1]}], observation example: [{obs_list[0]}, {obs_list[1]}]")
        return list(obs_list), np.array(reward_list), np.array(done_list), list(info_list)

    def render(self, env_idx: Optional[int] = None, mode='text'):
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

def build_kuhnpoker_fixed_opponent_envs(seed=0, env_num=1, group_n=1, is_train=True, env_kwargs=None):
    return KuhnPokerMultiProcessEnv(seed, env_num, group_n, is_train, env_kwargs)



