from typing import List
import numpy as np
from functools import partial


from omegaconf import OmegaConf


from .memory import SimpleMemoryTicTacToe
from .projection import tictactoe_projection
from .prompt import get_tictactoe_prompt
from ..base import EnvironmentManagerBase
from .envs_single import build_tictactoe_fixed_opponent_envs

def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)

class TicTacToeEnvironmentManagerFixedOpponent(EnvironmentManagerBase):
    """
    EnvironmentManager for Tic-Tac-Toe with FIXED opponent strategy.
    - Agent acts only in play phase
    - Opponent move happens inside env.step()
    - Reflection phase does NOT step env
    """
    def __init__(self, envs, projection_f, num_attempts, do_reflection, config):
        self.envs = envs
        self.projection_f = projection_f
        self.num_attempts = num_attempts
        self.do_reflection = do_reflection
        self.config = config
        self.num_processes = envs.num_processes
        self.max_turns = config.env.get("max_turns", 9)
        # trajectory bookkeeping
        self.curr_turn_idx = 0
        self.curr_traj_idx = 0
        # init states
        self.init_states = [None for _ in range(self.num_processes)]
        # memory: ONLY agent steps
        self.memories = [SimpleMemoryTicTacToe() for _ in range(num_attempts)]
        # reflections per env
        self.reflections = [{} for _ in range(self.num_processes)]
        super().__init__(envs, projection_f, config)

    # =========================
    # Reset / Restart
    # =========================
    def reset(self):
        obs, infos = self.envs.reset()
        self.init_states = obs
        for mem in self.memories:
            mem.reset(self.num_processes)
        self.reflections = [{} for _ in range(self.num_processes)]
        self.curr_turn_idx = 0
        self.curr_traj_idx = 0
        observations = {
            "text": self.build_text_obs(phase="play"),
            "image": None,
            "anchor": [o + info["opponent"] for o, info in zip(obs, infos)],
        }
        return observations, infos
    
    def restart(self):
        obs, infos = self.envs.restart()
        if self.do_reflection:
            self.curr_traj_idx += 1
        self.curr_turn_idx = 0
        observations = {
            "text": self.build_text_obs(phase="play"),
            "image": None,
            "anchor":  [o + info["opponent"] for o, info in zip(obs, infos)],
        }
        return observations, infos

    # =========================
    # Reflect phase
    # =========================

    def reflect(self):
        observations = {
            "text": self.build_text_obs(phase="reflect"),
            "image": None,
            "anchor": ["reflection" + self.envs.opponent_list[i] for i in range(self.num_processes)],
        }
        infos = [{"is_action_valid": True, "won": False}
                 for _ in range(self.num_processes)]
        return observations, infos

    # =========================
    # Step
    # =========================

    def step(self, text_actions: List[str], phase: str = "play"):
        assert phase in ["play", "reflect"]
        # ---------- Reflect ----------
        if phase == "reflect":
            reflections, valids = self.projection_f(
                text_actions, phase="reflect"
            )
            for i, r in enumerate(reflections):
                self.reflections[i][self.curr_traj_idx] = r
            rewards = np.array(valids, dtype=np.float32)
            dones = np.array([False] * self.num_processes)
            infos = [
                {"is_action_valid": to_numpy(valids[i]), "won": False}
                for i in range(self.num_processes)
            ]
            next_obs = {"text": "", "image": None, "anchor": ["reflection" + self.envs.opponent_list[i] for i in range(self.num_processes)]}

            return next_obs, rewards, dones, infos

        # ---------- Play ----------
        thoughts, actions, valids = self.projection_f(
            text_actions, phase="play"
        )

        # env.step executes:
        # agent move -> opponent fixed move -> terminal check
        next_obs, rewards, dones, infos = self.envs.step(actions)

        for i in range(self.num_processes):
            if "is_action_valid" in infos[i] and infos[i]["is_action_valid"] is not None:
                valids[i] = infos[i]["is_action_valid"]

            if not valids[i]:
                actions[i] = str(actions[i]) + ", but it is invalid"
                # actions[i] = "invalid operation"
            else:
                actions[i] = str(actions[i])

        # store ONLY agent perspective
        self.memories[self.curr_traj_idx].store({
            "text_obs": next_obs,
            "thought": thoughts,
            "action": actions,
            "reward": rewards,
            "dones": dones,
            "won": [info.get("won", False) for info in infos],
        })

        self.curr_turn_idx += 1

        observations = {
            "text": self.build_text_obs(phase="play"),
            "image": None,
            "anchor": [o + info["opponent"] for o, info in zip(next_obs, infos)],
        }

        return (
            observations,
            to_numpy(rewards),
            to_numpy(dones),
            infos,
        )

    # =========================
    # Prompt construction
    # =========================

    def build_text_obs(self, phase: str = "play"):
        post_obs = []

        obs_length = 2 if phase == "play" else 7
        if self.curr_turn_idx == 0:
            curr_trajs = [""] * self.num_processes
        else:
            curr_trajs, _ = self.memories[self.curr_traj_idx].fetch(
                obs_length=obs_length
            )
        past_trajs = [{} for _ in range(self.num_processes)]
        for t in range(self.curr_traj_idx):
            trajs, _ = self.memories[t].fetch(obs_length=obs_length)
            for i in range(self.num_processes):
                past_trajs[i][t] = trajs[i]

        for i in range(self.num_processes):
            prompt = get_tictactoe_prompt(
                board_size=self.config.env.tictactoe.board_size,
                phase=phase,
                turn_idx=self.curr_turn_idx,
                traj_idx=self.curr_traj_idx,
                init_observation=self.init_states[i],
                curr_traj=curr_trajs[i],
                past_traj=past_trajs[i],
                reflection=self.reflections[i],
            )
            post_obs.append(prompt)
        return post_obs
   
def make_envs(config):
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n must be int")

    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1

    env_kwargs = OmegaConf.to_container(
        config.env.tictactoe,
        resolve=True
    )
    val_env_kwargs = OmegaConf.to_container(
        config.env.tictactoe,
        resolve=True
    )
    val_env_kwargs['opponent_policy'] = 'mcts'
    val_env_kwargs['play_mode'] = config.env.tictactoe.get("val_play_mode", "first")

    train_envs = build_tictactoe_fixed_opponent_envs(
        seed=config.env.seed,
        env_num=config.data.train_batch_size,
        group_n=group_n,
        is_train=True,
        env_kwargs=env_kwargs,
    )

    print("=================val envs===============")
    val_envs = build_tictactoe_fixed_opponent_envs(
        seed=config.env.seed + 1000,
        env_num=config.data.val_batch_size,
        group_n=1,
        is_train=False,
        env_kwargs=val_env_kwargs,
    )

    num_attempts = config.env.get("num_attempts", 1)
    do_reflection = config.env.get("do_reflection", True)

    val_num_attempts = config.env.get("val_num_attempts", num_attempts)
    val_do_reflection = config.env.get("val_do_reflection", do_reflection)

    projection_f = partial(
        tictactoe_projection,
        board_size=config.env.tictactoe.board_size,
    )
    envs = TicTacToeEnvironmentManagerFixedOpponent(
        train_envs,
        projection_f,
        num_attempts,
        do_reflection,
        config,
    )
    val_envs = TicTacToeEnvironmentManagerFixedOpponent(
        val_envs,
        projection_f,
        val_num_attempts,
        val_do_reflection,
        config,
    )
    return envs, val_envs
