from typing import List, Dict, Any

class SimpleMemoryTicTacToe:
    """
    Memory manager for Tic-Tac-Toe:
    Stores per-environment interaction history (observations + actions)
    and provides formatted history for prompts.
    """
    def __init__(self, num_processes=0):
        self._data = [[] for _ in range(num_processes)]
        self.keys = None
        self.num_processes = num_processes

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def reset(self, num_processes: int):
        """Reset memory for all environments."""
        self._data = [[] for _ in range(num_processes)]
        self.num_processes = num_processes
        self.keys = None

    def store(self, record: Dict[str, List[Any]]):
        """
        Store a new record (one step of history) for each environment instance.
        Args:
            record (Dict[str, List[Any]]):
                Each key corresponds to a type of data (e.g., 'text_obs', 'action', 'reward')
                Each value is a list of length `num_processes`, one element per environment
        """
        if self.keys is None:
            self.keys = list(record.keys())
        assert self.keys == list(record.keys()), "Keys in record do not match memory keys"
        for env_idx in range(self.num_processes):
            self._data[env_idx].append({k: record[k][env_idx] for k in self.keys})

    def fetch(
        self,
        history_length: int = 7,
        obs_key: str = "text_obs",
        action_key: str = "action",
        obs_length: int = 2,
    ) -> List[List[Any]]:
        """
        Fetch and format recent history for each environment.
       
        Args:
            history_length: Max number of past steps to retrieve
            obs_key: Key for observation in stored record
            action_key: Key for action in stored record
            obs_length: How many steps to show full observation (others show '...')
       
        Returns:
            memory_contexts: List[str], formatted text history for each environment
            valid_lengths: List[int], number of valid steps per environment
        """
        memory_contexts, valid_lengths = [], []
        for env_idx in range(self.num_processes):
            recent = self._data[env_idx][-history_length:]
            valid_len = len(recent)
            start_idx = len(self._data[env_idx]) - valid_len
            lines = []
            for j, rec in enumerate(recent):
                step_num = start_idx + j + 1
                act = rec[action_key]
                obs = rec[obs_key]
                if len(recent) - j > obs_length:
                    lines.append(f"Step {step_num} - Action: {act}")
                else:
                    lines.append(f"Step {step_num} - Action: {act}\nObservation:\n{obs}")
                if 'dones' in rec.keys() and rec['dones']:
                    valid_len = step_num
                    break

            memory_contexts.append("\n".join(lines))
            valid_lengths.append(valid_len)
        return memory_contexts, valid_lengths

    def get_wins(self):
        final_wons = []
        for env_idx in range(self.num_processes):
            env_data = self._data[env_idx]   # list[dict]
            env_won = False
            for d in env_data:
                w = d.get("won", False)
                env_won = env_won or bool(w)
                if env_won:
                    break
            final_wons.append(env_won)
        return final_wons

