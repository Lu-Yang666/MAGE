from typing import List, Dict, Any

class SimpleMemoryKuhnPoker:
    def __init__(self, num_processes=0):
        self._data = [[] for _ in range(num_processes)]
        self.keys = None
        self.num_processes = num_processes

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def reset(self, num_processes: int):
        self._data = [[] for _ in range(num_processes)]
        self.num_processes = num_processes
        self.keys = None

    def store(self, record: Dict[str, List[Any]]):
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
    ):
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
            env_data = self._data[env_idx]
            env_won = False
            for d in env_data:
                w = d.get("won", False)
                env_won = env_won or bool(w)
                if env_won:
                    break
            final_wons.append(env_won)
        return final_wons
