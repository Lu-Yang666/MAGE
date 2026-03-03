# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import List, Tuple, Dict, Union, Any
import torch
import numpy as np
import os
from collections import defaultdict


def to_numpy(data):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        pass
    elif isinstance(data, (int, float, bool, Tuple, List)):
        data = np.array(data)
    else:
        raise ValueError(f"Unsupported type: {type(data)})")
    return data


class EnvironmentManagerBase:
    def __init__(self, envs, projection_f, config):
        """
        Initialize the environment manager.
       
        Parameters:
        - envs: The environment instance, usually a vectorized environment containing multiple sub-environments.
        - projection_f: A function that maps text actions to environment actions.
        - config: Configuration object.
        """
        self.envs = envs
        self.projection_f = projection_f
        self.config = config


    def reset(self) -> Dict[str, Any]:
        """
        Reset all environments and return the initial observations.
       
        Returns:
        - next_observations (Dict):
          - 'text' (None or List[str]): The textual observation.
          - 'image' (np.ndarray or torch.Tensor): The image observation as either a NumPy array or a PyTorch tensor.
          - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        """
        obs, infos = self.envs.reset()
        return {'text': None, 'image': obs, 'anchor': None}, infos
   
    def step(self, text_actions: List[str]):
        """
        Execute text actions and return the next state, rewards, done flags, and additional information.
       
        Parameters:
        - text_actions (List[str]): A list of text actions to execute.
       
        Returns:
        - next_observations (Dict):
          - 'text' (None or List[str]): The textual observation.
          - 'image' (np.ndarray or torch.Tensor): The image observation as either a NumPy array or a PyTorch tensor.
          - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        - rewards (np.ndarry or torch.Tensor): The rewards returned by the environment.
        - dones (np.ndarray or torch.Tensor): Done flags indicating which environments have completed.
        - infos (List[Dict]): Additional environment information.
       
        Exceptions:
        - NotImplementedError: If an observation key is not in ('text', 'image').
        """
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)


        next_observations = {
            'text': None, # Implement this if needed
            'image': next_obs,
            'anchor': None # For GiGPO only. anchor observation without any histories, hint, etc. Implement this if needed
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])


        rewards = to_numpy(rewards)
        dones = to_numpy(dones)
       
        return next_observations, rewards, dones, infos


    def build_text_obs(self,) -> List[str]:
        """
        This function builds the text observation for the agent.
       
        Returns:
        - postprocess_text_obs (List[str]): A list of processed text observations.
        """
        pass


    def close(self) -> None:
        """
        Close the environment and release resources.
        """
        self.envs.close()


    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """
        Evaluate if the episodes are successful or not.
        (Default) implementation is to check info['won'] of the last step.
       
        Returns:
        - success (np.ndarray or torch.Tensor): 1 if the episode is successful, 0 otherwise.
        """
        total_infos = kwargs['total_infos']
        total_batch_list = kwargs['total_batch_list']
        batch_size = len(total_batch_list)
       
        success = defaultdict(list)
       
        for bs in range(batch_size):
            # self._process_batch(bs, total_batch_list, total_infos, success)
            wons = [False for _ in range(self.num_attempts)]
            draws = [False for _ in range(self.num_attempts)]
            for i in reversed(range(len(total_batch_list[bs]))):  
                batch_item = total_batch_list[bs][i]
                if batch_item['active_masks']:
                    info = total_infos[bs][i]
                    traj_idx = batch_item['traj_idx']
                    if batch_item['phase'] == 'play':
                        wons[traj_idx] = wons[traj_idx] or info['won']
                        draws[traj_idx] = draws[traj_idx] or info.get('draw', False)  


            _won = False      
            _draw = False      
            for traj_idx in range(self.num_attempts):    
                _won = _won or wons[traj_idx]
                _draw = _draw or draws[traj_idx]
                _sep_won = False or wons[traj_idx]
                _sep_draw = False or draws[traj_idx]
                success[f'success_rate[{traj_idx}]'].append(_won)
                success[f'draw_rate[{traj_idx}]'].append(_draw)
                success[f'success&draw_rate[{traj_idx}]'].append(_won or _draw)
                success[f'independent_success_rate[{traj_idx}]'].append(_sep_won)
                success[f'independent_draw_rate[{traj_idx}]'].append(_sep_draw)
                success[f'independent_success&draw_rate[{traj_idx}]'].append(_sep_won or _sep_draw)
        return {key: np.array(value) for key, value in success.items()}
   
    def success_evaluator_opp(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """
        Evaluate if the episodes are successful or not.
        (Default) implementation is to check info['won'] of the last step.
       
        Returns:
        - success (np.ndarray or torch.Tensor): 1 if the episode is successful, 0 otherwise.
        """
        total_infos = kwargs['total_infos']
        total_batch_list = kwargs['total_batch_list']
        batch_size = len(total_batch_list)
       
        opponentslist = []
        for bs, infos in enumerate(total_infos):
            opponent = None
            opponents = set()


            for i, info in enumerate(infos):
                if 'opponent' in info:
                    opponents.add(info['opponent'])
            assert len(opponents) <= 1, (
                f'[ERROR][success_evaluator_opp] '
                f'Multiple opponents in total_infos[{bs}]: {opponents}'
            )
            if len(opponents) == 1:
                opponent = next(iter(opponents))
            if opponent is None:
                print(f'[DEBUG][success_evaluator_opp] No opponent found in total_infos[{bs}]')
                for i, info in enumerate(infos):
                    print(f'  step {i}: keys = {list(info.keys())}')
                continue
            for i, info in enumerate(infos):
                if 'opponent' not in info:
                    # print(f"lack opponent! info: {info}")
                    info['opponent'] = opponent
            opponentslist.append(opponent)
        # print(f"opponents: {len(opponentslist)}, {opponentslist}")


        success = defaultdict(list)  
       
        for bs in range(batch_size):
            # Initialize opponent_stats for each batch
            opponent_stats = defaultdict(
                lambda: {
                    'wons':  [False] * self.num_attempts,
                    'draws': [False] * self.num_attempts,
                    'scores': [-2.0] * self.num_attempts,
                }
            )
           
            pre_traj_idx = 10
            # Process each batch
            for i in reversed(range(len(total_batch_list[bs]))):  
                batch_item = total_batch_list[bs][i]
                if batch_item['active_masks']:
                    info = total_infos[bs][i]
                    traj_idx = batch_item['traj_idx']
                    opponent = info['opponent']
                   
                    if batch_item['phase'] == 'play':
                        opponent_stats[opponent]['wons'][traj_idx] = \
                            opponent_stats[opponent]['wons'][traj_idx] or info['won']
                        opponent_stats[opponent]['draws'][traj_idx] = \
                            opponent_stats[opponent]['draws'][traj_idx] or info.get('draw', False)
                        if traj_idx != pre_traj_idx:
                            print(f"pre_traj_idx: {pre_traj_idx}, bs {bs} traj {traj_idx} final score: {info['score']}")
                            opponent_stats[opponent]['scores'][traj_idx] =  info['score']                          
                    pre_traj_idx=traj_idx
            # print(f"opponent_stats: {opponent_stats}")
            # After processing the batch, calculate success and draw rates by opponent and traj_idx
            for opponent, stats in opponent_stats.items():
                _won = False
                _draw = False
                _score = -2.0
                for traj_idx in range(self.num_attempts):
                    _sep_won = False or stats['wons'][traj_idx]
                    _sep_draw = False or stats['draws'][traj_idx]  
                    # Aggregate the results for this opponent and traj_idx
                    _won = _won or stats['wons'][traj_idx]
                    _draw = _draw or stats['draws'][traj_idx]  
                    _score = max(_score, stats['scores'][traj_idx])                
                    # Store results in the success dictionary
                    success[f'score[{traj_idx}]'].append(_score)
                    print(f"==========scores! {success[f'score[{traj_idx}]']}")
                    success[f'success_rate[{traj_idx}]'].append(_won)
                    success[f'draw_rate[{traj_idx}]'].append(_draw)
                    success[f'success&draw_rate[{traj_idx}]'].append(_won or _draw)    
                    # success[f'independent_success_rate[{traj_idx}]'].append(_sep_won)
                    # success[f'independent_draw_rate[{traj_idx}]'].append(_sep_draw)
                    # success[f'independent_success&draw_rate[{traj_idx}]'].append(_sep_won or _sep_draw)
                    success[f'{opponent}_success_rate[{traj_idx}]'].append(_won)
                    success[f'{opponent}_draw_rate[{traj_idx}]'].append(_draw)
                    success[f'{opponent}_success&draw_rate[{traj_idx}]'].append(_won or _draw)
                    # success[f'{opponent}_independent_success_rate[{traj_idx}]'].append(_sep_won)
                    # success[f'{opponent}_independent_draw_rate[{traj_idx}]'].append(_sep_draw)
                    # success[f'{opponent}_independent_success&draw_rate[{traj_idx}]'].append(_sep_won or _sep_draw)
            # print(f"success: {success}")
        # Return results as numpy arrays
        return {key: np.array(value) for key, value in success.items()}


    def success_evaluator_multi_player(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """
        Evaluate if the episodes are successful or not.
        (Default) implementation is to check info['won'] of the last step.
       
        Returns:
        - success (np.ndarray or torch.Tensor): 1 if the episode is successful, 0 otherwise.
        """
        total_infos = kwargs['total_infos']
        total_batch_list = kwargs['total_batch_list']
        player_id = kwargs['player_id']
        batch_size = len(total_batch_list)
       
        success = defaultdict(list)
       
        for bs in range(batch_size):
            # self._process_batch(bs, total_batch_list, total_infos, success)
            wons = [False for _ in range(self.num_attempts)]
            draws = [False for _ in range(self.num_attempts)]
            for i in reversed(range(len(total_batch_list[bs]))):  
                batch_item = total_batch_list[bs][i]
                traj_idx = batch_item['traj_idx']
                if batch_item['active_masks']:
                    info = total_infos[bs][i]
                    if batch_item['phase'] == 'play':
                        wons[traj_idx] = wons[traj_idx] or info['won']
                        draws[traj_idx] = draws[traj_idx] or info.get('draw', False)  


            _won = False      
            _draw = False      
            for traj_idx in range(self.num_attempts):  
                _non_accumulate_won = False
                _non_accumulate_draw = False
                _won = _won or wons[traj_idx]
                _draw = _draw or draws[traj_idx]
                _non_accumulate_won = _non_accumulate_won or wons[traj_idx]
                _non_accumulate_draw = _non_accumulate_draw or draws[traj_idx]
                success[f'player{player_id}_accumulate_success_rate[{traj_idx}]'].append(_won)
                success[f'player{player_id}_accumulate_draw_rate[{traj_idx}]'].append(_draw)
                success[f'player{player_id}_accumulate_success&draw_rate[{traj_idx}]'].append(_won or _draw)
                success[f'player{player_id}_success_rate[{traj_idx}]'].append(_non_accumulate_won)
                success[f'player{player_id}_draw_rate[{traj_idx}]'].append(_non_accumulate_draw)
                success[f'player{player_id}_success&draw_rate[{traj_idx}]'].append(_non_accumulate_won or _non_accumulate_draw)
       
        return {key: np.array(value) for key, value in success.items()}
       
    # def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
    #     for i in reversed(range(len(total_batch_list[batch_idx]))):
    #         wons = [False for _ in range(self.num_attempts)]  
    #         batch_item = total_batch_list[batch_idx][i]
    #         if batch_item['active_masks']:
    #             info = total_infos[batch_idx][i]
    #             traj_idx = batch_item['traj_idx']
    #             if batch_item['phase'] == 'play':
    #                 wons[traj_idx] = wons[traj_idx] or info['won']


    #     for traj_idx, won in enumerate(wons):      
    #         success[f'success_rate[{traj_idx}]'].append(won)
    #     return
           
    def save_image(self, image, step):
        """
        Save an image to a file.
       
        Parameters:
        - image (np.ndarray or torch.Tensor): The image to save.
        - path (str): The path to save the image.
        """
        path = os.path.join(os.path.dirname(__file__), os.path.join("images", self.config.env.env_name))
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, f"step{step}.png")
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if isinstance(image, np.ndarray):
            pass
        else:
            raise ValueError(f"Unsupported type: {type(image)})")
       
        if len(image.shape) == 4:
            image = image[0]
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        if image.max() <= 1.0:
            image = (image * 255)


        image = image.astype(np.uint8)
       
        from PIL import Image
        image = Image.fromarray(image)
        image.save(path)

