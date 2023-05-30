import io
import os
import random
import warnings
from collections import defaultdict, Counter
from copy import deepcopy
from tqdm import tqdm, trange

from .mcts import MCTS, MCTSNode
from .models import QueryLM


def is_terminal_prompt_or_action(prompt):
    prompt = prompt.split('\n\n')[-1]
    if 'finish' in prompt.lower():
        return True
    return False


class ReasoningMCTSNode(MCTSNode):
    @property
    def visited(self):
        return self._visited

    def __init__(self, state, gen_fn, reward_fn, depth, r1, max_depth,
                 parent: 'ReasoningMCTSNode' = None, action=None):
        self._conf = None
        self.children = []
        self.state = state
        self.action = action
        self.gen_fn = gen_fn
        self.reward_fn = reward_fn
        self.depth = depth
        self.max_depth = max_depth
        self._r1 = r1
        if action == 'finish':
            self._r1 += 0.01
        self._ans_list = None
        self._visited = True # we do not need to visit again for ProntoQA MCTS settings
        self.parent = parent
        self._terminal = False

    def _child_node(self, action, next_state, r1):
        return ReasoningMCTSNode(next_state, self.gen_fn, self.reward_fn, self.depth + 1,
                                 r1, self.max_depth, parent=self, action=action)

    def _get_children(self):
        self._visited = True
        self._calculate_reward()
        if self.is_terminal or self._r1 <= -1 or self.depth == self.max_depth:
            return self.children
        for action, next_state, reward in self.gen_fn(self.state):
            self.children.append(self._child_node(action, next_state, reward))
        return self.children

    def find_children(self):
        self.children = self.children or self._get_children()
        return self.children

    def find_one_child(self) -> MCTSNode:
        return random.choice(self.find_children())

    def _calculate_reward(self):
        return self._r1

    @property
    def is_terminal(self):
        return self.action is not None and is_terminal_prompt_or_action(self.action)

    @property
    def reward(self):
        return self._r1

    def print(self, mcts: MCTS, file=None):
        def pprint(*args):
            if file is None:
                tqdm.write(*args)
            else:
                print(*args, file=file)
        p1 = '-' * (4 * self.depth - 4)
        prefix = ' ' * (4 * self.depth - 4)
        if self.action is None:
            action = f'S.{self.depth}: {self.state}'
        else:
            action = f'A.{self.depth}: {self.action}'
        pprint(p1 + action)
        pprint(prefix + f'R: {self.reward:.3f} ; N: {mcts.N[self]} ; M: {mcts.M[self]:.3f}')
        if self.action is not None and self.state is not None:
            term = '\u25A1' if self.is_terminal else ''
            state = f'S.{self.depth}: {self.state} ; r1: {self._r1:.3f} {term}'
            pprint(prefix + state)
        for child in self.children:
            child.print(mcts, file)
        if self.depth == 1:
            pprint("=" * 12)

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.gen_fn is None or self.reward_fn is None:
            warnings.warn('MCTSNode loaded from pickle is read-only; Do not further roll out the tree!')

    def __getstate__(self):
        state = self.__dict__.copy()
        state['gen_fn'] = None
        state['reward_fn'] = None
        return state


def reasoning_mcts_search(facts: str,
                          target: str,
                          agent_prompts,
                          wm_transit_prompts,
                          wm_output_prompts,
                          wm_finish_prompts,
                          wm_valid_prompts,
                          world_model: QueryLM,
                          n_sample_subquestion,
                          temperature,
                          mcts_rollouts,
                          w_exp,
                          max_depth,
                          eos_token_id,
                          logging=False):
    if logging:
        os.path.exists('agent.log') and os.remove('agent.log')
        os.path.exists('wm.log') and os.remove('wm.log')

    base_facts = facts[:facts.rindex('. ') + 1]
    init_state = facts.split('. ')[-1]

    next_action_state_cache = {}

    def gen_fn(state):
        if state in next_action_state_cache:
            return next_action_state_cache[state]

        prompt_examples = agent_prompts["input"]
        fact_input = agent_prompts["facts_format"].format(base_facts)
        target_input = agent_prompts["target_format"].format(target)
        claim_input = agent_prompts["claim_format"].format(state)
        output_prefix = agent_prompts["next_step_prefix"]
        agent_input = prompt_examples + fact_input + target_input + claim_input + output_prefix
        agent_output = world_model.query_LM(agent_input, do_sample=True, num_return_sequences=n_sample_subquestion,
                                            eos_token_id=eos_token_id, temperature=temperature)

        agent_output = [o.strip().split(output_prefix + ' ')[-1] for o in agent_output]
        agent_output_counter = Counter(agent_output)

        if logging:
            with open('agent.log', 'a') as f:
                print(agent_input[len(prompt_examples):], file=f)
                for o in agent_output:
                    print(f'{o}', file=f)
                print('=' * 20, file=f)

        next_state_dict = defaultdict(lambda: [])

        for action in sorted(agent_output_counter):
            next_state = transit_fn(state, action)
            next_state_dict[next_state].append((action, agent_output_counter[action]))

        ret_actions, ret_next_states = [], []
        for next_state, actions in next_state_dict.items():
            ret_actions.append(max(actions, key=lambda x: x[1])[0])
            ret_next_states.append(next_state)

        rewards = reward_fn(state, ret_actions, ret_next_states) if len(ret_actions) else []
        ret = list(zip(ret_actions, ret_next_states, rewards))
        next_action_state_cache[state] = ret
        return ret

    def transit_fn(state, action):
        if action == 'Finish.':
            prompt_examples = wm_output_prompts["input"]
            target_input = wm_output_prompts["target_format"].format(target)
            claim_input = wm_output_prompts["claim_format"].format(state)
            output_prefix = wm_output_prompts["output_prefix"]
            world_input = prompt_examples + target_input + claim_input + output_prefix
        else:
            prompt_examples = wm_transit_prompts["input"]
            facts_input = wm_transit_prompts["facts_format"].format(state, action)
            output_prefix = wm_transit_prompts["next_claim_prefix"]
            world_input = prompt_examples + facts_input + output_prefix

        world_output = world_model.query_LM(world_input, do_sample=False, num_return_sequences=1,
                                            eos_token_id=eos_token_id)[0]
        output = world_output.strip().split(output_prefix + " ")[-1]
        result = output.split(output_prefix + " ")[-1]

        if logging:
            with open('wm_transit.log', 'a') as f:
                print(world_input[len(prompt_examples):], file=f)
                print(world_output[len(prompt_examples):], file=f)
                print('='*20, file=f)
        return result

    def reward_fn(state, actions, next_states):
        world_inputs = []

        for action, next_state in zip(actions, next_states):
            if action == 'Finish.':
                prompt_examples = wm_finish_prompts["input"]
                target_input = wm_finish_prompts["target_format"].format(target)
                claim_input = wm_finish_prompts["claim_format"].format(state)
                output_prefix = wm_finish_prompts["output_prefix"]
                world_input = prompt_examples + target_input + claim_input + output_prefix
            else:
                prompt_examples = wm_valid_prompts["input"]
                facts_input = wm_valid_prompts["facts_format"].format(state, action)
                next_input = wm_valid_prompts["next_step_format"].format(next_state)
                output_prefix = wm_valid_prompts["valid_prefix"]
                world_input = prompt_examples + facts_input + next_input + output_prefix
            world_inputs.append(world_input)

        world_outputs = world_model.query_next_token(world_inputs)
        rewards = world_outputs[:, 0]
        rewards = rewards.detach().cpu().numpy().tolist()

        if logging:
            with open('wm_reward.log', 'a') as f:
                for world_input, reward in zip(world_inputs, rewards):
                    print(world_input.split('\n\n')[-1], file=f)
                    print(reward, file=f)
                    print('='*20, file=f)
        return rewards

    mcts = MCTS(w_exp=w_exp, prior=True, aggr_reward='mean', aggr_child='max')
    root = ReasoningMCTSNode(init_state, gen_fn, None,
                             depth=1, r1=1, max_depth=max_depth, parent=None)
    trajs = []
    outputs = []
    trees = []
    for _ in (pbar := trange(mcts_rollouts, disable=bool(int(os.environ.get("LOCAL_RANK", -1))), position=0)):
        mcts.rollout(root)
        root.print(mcts)
        max_n, max_r = mcts.max_mean_terminal(root)
        cur = max_n.parent
        traj = []
        if cur is not None:
            while cur != root:
                traj.append(cur.state)
                traj.append(cur.action)
                cur = cur.parent
            traj.append(cur.state)
            traj = list(reversed(traj))
        trajs.append(traj)
        for i in ['true', 'false']:
            if i in max_n.state.lower():
                temp_r = i
                break
        else:
            temp_r = 'none'
        outputs.append(temp_r)
        pbar.set_description(f'{max_r:.3f} {temp_r}')
        tree_copy = deepcopy(root)
        tree_copy.Q = dict(mcts.Q)
        tree_copy.N = dict(mcts.N)
        tree_copy.M = dict(mcts.M)
        trees.append(tree_copy)

    with io.StringIO() as f:
        root.print(mcts, file=f)
        tree = f.getvalue()
    return trajs, tree, trees, outputs
