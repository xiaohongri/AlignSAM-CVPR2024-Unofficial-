# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import mlflow
import yaml

import custom_gym_envs


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    mlflow_project_name: str = "sam_align"
    """the MLFlow's project name"""
    mlflow_user: str = None
    """the username (team) of MLFlow's project"""
    log_dir: str = "runs"
    """the logging directory for the experiment"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `{log_dir}/{run_id}/videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "SamSegEnv-v0"
    """the id of the environment"""
    env_cfg_path: str = "configs/envs/repvit_sam_cbseg100_only_rug.yaml"
    """the environment configuration path"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 16
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, log_dir, env_cfg):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, **env_cfg)
            video_folder = os.path.join(log_dir, "videos")
            env = gym.wrappers.RecordVideo(env, video_folder)
        else:
            env = gym.make(env_id, **env_cfg)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(256, 64, 8, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 128, 5, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(128, 256, 3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(256, 512, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(512 * 4 * 4, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def parse_obs(self, obs):
        sam_image_embeddings = obs["sam_image_embeddings"] # (b, c, h, w)
        sam_pred_mask_prob = obs["sam_pred_mask_prob"].unsqueeze(dim=1)  # (b, 1, h, w)

        embedding_shape = tuple(sam_image_embeddings.size())
        resized_sam_mask_prob = nn.functional.interpolate(
            sam_pred_mask_prob, size=(embedding_shape[2], embedding_shape[3]), 
            mode="bilinear", align_corners=False)
        resized_sam_mask_prob = resized_sam_mask_prob.repeat(1, embedding_shape[1], 1, 1)

        x = sam_image_embeddings * resized_sam_mask_prob 
        x += sam_image_embeddings # skip connection
        return x

    def get_value(self, obs):
        x = self.parse_obs(obs)
        hidden = self.network(x)
        return self.critic(hidden)

    def get_action_and_value(self, obs, action=None):
        x = self.parse_obs(obs)
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


def load_obs_to_tensor(obs, device):
    obs_tensor = dict()
    for key in obs.keys():
        obs_tensor[key] = torch.Tensor(obs[key]).to(device)
    return obs_tensor

def update_obs_at_step(obs, next_obs, step):
    for key in obs.keys():
        obs[key][step] = next_obs[key]

def flatten_obs(obs):
    new_obs = dict()
    for key, val in obs.items():
        val_shape = list(val.size())
        num_steps, num_envs = val_shape[:2] 
        new_obs[key] = val.reshape(num_steps * num_envs, *val_shape[2:])
    return new_obs

def get_obs_at_inds(obs, mb_inds):
    new_obs = dict()
    for key, val in obs.items():
        new_obs[key] = val[mb_inds]
    return new_obs


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if not os.path.exists(args.env_cfg_path):
        raise ValueError(f"env_cfg_path {args.env_cfg_path} does not exist")
    
    env_cfg = yaml.safe_load(open(args.env_cfg_path, "r"))

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    mlflow_run_tags = {
        "user": args.mlflow_user,
        "project": args.mlflow_project_name,
    }

    with mlflow.start_run(run_name=run_name, tags=mlflow_run_tags, description="parent") as parent_run:
        log_dir = os.path.join(args.log_dir.rstrip("/"), run_name)
        writer = SummaryWriter(log_dir)
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        for key, value in vars(args).items():
            mlflow.log_param(key, value)

        # env setup
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, i, args.capture_video, log_dir, env_cfg) for i in range(args.num_envs)],
        )
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        agent = Agent(envs).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

        # ALGO Logic: Storage setup
        obs = dict()
        # obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
        for key, space in envs.single_observation_space.spaces.items():
            obs[key] = torch.zeros((args.num_steps, args.num_envs) + space.shape).to(device)
        actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        raw_next_obs, _ = envs.reset(seed=args.seed)
        next_obs = load_obs_to_tensor(raw_next_obs, device)
        # next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(args.num_envs).to(device)

        for iteration in range(1, args.num_iterations + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - ((iteration - 1.0) / args.num_iterations)
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
                global_step += args.num_envs
                update_obs_at_step(obs, next_obs, step)
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_raw_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs = load_obs_to_tensor(next_raw_obs, device)
                next_done = torch.Tensor(next_done).to(device)

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            # b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_obs = flatten_obs(obs)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    mb_obs = get_obs_at_inds(b_obs, mb_inds)

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, 
                                                                                  b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            mlflow.log_metric("losses/value_loss", v_loss.item(), global_step)
            mlflow.log_metric("losses/policy_loss", pg_loss.item(), global_step)
            mlflow.log_metric("losses/entropy", entropy_loss.item(), global_step)


            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        envs.close()
        writer.close()

        print("Uploading TensorBoard events as a run artifact...")
        mlflow.log_artifacts(log_dir, artifact_path="events")
        print(
            "\nLaunch TensorBoard with:\n\ntensorboard --logdir=%s"
            % os.path.join(mlflow.get_artifact_uri(), "events")
        )