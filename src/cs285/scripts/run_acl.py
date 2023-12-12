import os
import time
import yaml

from cs285.agents.soft_actor_critic import SoftActorCritic
from cs285.agents.teacher import TeacherPGAgent
from cs285.infrastructure.replay_buffer import ReplayBuffer
import cs285.env_configs

import os
import time

import gymnasium as gym
from gym import wrappers
import numpy as np
import torch
from cs285.infrastructure import pytorch_util as ptu
import tqdm

from cs285.infrastructure import acl_utils
from cs285.infrastructure.logger import Logger

from cs285.scripts.scripting_utils import make_logger, make_config

import argparse

CONVERT_OBS = lambda ob: np.hstack((ob['observation'], ob['desired_goal'], ob['achieved_goal']))

def generate_start_state(env_name):
    """
    Generate options, start_state from env_name
    """
    start_state = None
    options = None
    if env_name == "PointMaze_Open-v3":
        # PointMaze: {'goal_cell': numpy.ndarray, shape=(2,0), type=int, 'reset_cell': numpy.ndarray, shape=(2,0), type=int}
        goal_cell = np.array([np.random.choice([1, 2, 3]), np.random.choice([1, 2, 3, 4, 5])])
        reset_cell = np.array([np.random.choice([1, 2, 3]), np.random.choice([1, 2, 3, 4, 5])])
        options = {'goal_cell': goal_cell, 'reset_cell': reset_cell}
        start_state = np.hstack((goal_cell, reset_cell))
    elif env_name == "AntMaze_Open-v4":
        # AntMaze: {'goal_cell': numpy.ndarray, shape=(2,0), type=int, 'reset_cell': numpy.ndarray, shape=(2,0), type=int}
        goal_cell = np.array([np.random.choice([1, 2, 3]), np.random.choice([1, 2, 3, 4, 5])])
        reset_cell = np.array([np.random.choice([1, 2, 3]), np.random.choice([1, 2, 3, 4, 5])])
        options = {'goal_cell': goal_cell, 'reset_cell': reset_cell}
        start_state = np.hstack((goal_cell, reset_cell))
    elif env_name == "AdroitHandHammer-v1":
        # AdroitHammer: {'qpos': numpy.ndarray, shape=(33,), 'qvel': numpy.ndarray, shape=(33,), 'board_pos': numpy.ndarray, shape=(3,)}
        qpos = np.zeros(33)
        qvel = np.zeros(33)
        board_pos = np.array([np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2), np.random.uniform(0.1, 0.25)])
        options = {'qpos': qpos, 'qvel': qvel, 'board_pos': board_pos}
        start_state = np.hstack((qpos, qvel, board_pos))
    elif env_name == "AdroitHandRelocate-v1":
        # AdroitRelocate: {'qpos': numpy.ndarray, shape=(36,), 'qvel': numpy.ndarray, shape=(36,), 'obj_pos': numpy.ndarray, shape=(3,), 'target_pos': numpy.ndarray, shape=(3,)}
        qpos = np.zeros(36)
        qvel = np.zeros(36)
        obj_pos = np.array([np.random.uniform(-0.15, 0.15), np.random.uniform(-0.15, 0.30), 0])
        target_pos = np.array([np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2), np.random.uniform(0.15, 0.35)])
        options = {'qpos': qpos, 'qvel': qvel, 'obj_pos': obj_pos, 'targer_pos': target_pos}
        start_state = np.hstack((qpos, qvel, obj_pos, target_pos))

    return options, start_state


def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    env = config["make_env"]()
    eval_env = config["make_env"]()
    render_env = config["make_env"](render=True)
    # print(isinstance(env.observation_space, gym.spaces.Box))
    # print(isinstance(eval_env.observation_space, gym.spaces.Box))
    # print(isinstance(render_env.observation_space, gym.spaces.Box))

    ep_len = config["ep_len"] or env.spec.max_episode_steps
    batch_size = config["batch_size"] or batch_size

    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    assert (
        not discrete
    ), "Our actor-critic implementation only supports continuous action spaces. (This isn't a fundamental limitation, just a current implementation decision.)"

    if isinstance(env.observation_space, gym.spaces.Box):
        ob_shape = env.observation_space.shape
    else:
        ob_shape = (env.observation_space['achieved_goal'].shape[0] + env.observation_space['desired_goal'].shape[0] + env.observation_space['observation'].shape[0],)
    ac_dim = env.action_space.shape[0]

    # simulation timestep, will be used for video saving
    if "model" in dir(env):
        fps = 1 / env.model.opt.timestep
    else:
        fps = env.env.metadata["render_fps"]

    # initialize agent
    agent = SoftActorCritic(
        ob_shape,
        ac_dim,
        **config["agent_kwargs"],
    )

    env_name = env.unwrapped.spec.id
    assert env_name in ["PointMaze_Open-v3", "AntMaze_Open-v4", "AdroitHandHammer-v1", "AdroitHandRelocate-v1"]

    teacher_ob_dim = None
    teacher_ac_dim = None

    if env_name == "PointMaze_Open-v3":
        # PointMaze: {'goal_cell': numpy.ndarray, shape=(2,0), type=int, 'reset_cell': numpy.ndarray, shape=(2,0), type=int}
        teacher_ac_dim = 2 + 2
        teacher_ob_dim = teacher_ac_dim + 1
    elif env_name == "AntMaze_Open-v4":
        # AntMaze: {'goal_cell': numpy.ndarray, shape=(2,0), type=int, 'reset_cell': numpy.ndarray, shape=(2,0), type=int}
        teacher_ac_dim = 2 + 2
        teacher_ob_dim = teacher_ac_dim + 1
    elif env_name == "AdroitHandHammer-v1":
        # AdroitHammer: {'qpos': numpy.ndarray, shape=(33,), 'qvel': numpy.ndarray, shape=(33,), 'board_pos': numpy.ndarray, shape=(3,)}
        teacher_ac_dim = 33 + 33 + 3
        teacher_ob_dim = teacher_ac_dim + 1
    elif env_name == "AdroitHandRelocate-v1":
        # AdroitRelocate: {'qpos': numpy.ndarray, shape=(36,), 'qvel': numpy.ndarray, shape=(36,), 'obj_pos': numpy.ndarray, shape=(3,), 'target_pos': numpy.ndarray, shape=(3,)}
        teacher_ac_dim = 36 + 36 + 3 + 3
        teacher_ob_dim = teacher_ac_dim + 1

    # IDEA: observation is the given task as well as the return achieved from the starting state
    teacher = TeacherPGAgent(
        teacher_ob_dim,
        teacher_ac_dim,
        **config["teacher_kwargs"]
    )

    replay_buffer = ReplayBuffer(config["replay_buffer_capacity"])

    ep_steps = 0
    if isinstance(env.observation_space, gym.spaces.Box):
        observation = env.reset()[0]
    else:
        observation = CONVERT_OBS(env.reset()[0])

    for step in tqdm.trange(config["total_steps"], dynamic_ncols=True):
        ep_steps += 1
        if step < config["random_steps"]:
            action = env.action_space.sample()
        else:
            action = agent.get_action(observation)

        # Step the environment and add the data to the replay buffer
        next_observation, reward, done, _, info = env.step(action)
        if not isinstance(env.observation_space, gym.spaces.Box):
            next_observation = CONVERT_OBS(next_observation)
        truncated = ep_steps > ep_len
        replay_buffer.insert(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=done and not info.get("TimeLimit.truncated", False),
        )

        if done or truncated:
            # logger.log_scalar(info["episode"]["r"], "train_return", step)
            # logger.log_scalar(info["episode"]["l"], "train_ep_len", step)

            options = None

            # if ready, update the teacher, generate a new starting state

            # PointMaze: {'goal_cell': numpy.ndarray, shape=(2,0), type=int, 'reset_cell': numpy.ndarray, shape=(2,0), type=int}
            # AntMaze: {'goal_cell': numpy.ndarray, shape=(2,0), type=int, 'reset_cell': numpy.ndarray, shape=(2,0), type=int}
            # AdroitHammer: {'qpos': numpy.ndarray, shape=(33,), 'qvel': numpy.ndarray, shape=(33,), 'board_pos': numpy.ndarray, shape=(3,)}
            # AdroitRelocate: {'qpos': numpy.ndarray, shape=(36,), 'qvel': numpy.ndarray, shape=(36,), 'obj_pos': numpy.ndarray, shape=(3,), 'target_pos': numpy.ndarray, shape=(3,)}

            options, start_state = generate_start_state(env_name)
            # print(options)

            ep_steps = 0
            if isinstance(env.observation_space, gym.spaces.Box):
                observation = env.reset()[0] if not options else env.reset(options=options)[0]
            else:
                observation = CONVERT_OBS(env.reset()[0]) if not options else CONVERT_OBS(env.reset(options=options)[0])
        else:
            observation = next_observation

        # Train the agent
        if step >= config["training_starts"]:
            batch = replay_buffer.sample(config['batch_size'])
            update_info = agent.update(batch["observations"], batch["actions"], batch["rewards"], batch["next_observations"], batch["dones"], step)

            # Logging
            update_info["actor_lr"] = agent.actor_lr_scheduler.get_last_lr()[0]
            update_info["critic_lr"] = agent.critic_lr_scheduler.get_last_lr()[0]

            if step % args.log_interval == 0:
                for k, v in update_info.items():
                    logger.log_scalar(v, k, step)
                    logger.log_scalars
                logger.flush()

        # Run evaluation
        if step % args.eval_interval == 0:
            trajectories = acl_utils.sample_n_trajectories(
                eval_env,
                policy=agent,
                ntraj=args.num_eval_trajectories,
                max_length=ep_len,
            )
            returns = [t["episode_statistics"]["r"] for t in trajectories]
            ep_lens = [t["episode_statistics"]["l"] for t in trajectories]

            logger.log_scalar(np.mean(returns), "eval_return", step)
            logger.log_scalar(np.mean(ep_lens), "eval_ep_len", step)

            if len(returns) > 1:
                logger.log_scalar(np.std(returns), "eval/return_std", step)
                logger.log_scalar(np.max(returns), "eval/return_max", step)
                logger.log_scalar(np.min(returns), "eval/return_min", step)
                logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", step)
                logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", step)
                logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", step)

            if args.num_render_trajectories > 0:
                video_trajectories = acl_utils.sample_n_trajectories(
                    render_env,
                    agent,
                    args.num_render_trajectories,
                    ep_len,
                    render=True,
                )

                logger.log_paths_as_videos(
                    video_trajectories,
                    step,
                    fps=fps,
                    max_videos_to_save=args.num_render_trajectories,
                    video_title="eval_rollouts",
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)

    parser.add_argument("--eval_interval", "-ei", type=int, default=5000)
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=10)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-g", default=0)
    parser.add_argument("--log_interval", type=int, default=1000)

    args = parser.parse_args()

    # create directory for logging
    logdir_prefix = "acl_"

    config = make_config(args.config_file)
    logger = make_logger(logdir_prefix, config)

    run_training_loop(config, logger, args)


if __name__ == "__main__":
    main()