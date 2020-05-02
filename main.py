import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
import datetime
import numpy as np
import os
import random
import shutil
import atari_py
import argparse
import time
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.environments.wrappers import PyEnvironmentBaseWrapper
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network, q_rnn_network
from tf_agents.policies import random_tf_policy, policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.specs import array_spec
import base64
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

AUTOTUNE = tf.data.experimental.AUTOTUNE

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


class AtariEnvWrapper(PyEnvironmentBaseWrapper):
    def __init__(self, env):
        super(AtariEnvWrapper, self).__init__(env)
        self._observation_spec = array_spec.update_spec_dtype(
            env.observation_spec(), np.float32)

    def _step(self, action):
        time_step = self._env.step(action)
        time_step = time_step._replace(observation=np.array(
            time_step.observation).astype(np.float32))
        return time_step

    def _reset(self):
        time_step = self._env.reset()
        time_step = time_step._replace(observation=np.array(
            time_step.observation).astype(np.float32))
        return time_step

    def observation_spec(self):
        return self._observation_spec


class DQNModel(object):
    def __init__(self,
                 env_name,
                 version="v1",
                 log_dir="logs",
                 train_sequence_length=1,
                 # Params for QNetwork
                 fc_layer_params=(100,),
                 conv_layer_params=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                 dropout_layer_params=[0.2],
                 # Params for QRnnNetwork
                 input_fc_layer_params=(50,),
                 lstm_size=(20,),
                 output_fc_layer_params=(20,),
                 initial_collection_steps=1000,
                 collection_steps_per_iteration=1,
                 replay_buffer_max_length=100000,
                 train_steps_per_iteration=1,
                 batch_size=64,
                 learning_rate=1e-3,
                 epsilon_greedy=0.1,
                 target_update_tau=0.05,
                 target_update_period=5,
                 n_step_update=1,
                 gamma=0.99,
                 reward_scale_factor=2.0,
                 gradient_clipping=None,
                 log_interval=200,
                 num_eval_episodes=10,
                 eval_interval=5000,
                 train_checkpoint_interval=10000,
                 policy_checkpoint_interval=5000,
                 rb_checkpoint_interval=20000,
                 save_playing_video_interval=20000,
                 debug_summaries=False,
                 clear_logs=False):
        self.env_name = env_name
        self.num_iterations = num_iterations
        self.initial_collection_steps = initial_collection_steps
        self.collection_steps_per_iteration = collection_steps_per_iteration
        self.train_steps_per_iteration = train_steps_per_iteration
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.log_interval = log_interval
        self.num_eval_episodes = num_eval_episodes
        self.eval_interval = eval_interval
        self.train_checkpoint_interval = train_checkpoint_interval
        self.policy_checkpoint_interval = policy_checkpoint_interval
        self.rb_checkpoint_interval = rb_checkpoint_interval
        self.save_playing_video_interval = save_playing_video_interval

        self.epsilon_greedy = epsilon_greedy
        self.n_step_update = n_step_update
        self.target_update_tau = target_update_tau
        self.target_update_period = target_update_period
        self.gamma = gamma
        self.reward_scale_factor = reward_scale_factor
        self.gradient_clipping = gradient_clipping
        self.debug_summaries = debug_summaries

        # if > 1 then use a QRNN network
        self.train_sequence_length = train_sequence_length
        # Params for QNetwork
        self.fc_layer_params = fc_layer_params
        self.conv_layer_params = conv_layer_params
        self.dropout_layer_params = dropout_layer_params

        # Params for QRnnNetwork
        self.input_fc_layer_params = input_fc_layer_params
        self.lstm_size = lstm_size
        self.output_fc_layer_params = output_fc_layer_params

        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.train_dir = os.path.join(
            log_dir, env_name, f"train-{version}")
        self.eval_dir = os.path.join(log_dir, env_name, f"eval-{version}")

        if clear_logs:
            shutil.rmtree(self.train_dir)
            os.mkdir(self.train_dir)
            shutil.rmtree(self.eval_dir)
            os.mkdir(self.eval_dir)

        self.global_step = tf.Variable(0, dtype=tf.int64)

        self.tf_env = self.create_tf_env()
        self.eval_py_env = self.create_py_env()
        self.eval_tf_env = tf_py_environment.TFPyEnvironment(self.eval_py_env)

        self.tf_agent = self.create_agent()

        self.policy = self.tf_agent.policy
        self.collect_policy = self.tf_agent.collect_policy

        self.eval_metrics = [
            tf_metrics.AverageReturnMetric(buffer_size=self.num_eval_episodes),
            tf_metrics.AverageEpisodeLengthMetric(
                buffer_size=self.num_eval_episodes)
        ]

        self.train_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(),
            tf_metrics.AverageEpisodeLengthMetric(),
        ]

        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.tf_agent.collect_data_spec,
            batch_size=self.tf_env.batch_size,
            max_length=replay_buffer_max_length)

        self.train_checkpoint = tf.train.Checkpoint(metrics=metric_utils.MetricsGroup(self.train_metrics, 'train_metrics'),
                                                    agent=self.tf_agent, global_step=self.global_step)
        self.train_checkpointer = tf.train.CheckpointManager(
            self.train_checkpoint, directory=self.train_dir, max_to_keep=20)

        self.policy_saver = policy_saver.PolicySaver(
            self.policy, train_step=self.global_step)

        self.replay_buffer_checkpoint = tf.train.Checkpoint(
            replay_buffer=self.replay_buffer)
        self.replay_buffer_checkpointer = tf.train.CheckpointManager(
            self.replay_buffer_checkpoint, directory=os.path.join(self.train_dir, 'replay_buffer'), max_to_keep=1)

    def train(self, num_iterations, restore_from_checkpoints=False):
        if restore_from_checkpoints:
            self.restore_from_checkpoints()

        train_summary_writer = tf.summary.create_file_writer(self.train_dir)
        train_summary_writer.set_as_default()

        eval_summary_writer = tf.summary.create_file_writer(self.eval_dir)

        time_step = None
        policy_state = self.collect_policy.get_initial_state(
            self.tf_env.batch_size)

        timed_at_step = self.global_step.numpy()
        time_acc = 0

        initial_collect_policy = self.create_random_policy()

        print(
            f'Initializing replay buffer by collecting experience for {self.initial_collection_steps} steps with a random policy.')
        dynamic_step_driver.DynamicStepDriver(
            self.tf_env,
            initial_collect_policy,
            observers=[self.replay_buffer.add_batch] + self.train_metrics,
            num_steps=self.initial_collection_steps).run()
        dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=AUTOTUNE,
            sample_batch_size=self.batch_size,
            num_steps=self.train_sequence_length+1)
        dataset = dataset.prefetch(5)
        iterator = iter(dataset)

        collect_driver = self.create_driver(self.train_metrics)
        print("Starting Training...")
        print(dataset)
        for _ in range(num_iterations):

            start_time = time.time()
            time_step, policy_state = collect_driver.run(
                time_step=time_step,
                policy_state=policy_state,
            )

            for _ in range(self.train_steps_per_iteration):
                experience, _ = next(iterator)
                train_loss = self.tf_agent.train(experience)

            for train_metric in self.train_metrics:
                train_metric.tf_summaries(
                    train_step=self.global_step, step_metrics=self.train_metrics[:2])

            time_acc += time.time() - start_time
            py_global_step = self.global_step.numpy()

            if py_global_step % self.eval_interval == 0:
                print("Evaluating metrics...")
                results = metric_utils.eager_compute(
                    self.eval_metrics,
                    self.eval_tf_env,
                    self.policy,
                    num_episodes=self.num_eval_episodes,
                    train_step=self.global_step,
                    summary_writer=eval_summary_writer,
                    summary_prefix='Metrics',
                )
                metric_utils.log_metrics(self.eval_metrics)
            if py_global_step % self.log_interval == 0:
                print(
                    f'step = {py_global_step}, loss = {train_loss.loss}')
                steps_per_sec = (py_global_step -
                                 timed_at_step) / time_acc
                print(f'{steps_per_sec} steps/sec')
                tf.summary.scalar(
                    name='global_steps_per_sec', data=steps_per_sec, step=self.global_step)
                timed_at_step = py_global_step
                time_acc = 0

            if py_global_step % self.train_checkpoint_interval == 0:
                print("Saving model...")
                self.train_checkpointer.save(py_global_step)

            if py_global_step % self.policy_checkpoint_interval == 0:
                print("Saving Policy...")
                self.policy_saver.save(
                    os.path.join(self.train_dir, 'policy'))

            if py_global_step % self.rb_checkpoint_interval == 0:
                print("Saving rb...")
                self.replay_buffer_checkpointer.save(py_global_step)

            if py_global_step % self.save_playing_video_interval == 0:
                print("Saving playing video...")
                self.save_playing_video(
                    filename=f"{py_global_step}-trained-agent.mp4",
                    num_episodes=1
                )

    def create_agent(self):
        if self.train_sequence_length > 1:
            q_net = q_rnn_network.QRnnNetwork(
                self.tf_env.observation_spec(),
                self.tf_env.action_spec(),
                preprocessing_layers=[keras.layers.Lambda(
                    lambda observation: tf.image.convert_image_dtype(observation, dtype=tf.float32))],
                input_fc_layer_params=self.input_fc_layer_params,
                lstm_size=self.lstm_size,
                output_fc_layer_params=self.output_fc_layer_params,
                conv_layer_params=self.conv_layer_params)
        else:
            q_net = q_network.QNetwork(
                self.tf_env.observation_spec(),
                self.tf_env.action_spec(),
                fc_layer_params=self.fc_layer_params,
                conv_layer_params=self.conv_layer_params
            )
            self.train_sequence_length = self.n_step_update
        tf_agent = dqn_agent.DqnAgent(
            self.tf_env.time_step_spec(),
            self.tf_env.action_spec(),
            q_network=q_net,
            epsilon_greedy=self.epsilon_greedy,
            n_step_update=self.n_step_update,
            target_update_tau=self.target_update_tau,
            target_update_period=self.target_update_period,
            optimizer=Adam(learning_rate=self.learning_rate),
            td_errors_loss_fn=common.element_wise_squared_loss,
            gamma=self.gamma,
            reward_scale_factor=self.reward_scale_factor,
            gradient_clipping=self.gradient_clipping,
            debug_summaries=self.debug_summaries,
            train_step_counter=self.global_step)

        tf_agent.initialize()

        tf_agent.train = common.function(tf_agent.train)
        return tf_agent

    def create_driver(self, train_metrics):
        collect_driver = dynamic_step_driver.DynamicStepDriver(
            self.tf_env,
            self.collect_policy,
            observers=[self.replay_buffer.add_batch] + train_metrics,
            num_steps=self.collection_steps_per_iteration)

        collect_driver.run = common.function(collect_driver.run)
        return collect_driver

    def create_py_env(self):
        return AtariEnvWrapper(suite_gym.load(self.env_name))

    def create_tf_env(self):
        return tf_py_environment.TFPyEnvironment(self.create_py_env())

    def create_random_policy(self):
        return random_tf_policy.RandomTFPolicy(
            self.tf_env.time_step_spec(), self.tf_env.action_spec())

    def show_playing_video(self):
        self.view_policy_eval_video(self.tf_agent.policy)

    def save_playing_video(self, filename="trained-agent.mp4", num_episodes=5, fps=30):
        self.create_policy_eval_video(
            self.tf_agent.policy, filename, num_episodes, fps)

    def show_random_video(self):
        self.view_policy_eval_video(self.create_random_policy())

    def save_random_video(self, filename=f"random-agent.mp4", num_episodes=5, fps=30):
        self.create_policy_eval_video(
            self.create_random_policy(), filename, num_episodes, fps)

    def create_policy_eval_video(self, policy, filename, num_episodes=5, fps=30):
        with imageio.get_writer(os.path.join(self.log_dir, self.env_name, f"{self.env_name}-{filename}"), fps=fps) as video:
            for _ in range(num_episodes):
                time_step = self.eval_tf_env.reset()
                video.append_data(self.eval_py_env.render())
                while not time_step.is_last():
                    action_step = policy.action(time_step)
                    time_step = self.eval_tf_env.step(action_step.action)
                    video.append_data(self.eval_py_env.render())

    def restore_from_checkpoints(self):
        self.train_checkpoint.restore(
            self.train_checkpointer.latest_checkpoint)
        self.replay_buffer_checkpoint.restore(
            self.replay_buffer_checkpointer.latest_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    available_games = list((''.join(x.capitalize() or '_' for x in word.split(
        '_'))+"-v0" for word in atari_py.list_games()))
    parser.add_argument("--log_dir", help="log directory", default="logs")
    parser.add_argument("-g", "--game", help="Choose from available games: " + str(
        available_games) + ". Default is 'breakout'.", default="CartPole-v1")
    parser.add_argument(
        "-m", "--mode", help="Choose either train or evaluate.", default="train")
    parser.add_argument("-tsl", "--total_step_limit",
                        help="Choose how many total steps (frames visible by agent) should be performed. Default is '150000.", default=150000, type=int)
    parser.add_argument("-cs", "--collection_steps_per_iteration",
                        help="Choose how many steps should be collected by the driver per iteration. Default 1", default=1, type=int)
    parser.add_argument("-ts", "--train_steps_per_iteration",
                        help="Choose how many steps should be trained per iteration. Default 1", default=1, type=int)
    parser.add_argument(
        "-c", "--clip", help="Choose whether we should clip rewards to max range. Default is 'None'", default=None, type=int)
    parser.add_argument(
        "--clear", help="Clear the logs for this model. Default is 'False'", action="store_true")
    args = parser.parse_args()
    game_name = args.game
    mode = args.mode
    log_dir = args.log_dir
    clip = args.clip
    clear_logs = args.clear
    num_iterations = args.total_step_limit
    train_steps_per_iteration = args.train_steps_per_iteration
    collection_steps_per_iteration = args.collection_steps_per_iteration

    print("Selected game: ",  str(game_name))
    print("Total step limit: ",  str(num_iterations))

    model = DQNModel(
        game_name,
        log_dir=log_dir,
        gradient_clipping=clip,
        train_steps_per_iteration=train_steps_per_iteration,
        collection_steps_per_iteration=collection_steps_per_iteration,
        clear_logs=clear_logs
    )

    if mode == "train":
        model.train(num_iterations)
        model.save_playing_video(num_episodes=1)
    elif mode == "evaluate":
        model.restore_from_checkpoints()
        print("Testing the trained agent")
        model.save_playing_video(num_episodes=1)
        print("Outputing a random agent results")
        model.save_random_video(num_episodes=1)
    else:
        print(f"Could not find mode: {mode}")
