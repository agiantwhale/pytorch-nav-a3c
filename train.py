import time
import torch
import torch.nn.functional as F

from envs import create_vizdoom_env, state_to_torch
from model import ActorCritic


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, shared_model, counter, lock, optimizer, loggers, kill):
    torch.manual_seed(args.seed + rank)

    env = create_vizdoom_env(args.config_path, args.train_scenario_path)
    env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.spaces[0].shape[0], env.action_space)

    model.train()

    state = env.reset()
    done = True
    episode_length = 0
    while not kill.is_set():
        try:
            # Sync with the shared model
            episode_start_time = time.time()
            model.load_state_dict(shared_model.state_dict())

            values = []
            log_probs = []
            rewards = []
            entropies = []
            real_depths = []
            conv_depths = []
            lstm_depths = []

            hidden = ((torch.zeros(1, 64, requires_grad=True), torch.zeros(1, 64, requires_grad=True)),
                      (torch.zeros(1, 256, requires_grad=True), torch.zeros(1, 256, requires_grad=True)))

            for step in range(args.num_steps):
                episode_length += 1
                torch_state = state_to_torch(state)
                value, logit, depth_f, depth_h, hidden = model((torch_state, hidden))
                prob = F.softmax(logit)
                log_prob = F.log_softmax(logit)
                entropy = -(log_prob * prob).sum(1, keepdim=True)
                entropies.append(entropy)

                action = prob.multinomial(1)
                log_prob = log_prob.gather(1, action)

                real_depths.append(torch_state[1])
                conv_depths.append(depth_f)
                lstm_depths.append(depth_h)

                state, reward, done, _ = env.step(action.numpy(), steps=4)

                if done:
                    episode_length = 0
                    state = env.reset()

                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)

                if done:
                    break

            R = torch.zeros(1, 1)
            if not done:
                value, _, _, _, _ = model((state_to_torch(state), hidden))
                R = value

            values.append(R)
            policy_loss = 0
            value_loss = 0
            conv_depth_loss = sum(F.binary_cross_entropy_with_logits(d, r)
                                  for d, r in zip(conv_depths, real_depths))
            lstm_depth_loss = sum(F.binary_cross_entropy_with_logits(d, r)
                                  for d, r in zip(lstm_depths, real_depths))

            gae = torch.zeros(1, 1)
            for i in reversed(range(len(rewards))):
                R = args.gamma * R + rewards[i]
                advantage = R - values[i]
                value_loss = value_loss + 0.5 * advantage.pow(2)

                # Generalized Advantage Estimataion
                delta_t = rewards[i] + args.gamma * values[i + 1] - values[i]
                gae = gae * args.gamma * args.tau + delta_t
                policy_loss = policy_loss - log_probs[i] * gae - args.entropy_coef * entropies[i]

            optimizer.zero_grad()

            final_loss = policy_loss
            final_loss += args.value_loss_coef * value_loss
            final_loss += args.conv_depth_loss_coef * conv_depth_loss
            final_loss += args.lstm_depth_loss_coef * lstm_depth_loss
            final_loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)
            ensure_shared_grads(model, shared_model)
            optimizer.step()

            with lock:
                counter.value += 1

            if loggers is not None:
                loggers['checkpoint'](counter.value)
                loggers['grad_norm'](grad_norm, counter.value)
                loggers['train_reward'](sum(rewards), counter.value)
                loggers['train_time'](time.time() - episode_start_time, counter.value)

            time.sleep(0.1)
        except Exception as err:
            print(err)
            kill.set()
