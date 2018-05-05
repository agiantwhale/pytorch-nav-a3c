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


def train(rank, args, shared_model, counter, lock, optimizer, loggers=None):
    torch.manual_seed(args.seed + rank)

    env = create_vizdoom_env(args.config_path, args.train_scenario_path)
    env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.spaces[0].shape[0], env.action_space)

    model.train()

    state = env.reset()
    done = True
    episode_length = 0
    while True:
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())

        values = []
        log_probs = []
        rewards = []
        entropies = []

        hidden = ((torch.zeros(1, 64, requires_grad=True), torch.zeros(1, 64, requires_grad=True)),
                  (torch.zeros(1, 256, requires_grad=True), torch.zeros(1, 256, requires_grad=True)))

        for step in range(args.num_steps):
            episode_length += 1
            value, logit, depth_f, depth_h, hidden = model((state_to_torch(state), hidden))
            prob = F.softmax(logit)
            log_prob = F.log_softmax(logit)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(1)
            log_prob = log_prob.gather(1, action)

            state, reward, done, _ = env.step(action.numpy())
            # done = done or episode_length >= args.max_episode_length

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
        (policy_loss + args.value_loss_coef * value_loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)
        ensure_shared_grads(model, shared_model)
        optimizer.step()

        with lock:
            counter.value += 1

        if loggers is not None:
            loggers['grad_norm'](grad_norm, counter.value)
