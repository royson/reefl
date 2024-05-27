import numpy as np

def update_lr(current_round: int, total_rounds, start_lr: float, end_lr: float):
    """Applies exponential learning rate decay using the defined start_lr and end_lr.
     The basic eq is as follows:
    init_lr * exp(-round_i*gamma) = lr_at_round_i
     A more common one is :
        end_lr = start_lr*gamma^total_rounds"""

    # first we need to compute gamma, which will later be used
    # to obtain the lr for the current round

    gamma = np.power(end_lr / start_lr, 1.0 / total_rounds)
    current_lr = start_lr * np.power(gamma, current_round)
    return current_lr, gamma

def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             minimum_learning_rate,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0,):
    """Cosine decay schedule with warm up period.
    Cosine annealing learning rate as described in:
      Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
      ICLR 2017. https://arxiv.org/abs/1608.03983
    In this schedule, the learning rate grows linearly from warmup_learning_rate
    to learning_rate_base for warmup_steps, then transitions to a cosine decay
    schedule.
    Arguments:
        global_step {int} -- global step.
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.
    Keyword Arguments:
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
    Returns:
      a float representing learning rate.
    Raises:
      ValueError: if warmup_learning_rate is larger than learning_rate_base,
        or if warmup_steps is larger than total_steps.
    """
 
    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
        np.pi *
        (global_step - warmup_steps - hold_base_rate_steps
         ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
    learning_rate = float(np.where(global_step > total_steps, 0.0, learning_rate))
    return max(minimum_learning_rate, learning_rate)

if __name__=='__main__':
    for ROUND in range(0,1000):
        print(ROUND, cosine_decay_with_warmup(ROUND,
            learning_rate_base=0.05,
            total_steps=1000,
            minimum_learning_rate=1e-3,
            warmup_learning_rate=0,
            warmup_steps=0,
            hold_base_rate_steps=0.))