import numpy as np
actions = list([1,2,3])
q_vals = list([10, 5, 2.5])
logits = list([0.4, 0.4, 0.4])
grads = list([0.01, 0.01, 0.01])


lr = 0.1
def update_q_value(actions):
    i = np.random.choice(list(range(len(actions))))
    # print('This is i: {}, actions[i]: {}, q_vals[i] :{}'.format(i, actions[i], q_vals[i]))
    value_est = q_vals[i] +  np.random.randn() * q_vals[i]
    logits[i] += lr * value_est * grads[i]
    return i, lr*value_est / 10

for i in range(20):
    action, val = update_q_value(actions)
    q_vals[action] += val
    print(q_vals)