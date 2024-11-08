import optax, jax
from jax import numpy as jnp

def create_learning_rate_fn(num_epochs, warmup_epochs, base_learning_rate, steps_per_epoch):
  """Creates learning rate schedule."""
  cosine_alpha =  0.1
  warmup_steps = min(1000, warmup_epochs*steps_per_epoch)
  warmup_fn = optax.linear_schedule(init_value=0.00001, end_value=base_learning_rate, transition_steps=warmup_steps)
  cosine_epochs = max(num_epochs - warmup_epochs, 1)
  cosine_fn = optax.cosine_decay_schedule(init_value=base_learning_rate, decay_steps=cosine_epochs * steps_per_epoch, alpha=cosine_alpha)
  schedule_fn = optax.join_schedules(schedules=[warmup_fn, cosine_fn], boundaries=[warmup_epochs * steps_per_epoch])
  return schedule_fn


# Initialize optimizer state
def initialize_optimizer(params, nb_epochs, steps_per_epoch, lr_init=5*1e-4):
    # Optimizer setup
    warmup_epochs = max(1, nb_epochs//10)
    lr_scheduler = create_learning_rate_fn(nb_epochs, warmup_epochs, lr_init, steps_per_epoch)
    optimizer = optax.adamw(lr_scheduler)
    return optimizer, optimizer.init(params), lr_scheduler
    
# Function to accumulate gradients
def avg_grads(grads_list):
    return jax.tree_util.tree_map(lambda *grads: sum(grads)/len(grads_list), *grads_list)

def count_jax_parameters(params):
    return sum(jnp.size(p) for p in jax.tree_util.tree_leaves(params))
