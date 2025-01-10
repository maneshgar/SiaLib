import optax, jax
from jax import numpy as jnp

def create_cosine_lr_fn(num_epochs, warmup_epochs, base_learning_rate, steps_per_epoch):
    """Creates learning rate schedule."""
    const_epochs = max(0, ((num_epochs) // 2) - warmup_epochs)
    cosine_epochs = max(0, num_epochs - (warmup_epochs + const_epochs))

    cosine_alpha =  0.1
    warmup_steps = warmup_epochs * steps_per_epoch
    const_steps = const_epochs * steps_per_epoch
    
    warmup_fn = optax.linear_schedule(init_value=0.0000001, end_value=base_learning_rate, transition_steps=warmup_steps)
    const_fn = optax.constant_schedule(value=base_learning_rate)
    cosine_fn = optax.cosine_decay_schedule(init_value=base_learning_rate, decay_steps=cosine_epochs * steps_per_epoch, alpha=cosine_alpha)
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, const_fn, cosine_fn], boundaries=[warmup_steps, warmup_steps + const_steps])
    return schedule_fn

def create_const_lr_fn(num_epochs, warmup_epochs, base_learning_rate, steps_per_epoch):
  """Creates learning rate schedule."""
  warmup_steps = warmup_epochs * steps_per_epoch
  warmup_fn = optax.linear_schedule(init_value=0.0000001, end_value=base_learning_rate, transition_steps=warmup_steps)
  const_fn = optax.constant_schedule(value=base_learning_rate)
  schedule_fn = optax.join_schedules(schedules=[warmup_fn, const_fn], boundaries=[warmup_steps])
  return schedule_fn

# Initialize optimizer state
def initialize_optimizer(params, nb_epochs, steps_per_epoch, lr_init, scheduler_type, clip_norm=1e2):
    # Optimizer setup
    warmup_epochs = max(1, nb_epochs//5)
    
    if scheduler_type == 'cosine':
        lr_scheduler = create_cosine_lr_fn(nb_epochs, warmup_epochs, lr_init, steps_per_epoch)
    elif scheduler_type == 'const':
        lr_scheduler = create_const_lr_fn(nb_epochs, warmup_epochs, lr_init, steps_per_epoch)
    else:
        raise ValueError(f"Invalid scheduler type: {scheduler_type}")
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(clip_norm),  # Clip gradients to a maximum global norm
        optax.adamw(lr_scheduler)
    )
    
    return optimizer, optimizer.init(params), lr_scheduler
    
# Function to accumulate gradients
def avg_grads(grads_list):
    return jax.tree_util.tree_map(lambda *grads: sum(grads)/len(grads_list), *grads_list)

def count_jax_parameters(params):
    return sum(jnp.size(p) for p in jax.tree_util.tree_leaves(params))
