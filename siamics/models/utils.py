import optax, jax

def create_learning_rate_fn(num_epochs, warmup_epochs, base_learning_rate, steps_per_epoch):
  """Creates learning rate schedule."""
  warmup_fn = optax.linear_schedule(init_value=0., end_value=base_learning_rate, transition_steps=warmup_epochs * steps_per_epoch)
  cosine_epochs = max(num_epochs - warmup_epochs, 1)
  cosine_fn = optax.cosine_decay_schedule(init_value=base_learning_rate, decay_steps=cosine_epochs * steps_per_epoch, alpha=0.01)
  schedule_fn = optax.join_schedules(schedules=[warmup_fn, cosine_fn], boundaries=[warmup_epochs * steps_per_epoch])
  return schedule_fn


# Initialize optimizer state
def initialize_optimizer(params, nb_epochs, steps_per_epoch, lr_init=1e-3):
    # Optimizer setup
    warmup_epochs = max(1, nb_epochs//5)
    lr_scheduler = create_learning_rate_fn(nb_epochs, warmup_epochs, lr_init, steps_per_epoch)
    optimizer = optax.adamw(lr_scheduler)
    return optimizer, optimizer.init(params), lr_scheduler
    
# Function to accumulate gradients
def sum_grads(grads_list):
    return jax.tree_util.tree_map(lambda *grads: sum(grads), *grads_list)
