import optax

def create_learning_rate_fn(num_epochs, warmup_epochs, base_learning_rate, steps_per_epoch):
  """Creates learning rate schedule."""
  warmup_fn = optax.linear_schedule(init_value=0., end_value=base_learning_rate, transition_steps=warmup_epochs * steps_per_epoch)
  cosine_epochs = max(num_epochs - warmup_epochs, 1)
  cosine_fn = optax.cosine_decay_schedule(init_value=base_learning_rate, decay_steps=cosine_epochs * steps_per_epoch)
  schedule_fn = optax.join_schedules(schedules=[warmup_fn, cosine_fn], boundaries=[warmup_epochs * steps_per_epoch])
  return schedule_fn


# Initialize optimizer state
def initialize_optimizer(params, nb_epochs, steps_per_epoch):
    # Optimizer setup
    learning_rate = 5 * 1e-4
    warmup_epochs = max(1, nb_epochs//5)
    lr_scheduler = create_learning_rate_fn(nb_epochs, warmup_epochs, learning_rate, steps_per_epoch)
    optimizer = optax.adamw(lr_scheduler)
    return optimizer, optimizer.init(params), lr_scheduler
    