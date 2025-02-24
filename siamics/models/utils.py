import optax, jax, wandb
from jax import numpy as jnp

def create_cosine_lr_fn(num_epochs, base_learning_rate, steps_per_epoch, warmup=True, cosine_alpha=0.01):
    """Creates learning rate schedule."""
    total_steps = num_epochs * steps_per_epoch
    
    if warmup:
        warmup_steps = min(5000, total_steps // 5)
        const_steps  = max(0, (total_steps) // 5)
        cosine_steps = max(0, total_steps - (warmup_steps + const_steps))

        warmup_fn = optax.linear_schedule(init_value=base_learning_rate * 0.001, end_value=base_learning_rate, transition_steps=warmup_steps)
        const_fn = optax.constant_schedule(value=base_learning_rate)
        cosine_fn = optax.cosine_decay_schedule(init_value=base_learning_rate, decay_steps=cosine_steps, alpha=cosine_alpha)
        schedule_fn = optax.join_schedules(schedules=[warmup_fn, const_fn, cosine_fn], boundaries=[warmup_steps, warmup_steps + const_steps])
    else: 
        schedule_fn = optax.cosine_decay_schedule(init_value=base_learning_rate, decay_steps=total_steps, alpha=cosine_alpha)

    return schedule_fn

def create_const_lr_fn(num_epochs, base_learning_rate, steps_per_epoch):
    """Creates learning rate schedule."""
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = total_steps // 5
    warmup_fn = optax.linear_schedule(init_value=base_learning_rate * 0.001, end_value=base_learning_rate, transition_steps=warmup_steps)
    const_fn = optax.constant_schedule(value=base_learning_rate)
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, const_fn], boundaries=[warmup_steps])
    return schedule_fn

def create_linear_lr_fn(num_epochs, base_learning_rate, steps_per_epoch):
    """Creates learning rate schedule."""
    total_steps = num_epochs * steps_per_epoch
   
    warmup_steps = min(1000, total_steps // 10)
    const_steps = max(0, ((total_steps) // 4) - warmup_steps)
    linear_steps = max(0, total_steps - (warmup_steps + const_steps))

    warmup_fn = optax.linear_schedule(init_value=base_learning_rate * 0.001, end_value=base_learning_rate, transition_steps=warmup_steps)
    const_fn = optax.constant_schedule(value=base_learning_rate)
    linear_fn = optax.linear_schedule(init_value=base_learning_rate, end_value=base_learning_rate * 0.1, transition_steps=linear_steps)
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, const_fn, linear_fn], boundaries=[warmup_steps, warmup_steps+const_steps])
    return schedule_fn

def create_cosineAnnealing_fn(num_epochs, base_learning_rate, steps_per_epoch, nb_cycles=3, cosine_alpha=0.01):
    """Creates learning rate schedule."""

    total_steps = num_epochs * steps_per_epoch
    bracket=[]
    for i in range(1, nb_cycles):
        bracket.insert(0, total_steps // (2**i))

    bracket.append(total_steps)
    bracket.insert(0,0)
    steps = [bracket[i]-bracket[i-1] for i in range(1,len(bracket))]
    warmup_fn = optax.linear_schedule(init_value=base_learning_rate * 0.001, end_value=base_learning_rate, transition_steps=steps[0])
    
    print(f"bracket, {bracket}")
    print(f"steps, {steps}")

    schedules = [warmup_fn]
    cinit = base_learning_rate
    for i in range(1, len(steps)):
        print(f"func: cinit, {cinit}, steps {steps[i]}")
        cosine_fn = optax.cosine_decay_schedule(init_value=cinit, decay_steps=steps[i], alpha=cosine_alpha)
        schedules.append(cosine_fn)
        cinit *= (1-cosine_alpha)
 
    schedule_fn = optax.join_schedules(schedules=schedules, boundaries=bracket[1:-1])
    return schedule_fn

def CosineAnnealingWarmupRestarts_fn(
        num_epochs, 
        steps_per_epoch,
        first_cycle_steps=15,
        cycle_mult=2,
        max_lr=5e-4,
        min_lr=1e-7,
        warmup_steps=5,
        gamma=0.9,
        cosine_alpha=0.1):
    
    """Creates learning rate schedule."""
    total_steps = num_epochs * steps_per_epoch
    bracket =[]
    schedules = []

    max_decayed_lr = max_lr
    cosine_steps = first_cycle_steps
    
    while calc_steps < total_steps:
        
        warmup_fn = optax.linear_schedule(init_value=min_lr, end_value=max_lr, transition_steps=warmup_steps)
        cosine_fn = optax.cosine_decay_schedule(init_value=max_decayed_lr, decay_steps=cosine_steps, alpha=cosine_alpha)
        schedules.append(warmup_fn, cosine_fn)
        bracket.append(warmup_steps, cosine_steps)        
        
        calc_steps += (warmup_steps + cosine_steps)
        max_decayed_lr *= gamma
        cosine_steps *= cycle_mult
 
    schedule_fn = optax.join_schedules(schedules=schedules, boundaries=bracket[:-1])
    return schedule_fn

# Initialize optimizer state
def initialize_optimizer(params, nb_epochs, steps_per_epoch, lr_init, scheduler_type, momentum=0.999, warmup=True, clip_norm=1.0):
    # Optimizer setup
    
    if scheduler_type == 'cosine':
        lr_scheduler = create_cosine_lr_fn(nb_epochs, lr_init, steps_per_epoch, warmup=warmup)
    elif scheduler_type == 'const':
        lr_scheduler = create_const_lr_fn(nb_epochs, lr_init, steps_per_epoch)
    elif scheduler_type == 'linear':
        lr_scheduler = create_linear_lr_fn(nb_epochs, lr_init, steps_per_epoch)
    elif scheduler_type == 'cosineAnnealing':
        lr_scheduler = create_cosineAnnealing_fn(nb_epochs, lr_init, steps_per_epoch)
    else:
        raise ValueError(f"Invalid scheduler type: {scheduler_type}")
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(clip_norm),  # Clip gradients to a maximum global norm
        optax.adamw(lr_scheduler, b2=momentum)
    )
    
    return optimizer, optimizer.init(params), lr_scheduler
    
# Function to accumulate gradients
def avg_grads(grads_list):
    return jax.tree_util.tree_map(lambda *grads: sum(grads)/len(grads_list), *grads_list)

def count_jax_parameters(params):
    return sum(jnp.size(p) for p in jax.tree_util.tree_leaves(params))

def compute_grad_norm(grads):
    """Compute the global norm of gradients."""
    norm = jnp.sqrt(sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(grads)))
    return norm

def plot_grads_hist(grads, wandb_prefix):
    # Log histograms of gradients per layer
    try: 
        grad_histograms = {}
        flat_grads = jax.tree_util.tree_flatten_with_path(grads)
        
        for path, grad in flat_grads[0]:  # flat_grads[0] contains (path, value) tuples
            if grad is not None:  # Some params might not have gradients
                if jnp.isnan(grad).any():
                    print(f"Warning: NaN detected in gradients of {path}")
                    return None # Skip this gradient to prevent errors
                
                # Convert path (a tuple of keys) into a readable name
                name = "/".join(str(k) for k in path)  
                grad_histograms[f"{wandb_prefix}-grad/{name}"] = wandb.Histogram(grad.flatten())
        
        return grad_histograms
    except:
        print("Warning:: plot grads hist failed!")
        return None
        
    