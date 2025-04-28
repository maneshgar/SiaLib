import jax
import jax.numpy as jnp

def cosine_similarity(a, b):
    a = a / jnp.linalg.norm(a, axis=1, keepdims=True)
    b = b / jnp.linalg.norm(b, axis=1, keepdims=True)
    return jnp.matmul(a, b.T)

def contrastive_loss(z_rna, z_text, temperature=0.1):
    sim = cosine_similarity(z_rna, z_text) / temperature
    labels = jnp.arange(sim.shape[0])

    loss_rna_to_text = -jax.nn.log_softmax(sim, axis=1)[jnp.arange(sim.shape[0]), labels]
    loss_text_to_rna = -jax.nn.log_softmax(sim.T, axis=1)[jnp.arange(sim.shape[0]), labels]

    loss = (loss_rna_to_text + loss_text_to_rna).mean()
    return loss