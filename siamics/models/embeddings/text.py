import os
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set GEOparse logger to WARNING (suppress DEBUG messages)
logging.getLogger("GEOparse").setLevel(logging.WARNING)

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
    return torch.sum(model_output * input_mask_expanded, dim=1) / torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

def gen_embedding(model,
                  LLM, 
                  tokenizer,
                  dataset,
                  num_devices,
                  batch_size,
                  path_to_save,
                  hidden_state=-1,
                  store_attentions=False,
                  overwrite=False):
    """
    Generate text embeddings using a pretrained model.

    Args:
        model: Pretrained model.
        tokenizer: Tokenizer for the model.
        dataset: Dataset object.
        num_devices: Number of devices to use.
        batch_size: Batch size for DataLoader.
    """
    logger.info("Generating Text Embeddings ...")
    logger.info(f"Available devices: {num_devices}")
    logger.info(f"Batch Size: {batch_size}")

    def format_text_description(gse_id, gsm_id):
        keys={"source_name_ch1", "organism_ch1", "characteristics_ch1", "treatment_protocol_ch1", "growth_protocol_ch1"}
        meta = dataset.get_metadata(gse_id, gsm_id, keys=keys)
        # print(meta)
        text = ""
        if 'source_name_ch1' in meta and meta['source_name_ch1']:
            text += f"name:{meta['source_name_ch1'][0]};"
        if 'organism_ch1' in meta and meta['organism_ch1']:
            text += f"Organism:{meta['organism_ch1'][0]};"
        if 'characteristics_ch1' in meta and meta['characteristics_ch1']:
            text += f"Characteristics:{meta['characteristics_ch1'][0]};"
        if 'treatment_protocol_ch1' in meta and meta['treatment_protocol_ch1']:
            text += f"Treatment:{meta['treatment_protocol_ch1'][0]};"
        if 'growth_protocol_ch1' in meta and meta['growth_protocol_ch1']:
            text += f"Growth:{meta['growth_protocol_ch1'][0]};"
        return text

    def collate_fn(batch):
        _, metadata, idx = dataset.collate_fn(batch, num_devices=num_devices)
        gse_id= metadata["group_id"].item()
        gsm_id= metadata["sample_id"].item()
        text = format_text_description(gse_id=gse_id, gsm_id=gsm_id)
        tokens = tokenizer([text], padding=True, truncation=True, max_length=512, return_tensors="pt", add_special_tokens=True)
        return (text, tokens), (metadata, gse_id, gsm_id), idx

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=num_devices * 2,
    )

    for (text, tokens), (metadata, gse_id, gsm_id), idx in tqdm(data_loader):
        feats = {}
                    # Save the embeddings
        template = os.path.join(
            path_to_save,
            LLM)
        filename=os.path.join(template, f"{gsm_id}.npz")
        
        if os.path.exists(filename) and not overwrite:
            try:
                np.load(filename)
                logger.info(f"File {filename} already exists and is not broken. Skipping...")
                continue
            except Exception as e:
                logger.warning(f"File {filename} exists but is broken. Reprocessing...")
        
        try:
            with torch.no_grad():
                input_ids = tokens["input_ids"].to(model.device)
                outputs = model(input_ids, output_hidden_states=True, output_attentions=store_attentions)
            
            feats["hidden_states"] = outputs.hidden_states[hidden_state].cpu().numpy()
            
            if store_attentions:
                feats["attentions"] = outputs.attentions[hidden_state].to(torch.float16).cpu().numpy()

            feats["sentence_embeddings"] = mean_pooling(
                torch.tensor(feats["hidden_states"]), tokens['attention_mask']
            ).cpu().numpy()

            os.makedirs(template, exist_ok=True)
            np.savez(filename, **feats)
            logger.info(f"Finished processing patient {gsm_id}")


        except Exception as e:
            logger.error(f"Error processing patient {gsm_id}: {e}")

    logger.info("Successfully Done!")
    return
