import argparse, logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from siamics.data.geo import GEO
from siamics.models.embeddings import text

ROOT_DIR = "/projects/ovcare/users/behnam_maneshgar/coding/SiaLib/cache/text_embeds/"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def embedding(LLM, path_to_save, store_attentions=False, hidden_state=-1):
    """
    Load the model and tokenizer, and generate embeddings.

    Args:
        model: model string.
        args: Parsed arguments.
    """
    model_id_map = {
        "Coral": "CohereForAI/c4ai-command-r-v01",
        "Gemma-7b": "google/gemma-7b",
        "BioGPT": "microsoft/biogpt",
        "Jamba": "ai21labs/Jamba-v0.1",
        "Llama3-70b": "meta-llama/Meta-Llama-3-70B",
        "Llama3-8b": "meta-llama/Meta-Llama-3-8B",
        "Bio-Llama3-8b": "aaditya/OpenBioLLM-Llama3-8B",
        "Med-Llama3": "ProbeMedicalYonseiMAILab/medllama3-v20",
        "Llama3-Med42-8b": "m42-health/Llama3-Med42-8B",
        "PubMedBERT": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "BioBERT":"pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"

    }

    if LLM not in model_id_map:
        raise ValueError(f"Unsupported LLM: {LLM}")

    model_id = model_id_map[LLM]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=ROOT_DIR,
        output_hidden_states=True,
        output_attentions=store_attentions,
    )
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare the datasets

    dataset = GEO()
    text.gen_embedding(model,
                       LLM,
                       tokenizer,
                       dataset,
                       num_devices=torch.cuda.device_count(),
                       batch_size=1, 
                       hidden_state=hidden_state,
                       path_to_save=path_to_save)

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Text Embeddings Retrieval (Patient Level)")
    parser.add_argument("--path_to_save", type=str, default=None, help="Path to save outputs")
    parser.add_argument("--LLM", type=str, default="PubMedBERT", help="LLM model name (e.g., Coral, Gemma-7b, BioGPT)")
    parser.add_argument("--store_attentions", type=bool, default=False, help="Store the token attentions or not")
    parser.add_argument("--hidden_state", type=int, default=-1, help="Hidden state index (0 or -1)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    embedding(args.LLM, args.path_to_save, args.store_attentions, args.hidden_state)
    logger.info("Application Ended Successfully!")