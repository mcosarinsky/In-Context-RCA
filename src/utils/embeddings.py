import torch
import faiss
from torchvision import transforms

def get_embedding(image, model, processor, normalize=True, emb_method='patch_mean'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = transforms.ToPILImage()(image) # Convert tensor to PIL image

    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)

    # Extract embedding and convert to numpy
    if emb_method=='patch_mean':
        embedding = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy() 
    elif emb_method=='output':
        embedding = outputs.pooler_output.detach().cpu().numpy()
    elif emb_method=='flat_patch':
        embedding = outputs.last_hidden_state[:, 1:].view(1, -1).detach().cpu().numpy() # first token is CLS
    else:
        raise ValueError(f"Invalid embedding method: {emb_method}. Choose from 'patch_mean', 'output', or 'flat_patch'.")
        
    if normalize: faiss.normalize_L2(embedding)

    return embedding

def get_embedding_size(model, processor):
    image_size = processor.crop_size["height"]
    patch_size = model.config.patch_size
    embeddings_size = image_size // patch_size
    
    return embeddings_size ** 2 * 768