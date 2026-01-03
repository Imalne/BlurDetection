import torch 

def predict(img):
    model.eval()
    with torch.no_grad():
        out, _ = model(img.cuda().unsqueeze(0))
        blur_map = out[:,1].cpu().squeeze().numpy()
    return blur_map
