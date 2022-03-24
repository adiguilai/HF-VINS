import torch

from Networks import netvlad, superpoint, superglue

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = superpoint.SuperPoint(nms_radius=3, max_keypoints=1024).eval().to(device)
    scripted_module = torch.jit.script(model)
    scripted_module.save("models/SuperPoint_1024.pt")
    print("SuperPoint Converted")

    model = netvlad.NetVLAD().eval().to(device)
    scripted_module = torch.jit.script(model)
    scripted_module.save("models/NetVLAD.pt")
    print("NetVLAD Converted")
