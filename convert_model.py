import torch

from Networks import netvlad, superpoint, superglue, ultrapoint

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = superpoint.SuperPoint(nms_radius=3, max_keypoints=256).eval().to(device)
    scripted_module = torch.jit.script(model)
    scripted_module.save("models/SuperPoint_256.pt")
    print("SuperPoint_256 Converted")

    # model = superglue.SuperGlue(weights='outdoor', sinkhorn_iterations=50).eval().to(device)
    # scripted_module = torch.jit.script(model)
    # scripted_module.save("models/SuperGlue_outdoor.pt")
    # print("SuperGlue_outdoor Converted")

    # model = netvlad.NetVLAD().eval().to(device)
    # scripted_module = torch.jit.script(model)
    # scripted_module.save("models/NetVLAD.pt")
    # print("NetVLAD Converted")

    # model = ultrapoint.UltraPoint().eval().to(device)
    # scripted_module = torch.jit.script(model)
    # scripted_module.save("models/UltraPoint.pt")
    # print("UltraPoint Converted")