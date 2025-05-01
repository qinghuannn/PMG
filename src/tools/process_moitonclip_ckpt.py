import argparse
import torch



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt_path, map_location="cpu")["state_dict"]
    net_state_dict = {".".join(x.split(".")[1:]): ckpt[x] for x in ckpt}
    torch.save(net_state_dict, args.save_path)
    print("Done!")