import sys
import os

sys.path.append(os.path.abspath("./"))

import argparse
import torch


from lib.engine import load_config, build_model, build_dataloaders, Tester, init_logger, load_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monolite detection script")
    parser.add_argument("--cfg", dest="cfg", required=True, help="path to config file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.cfg)
    output_dir = cfg.get("output_dir", "./runs/exp")
    logger = init_logger(output_dir)

    _, _, test_loader = build_dataloaders(cfg)

    model = build_model(cfg["model"]).to(device)
    checkpoint = cfg.get("test", {}).get("checkpoint") or cfg.get("resume", None)
    if checkpoint:
        logger.info(f"load checkpoint: {checkpoint}")
        load_checkpoint(model, None, None, checkpoint, map_location=device)

    tester = Tester(model, test_loader, logger, device)
    tester.test()
