import argparse
import json
import os
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.model as module_model
import src.utils.data as module_data
from src.utils import fix_seed, ROOT_PATH, CHECKPOINT_DIR, DATA_DIR
from src.utils.config_parser import ConfigParser
from src.utils.data import InferenceMelDataset, InferenceMelSpecCollator

DEFAULT_CHECKPOINT_PATH = CHECKPOINT_DIR / "hifi-gan.pth"
DEFAULT_INFERENCE_PATH = ROOT_PATH / "inference"


def main(config: ConfigParser, loader: DataLoader, out_dir: Path):
    logger = config.get_logger("test")

    # build model architecture
    featurizer_config = config.init_obj(config['mel_config'], module_data)
    model_config = config.init_obj(config['model_config'], module_model)

    # build all models
    generator = module_model.HiFiGenerator(model_config)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint["generator"]
    if config["n_gpu"] > 1:
        generator = torch.nn.DataParallel(generator)
    generator.load_state_dict(state_dict)
    generator.remove_weight_norm()

    # prepare model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = generator.to(device)
    generator.eval()

    results = []
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm(loader)):
            assert len(batch.transcript) == 1
            assert batch.spec is not None

            batch = batch.to(device)

            audio = generator(batch.spec)
            # batch dim becomes channel dim

            audio_path = audio_dir / (str(batch_num + 1) + ".wav")
            torchaudio.save(audio_path, audio, sample_rate=featurizer_config.sr)
            results.append({"spectrogram": loader.dataset.get_item_path(batch_num), "audio": str(audio_path)})

    metadata_file = out_dir / "metadata.json"
    with metadata_file.open("w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    fix_seed()

    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=str(ROOT_PATH / "src" / "config.json"),
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )

    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-s",
        "--source",
        default=str(DATA_DIR / "default_example_spec"),
        type=str,
        help="Path to source spectrograms",
    )

    args.add_argument(
        "-t",
        "--target_result_directory",
        default=str(DEFAULT_INFERENCE_PATH),
        type=str,
        help="Path to inference result directory",
    )

    args = args.parse_args()

    target_dir = Path(args.target_result_directory).absolute().resolve()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    with Path(args.config).open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    inference_dataset = InferenceMelDataset(Path(args.source))
    inference_loader = DataLoader(inference_dataset,
                                  batch_size=1,
                                  collate_fn=InferenceMelSpecCollator(config.init_obj(config['mel_config'],
                                                                                      module_data)),
                                  shuffle=False)

    main(config, inference_loader, target_dir)
