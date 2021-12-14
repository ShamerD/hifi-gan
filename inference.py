import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.model as module_model
import src.utils.data as module_data
from src.utils import fix_seed, ROOT_PATH, CHECKPOINT_DIR
from src.utils.config_parser import ConfigParser
from src.utils.data import TextDataset, TextCollator

DEFAULT_CHECKPOINT_PATH = CHECKPOINT_DIR / "fastspeech.pth"
DEFAULT_INFERENCE_PATH = ROOT_PATH / "inference"


def main(config: ConfigParser, loader: DataLoader, out_dir: Path):
    logger = config.get_logger("test")

    # build model architecture
    featurizer_config = config.init_obj(config['mel_config'], module_data)
    model_config = config.init_obj(config['model_config'], module_model)

    # build all models
    model = module_model.FastSpeech(model_config)
    vocoder = module_model.Waveglow()

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    vocoder = vocoder.to(device)
    model.eval()
    vocoder.eval()

    results = []
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm(loader)):
            assert len(batch.transcript) == 1
            batch = batch.to(device)

            mels, _, mels_lens = model(batch)
            audio = vocoder.inference(mels).cpu()  # not squeezing as batch dim becomes channel dim

            audio_path = audio_dir / (str(batch_num + 1) + ".wav")
            torchaudio.save(audio_path, audio, sample_rate=featurizer_config.sr)
            results.append({"transcript": batch.transcript[0], "audio": str(audio_path)})

    metadata_file = out_dir / "metadata.json"
    with metadata_file.open("w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    fix_seed()
    sys.path.append('waveglow/')

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
        default=None,
        type=str,
        help="Path to source text sentences",
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

    # prepare test sentences
    if args.source is not None:
        texts = []
        with Path(args.source).open() as f:
            for line in f:
                texts.append(line.rstrip())
    else:
        texts = None

    inference_dataset = TextDataset(texts)
    inference_loader = DataLoader(inference_dataset,
                                  batch_size=1,
                                  collate_fn=TextCollator(),
                                  shuffle=False)

    main(config, inference_loader, target_dir)
