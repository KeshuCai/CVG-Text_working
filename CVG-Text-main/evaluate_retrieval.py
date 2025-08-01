import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import load_text_image_dataset
from model_loader import load_model
from evaluate import evaluation, itm_eval


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model on json dataset")
    parser.add_argument('--version', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--expand', action='store_true')
    parser.add_argument('--img_type', type=str, choices=['sat', 'osm', 'stv'], required=True)
    parser.add_argument('--checkpoint', type=str, default='')
    return parser.parse_args()


def main():
    args = parse_args()

    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
        paths = cfg['paths']
        for key, value in paths.items():
            if isinstance(value, str) and '{version}' in value:
                paths[key] = value.format(version=args.version)
        globals().update(paths)

    if args.img_type == 'sat':
        root_dir = sat_root_dir
    elif args.img_type == 'osm':
        root_dir = osm_root_dir
    else:
        root_dir = stv_root_dir

    dataset = load_text_image_dataset(testset_path, root_dir)
    dataloader = DataLoader(dataset, batch_size=8, num_workers=4, shuffle=False)

    model, preprocessor, _, _ = load_model(
        args.model,
        checkpoint_path=args.checkpoint,
        expand_text=args.expand,
        is_stv=args.img_type == 'stv',
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    scores_i2t, scores_t2i = evaluation(model, dataloader, tokenizer, device, {'k_test': 128})
    metrics = itm_eval(scores_i2t, scores_t2i, dataset.txt2img, dataset.img2txt, dataset.img2building)
    for k, v in metrics.items():
        print(f'{k}: {v:.2f}')


if __name__ == '__main__':
    main()
