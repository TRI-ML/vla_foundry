import argparse

import ray
from img2dataset import download


def main(args, config):
    download(**config)


@ray.remote
def main_ray(args, config):
    config["distributor"] = "ray"
    return download(**config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", type=str, required=True, choices=["local", "ray"])
    parser.add_argument("--input_path", type=str, required=True, help="can be s3 or local")
    parser.add_argument("--output_path", type=str, required=True, help="can be s3 or local")
    parser.add_argument("--url_col", type=str, default="url", help="Column name for URLs")
    parser.add_argument("--caption_col", type=str, default="re_caption", help="Column name for captions")
    parser.add_argument(
        "--save_additional_columns",
        type=str,
        nargs="*",
        default=[],
        help="Additional columns to save (space-separated list)",
    )
    args = parser.parse_args()

    config = {
        "url_list": args.input_path,
        "image_size": 512,
        "output_folder": args.output_path,
        "processes_count": 1,
        "thread_count": 32,
        "output_format": "webdataset",
        "encode_quality": 101,
        "encode_format": "webp",
        "resize_mode": "keep_ratio_largest",
        "resize_only_if_bigger": True,
        "input_format": "parquet",
        "url_col": args.url_col,  # Use command line argument
        "caption_col": args.caption_col,  # Use command line argument
        "save_additional_columns": args.save_additional_columns,  # Use command line argument
        "number_sample_per_shard": 512,
        "oom_shard_count": 8,
        "retries": 2,
        "enable_wandb": True,
        "wandb_project": "vla_foundry",
    }

    if args.cluster == "ray":
        ray.init(address="auto")
        result = main_ray.remote(args, config)
        output = ray.get(result)
        print(output)
    else:
        main(args, config)
