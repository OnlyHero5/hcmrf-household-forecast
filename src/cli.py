"""命令行入口 — 统一训练 / 评估 / 可视化的 CLI 接口。

支持通过 YAML 配置文件管理所有超参数，也可通过命令行参数覆盖。

用法:
  # 训练模型（使用默认配置）
  python -m src.cli train --model lstm --horizon 90

  # 训练模型（使用自定义配置）
  python -m src.cli train --config configs/default.yaml --model transformer --seed 123

  # 运行所有实验
  python -m src.cli run-all --config configs/default.yaml

  # 评估 checkpoint
  python -m src.cli evaluate --model lstm --horizon 90 --ckpt outputs/checkpoints/lstm_h90_s42.ckpt

  # 生成可视化
  python -m src.cli visualize --horizon 90 --checkpoints lstm=path1 transformer=path2
"""
import argparse
import os
import sys

import yaml

from .config import Config
from .train import train, build_model
from .evaluate import evaluate


def load_config_from_yaml(yaml_path: str) -> dict:
    """从 YAML 文件加载配置字典。

    Args:
        yaml_path: YAML 配置文件路径

    Returns:
        配置字典
    """
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def yaml_to_config(yaml_cfg: dict, overrides: dict | None = None) -> Config:
    """将 YAML 配置字典转换为 Config 对象。

    Args:
        yaml_cfg: YAML 配置字典
        overrides: 命令行参数覆盖字典（可选）

    Returns:
        Config 对象
    """
    cfg = Config()

    # 从 YAML 加载
    if "data" in yaml_cfg:
        data = yaml_cfg["data"]
        for k in ["data_path", "input_len", "short_horizon", "long_horizon", "step_size", "val_ratio"]:
            if k in data:
                setattr(cfg, k, data[k])

    if "training" in yaml_cfg:
        train_cfg = yaml_cfg["training"]
        for k in ["batch_size", "max_epochs", "patience", "learning_rate", "weight_decay", "seed", "output_root"]:
            if k in train_cfg:
                setattr(cfg, k, train_cfg[k])

    for section, prefix, keys in [
        ("lstm", "lstm", ["hidden_dim", "num_layers", "dropout"]),
        ("transformer", "transformer", ["d_model", "n_heads", "n_layers", "dim_feedforward", "dropout"]),
        ("hcmrf", "hcmrf", [
            "d_model", "n_heads", "n_layers", "dim_feedforward", "dropout",
            "encoder_kernel_size", "drd_coarse_weeks", "drd_refine_layers",
            "drd_refine_kernel", "hcm_compress_factor", "hcm_min_steps",
            "patch_size_90d", "patch_size_365d",
        ]),
    ]:
        if section in yaml_cfg:
            for key in keys:
                if key in yaml_cfg[section]:
                    setattr(cfg, f"{prefix}_{key}", yaml_cfg[section][key])

    # 命令行参数覆盖
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                setattr(cfg, k, v)

    return cfg


def cmd_train(args):
    """训练单个模型。"""
    if args.config:
        yaml_cfg = load_config_from_yaml(args.config)
        overrides = {
            "model_name": args.model,
            "horizon": args.horizon,
            "seed": args.seed,
        }
        config = yaml_to_config(yaml_cfg, overrides)
    else:
        config = Config(
            model_name=args.model,
            horizon=args.horizon,
            seed=args.seed,
        )

    print(f"Training {config.model_name} (horizon={config.horizon}, seed={config.seed})")
    ckpt = train(config)
    print(f"Best checkpoint: {ckpt}")


def cmd_run_all(args):
    """运行所有实验。"""
    if args.config:
        yaml_cfg = load_config_from_yaml(args.config)
        config = yaml_to_config(yaml_cfg)
    else:
        config = Config()

    from .run import run_all
    run_all(config=config)


def cmd_evaluate(args):
    """评估 checkpoint。"""
    config = Config(
        model_name=args.model,
        horizon=args.horizon,
    )
    result = evaluate(config, args.ckpt)
    print(f"Results: {result}")


def cmd_visualize(args):
    """生成可视化。"""
    from .visualize import plot_model_comparison

    ckpt_paths = {}
    for item in args.checkpoints:
        name, path = item.split("=", 1)
        ckpt_paths[name] = path

    save_path = f"outputs/figures/comparison_{args.horizon}d.png"
    plot_model_comparison(ckpt_paths, args.horizon, save_path)


def main():
    parser = argparse.ArgumentParser(description="家庭电力消耗预测 — 命令行工具")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # train 子命令
    train_parser = subparsers.add_parser("train", help="训练单个模型")
    train_parser.add_argument("--config", "-c", type=str, default=None, help="YAML 配置文件路径")
    train_parser.add_argument("--model", "-m", type=str, default="lstm",
                              choices=["lstm", "transformer", "hcmrf",
                                       "hcmrf_wo_MultiScale", "hcmrf_wo_Patch",
                                       "hcmrf_wo_DRD", "hcmrf_wo_Shared"],
                              help="模型名称")
    train_parser.add_argument("--horizon", type=int, default=90, choices=[90, 365], help="预测 horizon")
    train_parser.add_argument("--seed", type=int, default=42, help="随机种子")
    train_parser.set_defaults(func=cmd_train)

    # run-all 子命令
    run_parser = subparsers.add_parser("run-all", help="运行所有实验")
    run_parser.add_argument("--config", "-c", type=str, default=None, help="YAML 配置文件路径")
    run_parser.set_defaults(func=cmd_run_all)

    # evaluate 子命令
    eval_parser = subparsers.add_parser("evaluate", help="评估 checkpoint")
    eval_parser.add_argument("--model", "-m", type=str, required=True, help="模型名称")
    eval_parser.add_argument("--horizon", type=int, required=True, choices=[90, 365], help="预测 horizon")
    eval_parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint 文件路径")
    eval_parser.set_defaults(func=cmd_evaluate)

    # visualize 子命令
    vis_parser = subparsers.add_parser("visualize", help="生成可视化")
    vis_parser.add_argument("--horizon", type=int, required=True, choices=[90, 365], help="预测 horizon")
    vis_parser.add_argument("--checkpoints", nargs="+", required=True,
                            help="模型检查点，格式: name=path（可多个）")
    vis_parser.set_defaults(func=cmd_visualize)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
