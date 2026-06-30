"""
Configuration and argument parser utilities.

This module provides:
    - YAML configuration loading.
    - Dynamic argparse creation.
    - Separate train/test option handlers.
"""

import argparse
import yaml
import os



def load_config(config_path: str, stage: str) -> dict:
    """
    Load YAML configuration.

    Args:
        config_path (str):
            Path to YAML configuration file.

        stage (str):
            Experiment stage:
                - train
                - test

    Returns:
        dict:
            Merged configuration.
    """

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file does not exist: {config_path}"
        )


    with open(config_path, "r") as f:
        yaml_config = yaml.safe_load(f)


    general_config = yaml_config.get(
        "general",
        {}
    )

    stage_config = yaml_config.get(
        stage,
        {}
    )


    return {
        **general_config,
        **stage_config
    }


def create_parser(stage):
    """
    Create parser dynamically from YAML.

    Args:
        stage:
            train or test

    Returns:
        argparse.ArgumentParser
    """

    parser = argparse.ArgumentParser(
        description=f"{stage} script"
    )


    # arguments needed before loading yaml
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Path to yaml configuration"
    )


    args, _ = parser.parse_known_args()


    config = load_config(
        args.config_file,
        stage
    )


    for key, value in config.items():

        parser.add_argument(
            f"--{key}",
            default=value,
            type=type(value),
            required=False,
            help=f"{key} (default: %(default)s)"
        )


    return parser



def get_options(stage):
    """
    Parse arguments.

    Args:
        stage:
            train or test

    Returns:
        argparse.Namespace
    """

    parser = create_parser(stage)

    return parser.parse_args()





def train_options():
    """
    Get training configuration.

    Returns:
        argparse.Namespace

    Example:
        args = train_options()

        args.epochs
        args.batch_size
        args.lr
    """

    return get_options(
        stage="train"
    )



def test_options():
    """
    Get testing configuration.

    Returns:
        argparse.Namespace

    Example:
        args = test_options()

        args.checkpoint
        args.batch_size
    """

    return get_options(
        stage="test"
    )