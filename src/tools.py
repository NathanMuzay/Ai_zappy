#!/usr/bin/python3

from pathlib import Path

import yaml


def load_yaml_config(config_path: str) -> dict:
	"""Load a YAML config file and return a dict."""
	try:
		with open(config_path, "r", encoding="utf-8") as file:
			data = yaml.safe_load(file) or {}
	except yaml.YAMLError as exc:
		raise ValueError(
			f"Invalid YAML in {config_path}. Use spaces (no tabs) for indentation."
		) from exc
	if not isinstance(data, dict):
		raise ValueError(f"Config file must contain a mapping: {config_path}")
	return data


def ensure_parent_dir(path: str) -> None:
	"""Create the parent directory of a file path when missing."""
	Path(path).parent.mkdir(parents=True, exist_ok=True)