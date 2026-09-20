import re
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_readme_assets_and_relative_paths() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "https://huggingface.co/blog/smolvla" in readme
    assert "/home/" not in readme
    figure = ROOT / "assets/ipme_vla_simplified_architecture.svg"
    assert figure.is_file()
    svg = ET.parse(figure).getroot()
    assert svg.attrib["viewBox"] == "0 0 1600 900"
    assert "assets/ipme_vla_simplified_architecture.svg" in readme
    assert "ipme_vla_paper_workflow" not in readme
    assert "Paper-ready versions" not in readme
    local_links = re.findall(r"\[[^\]]+\]\((?!https?://)([^)#]+)", readme)
    assert all((ROOT / link).exists() for link in local_links)


def test_release_has_no_nested_git_or_experiment_artifacts() -> None:
    nested_git = [path for path in ROOT.rglob(".git") if path != ROOT / ".git"]
    assert not nested_git
    forbidden = {"wandb", "runs", "logs", "checkpoints", "datasets"}
    present = {
        path.name
        for path in ROOT.rglob("*")
        if path.is_dir() and ".git" not in path.relative_to(ROOT).parts
    }
    assert not (forbidden & present)
