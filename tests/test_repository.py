import re
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).parents[1]


def test_readme_assets_and_relative_paths() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "https://huggingface.co/blog/smolvla" in readme
    assert "/home/" not in readme
    for relative in (
        "assets/ipme_vla_paper_workflow.png",
        "assets/ipme_vla_paper_workflow.svg",
        "assets/ipme_vla_paper_workflow.pdf",
    ):
        assert (ROOT / relative).is_file()
    with Image.open(ROOT / "assets/ipme_vla_paper_workflow.png") as image:
        assert image.width > 1000 and image.height > 500
        image.verify()
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
