from conftest import require_torch

require_torch()

import pytest

pytest.importorskip("transformers")

from ipme_vla.evaluation.cli import build_parser as evaluation_parser
from ipme_vla.inference.cli import build_parser as inference_parser
from ipme_vla.training.cli import build_parser as training_parser


@pytest.mark.parametrize(
    "factory",
    [training_parser, evaluation_parser, inference_parser],
)
def test_cli_help(factory, capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        factory().parse_args(["--help"])
    assert exit_info.value.code == 0
    assert "IPME-VLA" in capsys.readouterr().out
