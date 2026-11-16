from prism.eval.provenance import git_provenance
from prism.eval.report import _markdown


def test_git_provenance_has_commit_in_repo():
    p = git_provenance()
    # In this workspace we expect a git repo; tolerate missing git in odd CI sandboxes
    if p.get("commit"):
        assert len(p["commit"]) >= 7
        assert p.get("commit_short") == p["commit"][:12]


def test_markdown_includes_provenance():
    md = _markdown(
        {
            "run_id": "x",
            "eval_config": "c",
            "rag_config": "r",
            "dataset": "d",
            "n_questions": 1,
            "generation_enabled": False,
            "provenance": {
                "commit": "abcd1234eeee",
                "commit_short": "abcd1234eeee",
                "branch": "develop",
                "dirty": True,
                "describe": "abcd1234-dirty",
            },
            "retrieval": {},
            "ops": {},
        }
    )
    assert "abcd1234eeee" in md
    assert "dirty" in md
