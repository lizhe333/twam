from pathlib import Path

import pytest

from contact_belief_wam.paths import discover_project_root


def _make_project(path: Path):
    path.mkdir(parents=True)
    (path / "pyproject.toml").write_text("[project]\nname='contact-belief-wam'\n")
    (path / "src/contact_belief_wam").mkdir(parents=True)
    (path / "scripts").mkdir()
    (path / "scripts/cloud_train.sh").write_text("#!/usr/bin/env bash\n")
    return path


def test_wheel_site_packages_layout_resolves_checkout_root(tmp_path):
    project = _make_project(tmp_path / "contact-belief-wam")
    installed_module = project / ".venv/lib/python3.10/site-packages/contact_belief_wam/paths.py"

    result = discover_project_root(module_file=installed_module, cwd=tmp_path / "elsewhere",
                                   prefix=project / ".venv", env={})

    assert result == project.resolve()
    assert result != installed_module.resolve().parents[2]


def test_installed_wheel_uses_project_from_current_directory(tmp_path):
    project = _make_project(tmp_path / "checkout")
    installed_module = tmp_path / "python310/lib/python3.10/site-packages/contact_belief_wam/paths.py"

    assert discover_project_root(module_file=installed_module, cwd=project / "reports",
                                 prefix=tmp_path / "python310", env={}) == project.resolve()


def test_installed_wheel_finds_unique_project_next_to_environment(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    project = _make_project(workspace / "contact-belief-wam")
    installed_module = tmp_path / "external/lib/python3.10/site-packages/contact_belief_wam/paths.py"

    assert discover_project_root(module_file=installed_module, cwd=tmp_path / "elsewhere",
                                 prefix=workspace / ".venv", env={}) == project.resolve()


def test_project_local_environment_symlink_keeps_checkout_ancestor(tmp_path):
    project = _make_project(tmp_path / "checkout")
    external_env = tmp_path / "external-python"
    external_env.mkdir()
    (project / ".venv").symlink_to(external_env, target_is_directory=True)
    installed_module = project / ".venv/lib/python3.10/site-packages/contact_belief_wam/paths.py"

    assert discover_project_root(module_file=installed_module, cwd=tmp_path / "elsewhere",
                                 prefix=project / ".venv", env={}) == project.resolve()


def test_explicit_root_wins_and_is_validated(tmp_path):
    project = _make_project(tmp_path / "project")
    assert discover_project_root(module_file=tmp_path / "unrelated/paths.py", cwd=tmp_path,
                                 prefix=tmp_path, env={"CBWAM_ROOT": str(project)}) == project.resolve()

    with pytest.raises(RuntimeError, match="CBWAM_ROOT=.*not a Contact-Belief WAM project root"):
        discover_project_root(module_file=tmp_path / "unrelated/paths.py", cwd=tmp_path,
                              prefix=tmp_path, env={"CBWAM_ROOT": str(tmp_path)})


def test_non_project_wheel_install_explains_root_override(tmp_path):
    with pytest.raises(RuntimeError, match="Set CBWAM_ROOT"):
        discover_project_root(module_file=tmp_path / "python/lib/python3.10/site-packages/contact_belief_wam/paths.py",
                              cwd=tmp_path / "elsewhere", prefix=tmp_path / "python", env={})


def test_unreadable_neighbor_candidate_is_skipped_but_explicit_root_errors_clearly(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    project = _make_project(workspace / "contact-belief-wam")
    denied = workspace / "codex-daemon-0"
    denied.mkdir()
    installed_module = tmp_path / "external/lib/python3.10/site-packages/contact_belief_wam/paths.py"
    original_resolve = Path.resolve
    original_is_file = Path.is_file

    def is_file_with_unreadable_marker(path):
        if path == denied / "pyproject.toml":
            raise PermissionError("simulated unreadable pyproject")
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", is_file_with_unreadable_marker)

    result = discover_project_root(module_file=installed_module, cwd=tmp_path / "elsewhere",
                                   prefix=workspace / ".venv", env={})
    assert result == project.resolve()

    with pytest.raises(RuntimeError, match="CBWAM_ROOT=.*not accessible"):
        discover_project_root(module_file=installed_module, cwd=tmp_path / "elsewhere",
                              prefix=workspace / ".venv", env={"CBWAM_ROOT": str(denied)})

    monkeypatch.setattr(Path, "is_file", original_is_file)

    def resolve_with_unreadable_candidate(path, *args, **kwargs):
        if path == denied:
            raise PermissionError("simulated inaccessible candidate")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve_with_unreadable_candidate)

    resolved_result = discover_project_root(module_file=installed_module, cwd=tmp_path / "elsewhere",
                                             prefix=workspace / ".venv", env={})
    assert resolved_result == project.resolve()

    with pytest.raises(RuntimeError, match="explicit CBWAM_ROOT=.*simulated inaccessible candidate"):
        discover_project_root(module_file=installed_module, cwd=tmp_path / "elsewhere",
                              prefix=workspace / ".venv", env={"CBWAM_ROOT": str(denied)})

    neighbor_project = _make_project(tmp_path / "neighbor-checkout")
    original_iterdir = Path.iterdir

    def iterdir_with_unreadable_base(path):
        if path == workspace:
            raise PermissionError("simulated inaccessible directory")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", iterdir_with_unreadable_base)
    neighbor_result = discover_project_root(module_file=installed_module, cwd=tmp_path / "elsewhere",
                                            prefix=workspace / ".venv", env={})
    assert neighbor_result == neighbor_project.resolve()

