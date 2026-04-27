"""
=====================================================
Tests for run command
=====================================================
"""

import hide_deconv.cli as hide_cli
from hide_deconv.constants import MSG_FAILURE, MSG_SUCCESS, MSG_USER_ABORT


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def get_run_callback():
    """
    Return callback function of click run command.
    """

    callback = hide_cli.cli.commands["run"].callback

    if callback is None:
        raise RuntimeError("Run command callback is not available")

    return callback


def create_confirm_sequence(values):
    """
    Create mocked Confirm.ask behavior with predefined answers.
    """

    values_iter = iter(values)

    def _ask(*args, **kwargs):
        return next(values_iter)

    return _ask


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class TestRunCommand:
    """
    Tests for hide-deconv run command behavior.
    """

    def test_run_returns_failure_on_setup_failure(self, monkeypatch, tmp_path) -> None:
        """
        Test that run command returns MSG_FAILURE if setup fails.
        """

        calls = {"preprocess": 0, "train": 0, "deconv": 0}

        monkeypatch.setattr(hide_cli, "setup_project", lambda *args: MSG_FAILURE)
        monkeypatch.setattr(
            hide_cli.console, "print_exception", lambda *args, **kwargs: None
        )
        monkeypatch.setattr(
            hide_cli,
            "preprocess",
            lambda *args: calls.__setitem__("preprocess", calls["preprocess"] + 1),
        )
        monkeypatch.setattr(
            hide_cli,
            "train_model",
            lambda *args: calls.__setitem__("train", calls["train"] + 1),
        )
        monkeypatch.setattr(
            hide_cli,
            "deconvolve_command",
            lambda *args: calls.__setitem__("deconv", calls["deconv"] + 1),
        )
        monkeypatch.setattr(hide_cli.Confirm, "ask", lambda *args, **kwargs: True)

        result = get_run_callback()(hidedeconv_path=tmp_path, fDomTransfer=True)

        assert result == MSG_FAILURE
        assert calls["preprocess"] == 0
        assert calls["train"] == 0
        assert calls["deconv"] == 0

    def test_run_returns_failure_on_user_abort(self, monkeypatch, tmp_path) -> None:
        """
        Test that run command returns MSG_FAILURE if setup is aborted by user.
        """

        calls = {"preprocess": 0, "train": 0, "deconv": 0}

        monkeypatch.setattr(hide_cli, "setup_project", lambda *args: MSG_USER_ABORT)
        monkeypatch.setattr(
            hide_cli,
            "preprocess",
            lambda *args: calls.__setitem__("preprocess", calls["preprocess"] + 1),
        )
        monkeypatch.setattr(
            hide_cli,
            "train_model",
            lambda *args: calls.__setitem__("train", calls["train"] + 1),
        )
        monkeypatch.setattr(
            hide_cli,
            "deconvolve_command",
            lambda *args: calls.__setitem__("deconv", calls["deconv"] + 1),
        )
        monkeypatch.setattr(hide_cli.Confirm, "ask", lambda *args, **kwargs: True)

        result = get_run_callback()(hidedeconv_path=tmp_path, fDomTransfer=True)

        assert result == MSG_FAILURE
        assert calls["preprocess"] == 0
        assert calls["train"] == 0
        assert calls["deconv"] == 0

    def test_run_stops_when_preprocessing_is_declined(
        self, monkeypatch, tmp_path
    ) -> None:
        """
        Test that run command exits after setup if preprocessing is declined.
        """

        calls = {"preprocess": 0, "train": 0, "deconv": 0}

        monkeypatch.setattr(hide_cli, "setup_project", lambda *args: MSG_SUCCESS)
        monkeypatch.setattr(
            hide_cli,
            "preprocess",
            lambda *args: calls.__setitem__("preprocess", calls["preprocess"] + 1),
        )
        monkeypatch.setattr(
            hide_cli,
            "train_model",
            lambda *args: calls.__setitem__("train", calls["train"] + 1),
        )
        monkeypatch.setattr(
            hide_cli,
            "deconvolve_command",
            lambda *args: calls.__setitem__("deconv", calls["deconv"] + 1),
        )
        monkeypatch.setattr(hide_cli.Confirm, "ask", create_confirm_sequence([False]))

        result = get_run_callback()(hidedeconv_path=tmp_path, fDomTransfer=True)

        assert result == MSG_SUCCESS
        assert calls["preprocess"] == 0
        assert calls["train"] == 0
        assert calls["deconv"] == 0

    def test_run_executes_preprocess_but_skips_training(
        self, monkeypatch, tmp_path
    ) -> None:
        """
        Test that run command executes preprocessing but stops before training.
        """

        calls = {"preprocess": [], "train": 0, "deconv": 0}

        monkeypatch.setattr(hide_cli, "setup_project", lambda *args: MSG_SUCCESS)
        monkeypatch.setattr(
            hide_cli,
            "preprocess",
            lambda path, fDomTransfer: calls["preprocess"].append((path, fDomTransfer)),
        )
        monkeypatch.setattr(
            hide_cli,
            "train_model",
            lambda *args: calls.__setitem__("train", calls["train"] + 1),
        )
        monkeypatch.setattr(
            hide_cli,
            "deconvolve_command",
            lambda *args: calls.__setitem__("deconv", calls["deconv"] + 1),
        )
        monkeypatch.setattr(
            hide_cli.Confirm,
            "ask",
            create_confirm_sequence([True, False]),
        )

        result = get_run_callback()(hidedeconv_path=tmp_path, fDomTransfer=False)

        assert result == MSG_SUCCESS
        assert len(calls["preprocess"]) == 1
        assert calls["preprocess"][0][0] == tmp_path.expanduser().resolve()
        assert calls["preprocess"][0][1] is False
        assert calls["train"] == 0
        assert calls["deconv"] == 0

    def test_run_executes_full_workflow(self, monkeypatch, tmp_path) -> None:
        """
        Test that run command executes preprocessing, training and deconvolution.
        """

        calls = {"preprocess": 0, "train": 0, "deconv": []}

        monkeypatch.setattr(hide_cli, "setup_project", lambda *args: MSG_SUCCESS)
        monkeypatch.setattr(
            hide_cli,
            "preprocess",
            lambda *args: calls.__setitem__("preprocess", calls["preprocess"] + 1),
        )
        monkeypatch.setattr(
            hide_cli,
            "train_model",
            lambda *args: calls.__setitem__("train", calls["train"] + 1),
        )
        monkeypatch.setattr(
            hide_cli,
            "deconvolve_command",
            lambda path, alt_bulk: calls["deconv"].append((path, alt_bulk)),
        )
        monkeypatch.setattr(
            hide_cli.Confirm,
            "ask",
            create_confirm_sequence([True, True, True]),
        )

        result = get_run_callback()(hidedeconv_path=tmp_path, fDomTransfer=True)

        assert result == MSG_SUCCESS
        assert calls["preprocess"] == 1
        assert calls["train"] == 1
        assert len(calls["deconv"]) == 1
        assert calls["deconv"][0][0] == tmp_path.expanduser().resolve()
        assert calls["deconv"][0][1] is None

    def test_run_skips_deconvolution_if_declined(self, monkeypatch, tmp_path) -> None:
        """
        Test that run command skips deconvolution when final prompt is declined.
        """

        calls = {"preprocess": 0, "train": 0, "deconv": 0}

        monkeypatch.setattr(hide_cli, "setup_project", lambda *args: MSG_SUCCESS)
        monkeypatch.setattr(
            hide_cli,
            "preprocess",
            lambda *args: calls.__setitem__("preprocess", calls["preprocess"] + 1),
        )
        monkeypatch.setattr(
            hide_cli,
            "train_model",
            lambda *args: calls.__setitem__("train", calls["train"] + 1),
        )
        monkeypatch.setattr(
            hide_cli,
            "deconvolve_command",
            lambda *args: calls.__setitem__("deconv", calls["deconv"] + 1),
        )
        monkeypatch.setattr(
            hide_cli.Confirm,
            "ask",
            create_confirm_sequence([True, True, False]),
        )

        result = get_run_callback()(hidedeconv_path=tmp_path, fDomTransfer=True)

        assert result == MSG_SUCCESS
        assert calls["preprocess"] == 1
        assert calls["train"] == 1
        assert calls["deconv"] == 0
