from src.gimap.features.fitting.presentation.view_binding import FittingViewBinding


class Check:
    def __init__(self, value):
        self.value = value

    def isChecked(self):
        return self.value


class Number:
    def __init__(self, value):
        self.value_ = value

    def value(self):
        return self.value_


class Text:
    def __init__(self, value):
        self.value_ = value

    def currentText(self):
        return self.value_


class Index:
    def __init__(self, value):
        self.value_ = value

    def currentIndex(self):
        return self.value_


def test_simulated_insitu_settings_follow_the_fit_mode_without_acquisition():
    fake = type("FakeController", (), {})()
    fake._insitu_workflow_widgets = {
        "run_mode": Text("Process Existing Sequence"),
        "fit_mode": Index(0),
        "poll": Number(2.0),
        "ui_every": Number(5),
        "stable": Check(True),
        "recursive": Check(False),
    }
    settings = FittingViewBinding._insitu_workflow_settings(fake)
    assert settings["auto_fit"] is True and settings["full_auto_fit"] is True
    assert settings["use_previous"] is False and settings["auto_refine"] is False
    assert settings["fit_every"] == 1 and settings["recursive"] is False
    fake._insitu_workflow_widgets["fit_mode"] = Index(2)  # plot curves only
    assert FittingViewBinding._insitu_workflow_settings(fake)["auto_fit"] is False


def test_ai_session_settings_migrate_without_breaking_old_sessions():
    saved = {}
    fake = type("FakeController", (), {})()
    fake._default_ai_run_settings = lambda: {
        "profile": "Balanced",
        "profile_overrides": {},
        "random_seed": 123,
        "constraint_set": {},
    }
    fake._save_ai_fitting_settings = lambda **updates: saved.update(updates)
    fake._restore_ai_run_settings_to_widgets = lambda: None
    fake._sync_workspace_ai_run_widgets = lambda: None

    FittingViewBinding._restore_ai_session_settings(
        fake,
        {"profile": "Fast", "random_seed": 9, "unknown_future_key": "ignored"},
    )
    assert saved == {"profile": "Fast", "random_seed": 9}

    # Old sessions have no ai_fitting block and therefore retain defaults or
    # current user settings without raising.
    FittingViewBinding._restore_ai_session_settings(fake, None)


def test_candidate_row_preview_loads_parameters_and_requests_plot_refresh():
    class FakeController:
        def __init__(self):
            self.calls = []

        def _load_ai_candidate_params(self, row, *, refresh_plot=True):
            self.calls.append((row, refresh_plot))
            return True

    fake = FakeController()
    rows = [{"rank": 1}, {"rank": 2}]

    FittingViewBinding._preview_ai_candidate_from_table(fake, 1, rows)

    assert fake.calls == [(rows[1], True)]
