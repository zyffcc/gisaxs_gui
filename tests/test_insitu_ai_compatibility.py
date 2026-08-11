from controllers.fitting_controller import FittingController


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


def test_simulated_insitu_settings_include_profile_without_acquisition():
    fake = type("FakeController", (), {})()
    fake._insitu_workflow_widgets = {
        "run_mode": Text("Process Existing Sequence"),
        "auto_show": Check(True),
        "auto_cut": Check(True),
        "auto_fit": Check(True),
        "use_previous": Check(True),
        "full_auto_fit": Check(True),
        "profile": Text("Fast"),
        "auto_refine": Check(False),
        "poll": Number(2.0),
        "fit_every": Number(1),
        "ui_every": Number(5),
        "stable": Check(True),
    }
    settings = FittingController._insitu_workflow_settings(fake)
    assert settings["profile"] == "Fast"
    assert settings["auto_fit"] is True
    assert settings["full_auto_fit"] is True


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

    FittingController._restore_ai_session_settings(
        fake,
        {"profile": "Fast", "random_seed": 9, "unknown_future_key": "ignored"},
    )
    assert saved == {"profile": "Fast", "random_seed": 9}

    # Old sessions have no ai_fitting block and therefore retain defaults or
    # current user settings without raising.
    FittingController._restore_ai_session_settings(fake, None)


def test_candidate_row_preview_loads_parameters_and_requests_plot_refresh():
    class FakeController:
        def __init__(self):
            self.calls = []

        def _load_ai_candidate_params(self, row, *, refresh_plot=True):
            self.calls.append((row, refresh_plot))
            return True

    fake = FakeController()
    rows = [{"rank": 1}, {"rank": 2}]

    FittingController._preview_ai_candidate_from_table(fake, 1, rows)

    assert fake.calls == [(rows[1], True)]
