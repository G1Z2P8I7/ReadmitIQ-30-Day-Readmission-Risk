"""Unit tests for the Streamlit dashboard rendering and multi-page routing."""

import pathlib

from streamlit.testing.v1 import AppTest


def test_streamlit_pages_render():
    """Verify that all Streamlit dashboard pages, widgets, and tables render without error."""
    app_path = pathlib.Path(__file__).parent.parent / "app" / "streamlit_app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert len(at.exception) == 0, f"Exception on Executive Summary: {at.exception}"

    # Page: Patient Risk Scoring
    at.sidebar.radio[0].set_value("Patient Risk Scoring").run()
    assert len(at.exception) == 0, f"Exception on Patient Risk Scoring: {at.exception}"

    # Page: Fairness by Group
    at.sidebar.radio[0].set_value("Fairness by Group").run()
    assert len(at.exception) == 0, f"Exception on Fairness by Group: {at.exception}"

    # Page: Model Comparison & Capacity
    at.sidebar.radio[0].set_value("Model Comparison & Capacity").run()
    assert len(at.exception) == 0, f"Exception on Model Comparison: {at.exception}"

    # Page: SHAP Interpretability
    at.sidebar.radio[0].set_value("SHAP Interpretability").run()
    assert len(at.exception) == 0, f"Exception on SHAP Interpretability: {at.exception}"
