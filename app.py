"""Streamlit app entrypoint for the Motion Estimation dashboard."""
from ui.dashboard import show_dashboard


def main() -> None:
    show_dashboard()


if __name__ == "__main__":
    main()
