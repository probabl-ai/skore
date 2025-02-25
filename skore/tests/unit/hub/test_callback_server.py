import httpx
from skore.hub.callback_server import launch_callback_server


def test_server_dies_and_send_state():
    def callback(state):
        assert state == "azertyuiop"

    port, event = launch_callback_server(callback=callback)

    try:
        with httpx.Client() as client:
            client.get(f"http://localhost:{port}?state=azertyuiop")
    except Exception:
        event.set()
