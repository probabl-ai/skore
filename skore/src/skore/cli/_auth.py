import json
import time
import webbrowser
from pathlib import Path

import httpx

skore_hub_uri = "https://skh.k.probabl.dev"
config_dir = Path.home() / ".skore"
token_file = config_dir / "tokens.json"


def save_token(atoken: str, rtoken: str):
    # Create .skore directory in user's home if it doesn't exist
    config_dir.mkdir(exist_ok=True)

    # Save tokens to tokens.json file
    with open(token_file, "w") as f:
        json.dump({"access_token": atoken, "refresh_token": rtoken}, f, indent=4)


def read_token() -> str:
    with open(token_file) as f:
        tokens = json.load(f)
        return tokens.get("access_token")


def refresh_token() -> str:
    with open(token_file) as f:
        tokens = json.load(f)

        access_token = tokens.get("access_token")
        refresh_token = tokens.get("refresh_token")

        with httpx.Client() as client:
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            }
            data = {
                "refresh_token": refresh_token,
            }
            response = client.post(
                f"{skore_hub_uri}/identity/oauth/token/refresh",
                headers=headers,
                json=data,
            )
            response.raise_for_status()

            # breakpoint()
            token = response.json().get("token", {})
            access_token = token.get("access_token")
            refresh_token = token.get("refresh_token")

            save_token(access_token, refresh_token)
            return access_token


def start_authentication_process():
    with httpx.Client() as client:
        # Request a new authorization URL
        response = client.get(f"{skore_hub_uri}/identity/oauth/device/login")
        response.raise_for_status()
        data = response.json()

        authorization_url = data["authorization_url"]
        device_code = data["device_code"]
        user_code = data["user_code"]

        print(f"Your users code is {user_code}")

        webbrowser.open(authorization_url)

        # Start polling Skore-Hub, waiting for the token
        while True:
            response = client.get(
                f"{skore_hub_uri}/identity/oauth/device/token?device_code={device_code}"
            )
            if response.is_error:
                time.sleep(1)
            else:
                r = response.json()
                token = r.get("token", {})
                save_token(
                    token.get("access_token"),
                    token.get("refresh_token"),
                )
                print(read_token())
                print("Authenticated, token is available:")
                print(json.dumps(response.json(), indent=4))
                break


if __name__ == "__main__":
    start_authentication_process()
