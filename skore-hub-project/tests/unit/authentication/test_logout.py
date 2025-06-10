from skore_hub_project.authentication.logout import logout
from skore_hub_project.authentication.token import Token


def test_logout(nowstr):
    token = Token("A", "B", nowstr)

    assert token.valid
    assert token.filepath.exists()

    logout()

    assert not token.valid
    assert not token.filepath.exists()
