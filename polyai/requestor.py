import polyai
import requests

MAX_CONNECTION_RETRIES = 2
MAX_SESSION_LIFETIME_SECS = 180
TIMEOUT_SECS = 600

def _make_session() -> requests.Session:
    if polyai.session:
        if isinstance(polyai.session, requests.Session):
            return polyai.session
        else:
            return polyai.session()

    s = requests.Session()
    s.mount(
        "http://",
        requests.adapters.HTTPAdapter(max_retries=MAX_CONNECTION_RETRIES)
    )

    return s


