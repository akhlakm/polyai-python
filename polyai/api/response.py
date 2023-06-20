import json

class PolyAIResponse:
    def __init__(self, data, headers):
        self._headers = headers
        self.data = data

    def __getitem__(self, item):
            return self.data[item]
    
    def dict(self):
        return self.data

    def __repr__(self):
        return json.dumps(self.data)

    @property
    def request_id(self) -> str:
        return self._headers.get("request-id")

    @property
    def retry_after(self) -> int:
        try:
            return int(self._headers.get("retry-after"))
        except TypeError:
            return None

    @property
    def organization(self) -> str:
        return self._headers.get("PolyAI-Organization")

    @property
    def response_ms(self) -> int:
        h = self._headers.get("Polyai-Processing-Ms")
        return None if h is None else round(float(h))
