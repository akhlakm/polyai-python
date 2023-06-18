class PolyAIResponse:
    def __init__(self, data, headers):
        self._headers = headers
        self.data = data

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
        return self._headers.get("OpenAI-Organization")

    @property
    def response_ms(self) -> int:
        h = self._headers.get("Openai-Processing-Ms")
        return None if h is None else round(float(h))
