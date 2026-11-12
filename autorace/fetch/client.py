import logging
import time
import urllib.parse
from typing import Any, Dict, Optional
import requests

logger = logging.getLogger(__name__)


class AutoraceAPIError(Exception):
    """APIが論理エラー(result == 'Failure')や想定外の構造を返した場合の例外"""
    def __init__(self, message: str, errors: Optional[list] = None, response_data: Optional[dict] = None):
        super().__init__(message)
        self.errors = errors or []
        self.response_data = response_data


class AutoraceClient:
    BASE_URL = "https://autorace.jp"
    CSRF_INIT_PATH = "/race_info/Live/kawaguchi"

    def __init__(
        self,
        min_interval: float = 3.0,
        timeout: float = 15.0,
        max_retries: int = 3,
        user_agent: str = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
    ):
        self.min_interval = max(3.0, min_interval)
        self.timeout = timeout
        self.max_retries = max_retries
        self._last_request_time: float = 0.0

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
            "Origin": self.BASE_URL,
            "Referer": f"{self.BASE_URL}{self.CSRF_INIT_PATH}",
        })
        self._csrf_initialized = False

    def _wait_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def _refresh_csrf_token(self) -> None:
        """初回またはトークン不一致時にGETリクエストを送りXSRF-TOKENを更新する"""
        self._wait_rate_limit()
        url = f"{self.BASE_URL}{self.CSRF_INIT_PATH}"
        logger.debug("Fetching CSRF token from %s", url)
        resp = self.session.get(url, timeout=self.timeout)
        self._last_request_time = time.monotonic()
        resp.raise_for_status()

        raw_token = self.session.cookies.get("XSRF-TOKEN")
        if not raw_token:
            raise RuntimeError("XSRF-TOKEN cookie was not set by server.")

        decoded_token = urllib.parse.unquote(raw_token)
        self.session.headers.update({
            "X-XSRF-TOKEN": decoded_token,
            "X-Requested-With": "XMLHttpRequest",
        })
        self._csrf_initialized = True

    def _post(self, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self._csrf_initialized:
            self._refresh_csrf_token()

        url = f"{self.BASE_URL}{path}"
        csrf_retried = False

        for attempt in range(1, self.max_retries + 1):
            self._wait_rate_limit()
            try:
                resp = self.session.post(
                    url,
                    json=payload if payload is not None else {},
                    timeout=self.timeout,
                )
                self._last_request_time = time.monotonic()

                # CSRF token mismatch check (status 419 or message in JSON)
                is_csrf_mismatch = resp.status_code == 419
                if not is_csrf_mismatch and resp.content:
                    try:
                        data = resp.json()
                        if isinstance(data, dict) and data.get("message") == "CSRF token mismatch.":
                            is_csrf_mismatch = True
                    except ValueError:
                        pass

                if is_csrf_mismatch:
                    if not csrf_retried:
                        logger.info("CSRF token mismatch detected. Refreshing token and retrying once.")
                        csrf_retried = True
                        self._refresh_csrf_token()
                        continue
                    else:
                        resp.raise_for_status()

                if resp.status_code >= 500 or resp.status_code == 429:
                    raise requests.HTTPError(f"Server error: HTTP {resp.status_code}", response=resp)

                if 400 <= resp.status_code < 500:
                    # 404などはやり直しても変わらないので、すぐに失敗として扱う
                    raise AutoraceAPIError(f"HTTP {resp.status_code} for {path}")
                data = resp.json()

                if not isinstance(data, dict):
                    raise AutoraceAPIError("Response is not a JSON object.")

                result_status = data.get("result")
                if result_status == "Failure":
                    errors = data.get("errors", [])
                    raise AutoraceAPIError(
                        f"API returned Failure for {path}",
                        errors=errors,
                        response_data=data,
                    )

                return data

            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                if attempt >= self.max_retries:
                    logger.error("Request failed after %d attempts: %s (%s)", attempt, path, type(exc).__name__)
                    raise
                backoff = 2 ** attempt
                logger.warning(
                    "Request error (%s) on %s. Retrying in %ds (%d/%d)...",
                    type(exc).__name__, path, backoff, attempt, self.max_retries
                )
                time.sleep(backoff)

        raise RuntimeError(f"Unreachable retry loop exit for {path}")

    def hold_today(self) -> Dict[str, Any]:
        # 開催情報はGET（ブラウザの通信で確認済み）。POSTだと404になる。
        self._wait_rate_limit()
        resp = self.session.get(f"{self.BASE_URL}/race_info/XML/Hold/Today", timeout=self.timeout)
        self._last_request_time = time.monotonic()
        resp.raise_for_status()
        data = resp.json()
        if data.get("result") == "Failure":
            raise AutoraceAPIError("API returned Failure for Hold/Today", errors=data.get("errors"), response_data=data)
        return data

    def program(self, place_code: int, race_date: str, race_no: int) -> Dict[str, Any]:
        payload = {"placeCode": place_code, "raceDate": race_date, "raceNo": race_no}
        return self._post("/race_info/Program", payload)

    def race_result(self, place_code: int, race_date: str, race_no: int) -> Dict[str, Any]:
        payload = {"placeCode": place_code, "raceDate": race_date, "raceNo": race_no}
        return self._post("/race_info/RaceResult", payload)

    def odds(self, place_code: int, race_date: str, race_no: int) -> Dict[str, Any]:
        payload = {"placeCode": place_code, "raceDate": race_date, "raceNo": race_no}
        return self._post("/race_info/Odds", payload)

    def race_refund(self, place_code: int, race_date: str) -> Dict[str, Any]:
        payload = {"placeCode": place_code, "raceDate": race_date}
        return self._post("/race_info/RaceRefund", payload)

    def other_race_info(self, place_code: int, race_date: str, race_no: int) -> Dict[str, Any]:
        """レース単位の情報（距離・走路状態・気温・グレード・節の何日目など）。出走表ページの通信で確認済み。"""
        payload = {"placeCode": place_code, "raceDate": race_date, "raceNo": race_no}
        return self._post("/race_info/OtherRaceInfo", payload)
