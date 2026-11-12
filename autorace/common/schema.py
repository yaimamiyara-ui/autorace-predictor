from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


def _empty_to_none(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip()
        if v in ("", "-", "---"):
            return None
    return v


def _to_bool_or_none(v: Any) -> Optional[bool]:
    v = _empty_to_none(v)
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        low = v.lower()
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off"):
            return False
    return bool(v)


def _to_code_str(v: Any) -> Optional[str]:
    """コード類（事故・反則・再試走・欠車）は形式が未確定なので文字列で保持する"""
    v = _empty_to_none(v)
    return None if v is None else str(v)


class BaseAutoraceModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    @field_validator("*", mode="before")
    @classmethod
    def clean_empty_strings(cls, v: Any) -> Any:
        return _empty_to_none(v)


class HoldInfo(BaseAutoraceModel):
    place_code: int = Field(alias="placeCode")
    place_key: Optional[str] = Field(default=None, alias="placeKey")
    grade_name: Optional[str] = Field(default=None, alias="gradeName")
    grade_code: Optional[int] = Field(default=None, alias="gradeCode")
    period_start_date: Optional[str] = Field(default=None, alias="periodStartDate")
    period_end_date: Optional[str] = Field(default=None, alias="periodEndDate")
    paragraph_day: Optional[int] = Field(default=None, alias="paragraphday")
    nighter_code: Optional[int] = Field(default=None, alias="nighterCode")
    nighter_name: Optional[str] = Field(default=None, alias="nighterName")
    now_race_no: Optional[int] = Field(default=None, alias="nowRaceNo")
    race_start_time: Optional[str] = Field(default=None, alias="raceStartTime")
    temp: Optional[float] = Field(default=None, alias="temp")
    humid: Optional[float] = Field(default=None, alias="humid")
    road_temp: Optional[float] = Field(default=None, alias="roadtemp")
    weather: Optional[str] = Field(default=None, alias="weather")
    race_name: Optional[str] = Field(default=None, alias="raceName")
    race_class: Optional[str] = Field(default=None, alias="raceClass")
    distance: Optional[int] = Field(default=None, alias="distance")
    telvote_time: Optional[str] = Field(default=None, alias="telvoteTime")
    cancel_flg: Optional[bool] = Field(default=None, alias="cancelFlg")

    @field_validator("cancel_flg", mode="before")
    @classmethod
    def parse_cancel_flg(cls, v: Any) -> Optional[bool]:
        return _to_bool_or_none(v)


class ProgramEntry(BaseAutoraceModel):
    # 結合キー
    place_code: int
    race_date: str
    race_no: int
    player_code: int = Field(alias="playerCode")

    # 出走表属性 (APIの placeCode は選手の所属場コードのため branch_place_code として保持)
    car_no: int = Field(alias="carNo")
    player_name: Optional[str] = Field(default=None, alias="playerName")
    retire_flag: Optional[bool] = Field(default=None, alias="retireFlag")
    branch_place_code: Optional[int] = None
    graduation_code: Optional[int] = Field(default=None, alias="graduationCode")
    age: Optional[int] = Field(default=None, alias="age")
    bike_class: Optional[int] = Field(default=None, alias="bikeClass")
    bike_name: Optional[str] = Field(default=None, alias="bikeName")
    rank: Optional[str] = Field(default=None, alias="rank")
    handicap: Optional[float] = Field(default=None, alias="handicap")
    trial_run_time: Optional[float] = Field(default=None, alias="trialRunTime")
    trial_retry_code: Optional[str] = Field(default=None, alias="trialRetryCode")
    absent: Optional[str] = Field(default=None, alias="absent")
    race_dev: Optional[float] = Field(default=None, alias="raceDev")
    rate2: Optional[float] = Field(default=None, alias="rate2")
    rate3: Optional[float] = Field(default=None, alias="rate3")

    @field_validator("retire_flag", mode="before")
    @classmethod
    def parse_retire(cls, v: Any) -> Optional[bool]:
        return _to_bool_or_none(v)

    @field_validator("trial_retry_code", "absent", mode="before")
    @classmethod
    def parse_codes(cls, v: Any) -> Optional[str]:
        return _to_code_str(v)

    @classmethod
    def from_api_response(cls, place_code: int, race_date: str, race_no: int, body: dict) -> List["ProgramEntry"]:
        if not isinstance(body, dict):
            return []
        player_list = body.get("playerList")
        if not isinstance(player_list, list):
            return []
        entries: List[ProgramEntry] = []
        for item in player_list:
            if not isinstance(item, dict) or not _empty_to_none(item.get("playerCode")):
                continue
            data = dict(item)
            data["branch_place_code"] = data.pop("placeCode", None)
            data["place_code"] = place_code
            data["race_date"] = race_date
            data["race_no"] = race_no
            entries.append(cls.model_validate(data))
        return entries


class ResultEntry(BaseAutoraceModel):
    # 結合キー
    place_code: int
    race_date: str
    race_no: int
    player_code: int = Field(alias="playerCode")

    # レース結果属性
    car_no: int = Field(alias="carNo")
    order: Optional[int] = Field(default=None, alias="order")
    accident_code: Optional[str] = Field(default=None, alias="accidentCode")
    accident_name: Optional[str] = Field(default=None, alias="accidentName")
    retire_flag: Optional[bool] = Field(default=None, alias="retireFlag")
    motorcycle_name: Optional[str] = Field(default=None, alias="motorcycleName")
    handicap: Optional[float] = Field(default=None, alias="handicap")
    trial_retry_code: Optional[str] = Field(default=None, alias="traialRetryCode")
    trial_time: Optional[float] = Field(default=None, alias="traialTime")
    race_time: Optional[float] = Field(default=None, alias="raceTime")
    st: Optional[float] = Field(default=None, alias="st")
    foul_code: Optional[str] = Field(default=None, alias="foulCode")

    @field_validator("order", mode="before")
    @classmethod
    def parse_order(cls, v: Any) -> Optional[int]:
        v = _empty_to_none(v)
        if v is None:
            return None
        try:
            val = int(v)
            return val if val > 0 else None
        except (ValueError, TypeError):
            return None

    @field_validator("retire_flag", mode="before")
    @classmethod
    def parse_retire(cls, v: Any) -> Optional[bool]:
        return _to_bool_or_none(v)

    @field_validator("accident_code", "trial_retry_code", "foul_code", mode="before")
    @classmethod
    def parse_codes(cls, v: Any) -> Optional[str]:
        return _to_code_str(v)

    @classmethod
    def from_api_response(cls, place_code: int, race_date: str, race_no: int, body: dict) -> List["ResultEntry"]:
        if not isinstance(body, dict):
            return []
        result_list = body.get("raceResult")
        if not isinstance(result_list, list):
            return []
        entries: List[ResultEntry] = []
        for item in result_list:
            if not isinstance(item, dict) or not _empty_to_none(item.get("playerCode")):
                continue
            data = dict(item)
            data["place_code"] = place_code
            data["race_date"] = race_date
            data["race_no"] = race_no
            entries.append(cls.model_validate(data))
        return entries
