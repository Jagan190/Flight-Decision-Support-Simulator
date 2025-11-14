"""
Updated Flight Decision Support Simulator (Flask)
- METAR fetch + decode
- Decoded METAR ATIS-style audio: cloud heights/readings like 1200 -> "twelve hundred"
- Keeps previous features (manual METAR, audio files, decision engine)
"""

import os
import re
import uuid
import requests
import xml.etree.ElementTree as ET
from math import radians
from flask import Flask, render_template, request, url_for

# text-to-speech
try:
    from gtts import gTTS
except Exception:
    gTTS = None

app = Flask(__name__)

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_DIR = os.path.join(PROJECT_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

AVWX_API_KEY = os.getenv("AVWX_API_KEY", None)


def generate_audio_briefing(text, filename_prefix="briefing", lang="en"):
    """Generate MP3 in static/ and return filename or None on failure."""
    if gTTS is None:
        return None
    try:
        filename = f"{filename_prefix}_{uuid.uuid4().hex}.mp3"
        filepath = os.path.join(STATIC_DIR, filename)
        tts = gTTS(text=text, lang=lang)
        tts.save(filepath)
        return filename
    except Exception as e:
        print("TTS generation failed:", e)
        return None


def fetch_metar_for_icao(icao_code, hours_before_now=3):
    if not icao_code:
        return {"icao": None, "raw_text": None, "observation_time": None, "source": None, "error": "No ICAO provided."}
    icao = icao_code.strip().upper()
    try:
        tgftp_url = f"https://tgftp.nws.noaa.gov/data/observations/metar/stations/{icao}.TXT"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        r = requests.get(tgftp_url, headers=headers, timeout=8)
        if r.status_code == 200 and r.text.strip():
            lines = [ln.strip() for ln in r.text.strip().splitlines() if ln.strip()]
            if len(lines) >= 2:
                obs_time = lines[0]
                raw = lines[1]
                return {"icao": icao, "raw_text": raw, "observation_time": obs_time, "source": "TGFTP", "error": None}
            else:
                return {"icao": icao, "raw_text": lines[0], "observation_time": None, "source": "TGFTP-singleline", "error": None}
    except Exception as e:
        tgftp_err = f"TGFTP error: {e}"

    try:
        aw_url = "https://aviationweather.gov/cgi-bin/data/metar.php" f"?ids={icao}&format=xml&hours={hours_before_now}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                   "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}
        r = requests.get(aw_url, headers=headers, timeout=10)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        metar_elem = root.find(".//METAR") or root.find(".//metar")
        if metar_elem is not None:
            raw = metar_elem.findtext("raw_text") or metar_elem.findtext("rawText") or None
            obs = metar_elem.findtext("observation_time") or metar_elem.findtext("observationTime") or None
            if raw:
                return {"icao": icao, "raw_text": raw.strip(), "observation_time": obs, "source": "AviationWeather-CGI", "error": None}
    except requests.HTTPError as he:
        aw_err = f"AviationWeather HTTP error: {he}"
    except Exception as e:
        aw_err = f"AviationWeather parse error: {e}"

    if AVWX_API_KEY:
        try:
            avwx_url = f"https://avwx.rest/api/metar/{icao}"
            headers = {"Authorization": AVWX_API_KEY, "Accept": "application/json"}
            r = requests.get(avwx_url, headers=headers, timeout=10)
            r.raise_for_status()
            j = r.json()
            raw = j.get("raw") or j.get("raw_text") or None
            obs = None
            time_node = j.get("time")
            if isinstance(time_node, dict):
                obs = time_node.get("dt") or time_node.get("observed")
            elif isinstance(time_node, str):
                obs = time_node
            if raw:
                return {"icao": icao, "raw_text": raw.strip(), "observation_time": obs, "source": "AVWX", "error": None}
        except Exception as e:
            avwx_err = f"AVWX error: {e}"

    err_parts = []
    if 'tgftp_err' in locals(): err_parts.append(tgftp_err)
    if 'aw_err' in locals(): err_parts.append(aw_err)
    if 'avwx_err' in locals(): err_parts.append(avwx_err)
    err_text = " / ".join(err_parts) if err_parts else "No METAR source succeeded."
    return {"icao": icao, "raw_text": None, "observation_time": None, "source": None, "error": f"METAR fetch failed: {err_text}"}


def simple_parse_metar(raw):
    out = {"wind_dir_deg": None, "wind_kts": None, "gusts_kts": None, "vis_km": None}
    if not raw:
        return out
    m = re.search(r"\b(VRB|\d{3})(\d{2,3})(?:G(\d{2,3}))?KT\b", raw)
    if m:
        dir_part = m.group(1); spd = m.group(2); gust = m.group(3)
        out["wind_dir_deg"] = 0.0 if dir_part == "VRB" else float(dir_part)
        out["wind_kts"] = float(spd)
        out["gusts_kts"] = float(gust) if gust else None
    mvis = re.search(r"\b(\d{4})\b", raw)
    if mvis:
        vis_m = int(mvis.group(1))
        out["vis_km"] = round(vis_m / 1000.0, 1) if vis_m < 10000 else 10.0
    else:
        msm = re.search(r"(\d+\/\d+|\d+)(?=SM)", raw)
        if msm:
            try:
                sm_str = msm.group(1)
                if "/" in sm_str:
                    a, b = sm_str.split("/")
                    val = float(a) / float(b)
                else:
                    val = float(sm_str)
                out["vis_km"] = round(val * 1.852, 1)
            except Exception:
                pass
    return out


class FlightDecisionSupport:
    def __init__(self, ac_perf):
        self.ac = ac_perf

    def density_altitude_estimate(self, field_elevation_ft, oat_celsius, qnh_hpa=1013.25):
        pressure_diff_hpa = 1013.25 - qnh_hpa
        pressure_altitude_ft = pressure_diff_hpa * 27 + field_elevation_ft
        isa_temp_c = 15.0 - 2.0 * (pressure_altitude_ft / 1000.0)
        density_altitude_ft = pressure_altitude_ft + 120.0 * (oat_celsius - isa_temp_c)
        return pressure_altitude_ft, isa_temp_c, density_altitude_ft

    def wind_components(self, wind_dir_deg, wind_speed_kts, runway_heading_deg):
        try:
            ang = abs((float(wind_dir_deg) - float(runway_heading_deg) + 360) % 360)
        except Exception:
            ang = 0.0
        if ang > 180:
            ang = 360 - ang
        from math import cos, sin, radians
        head = round(float(wind_speed_kts) * cos(radians(ang)), 1) if wind_speed_kts is not None else 0.0
        cross = round(float(wind_speed_kts) * sin(radians(ang)), 1) if wind_speed_kts is not None else 0.0
        return head, cross

    def compute_takeoff_requirement(self, density_altitude_ft):
        da_thousands = density_altitude_ft / 1000.0
        factor = 1.0 + 0.10 * da_thousands
        required_takeoff_m = self.ac['takeoff_run_available_m'] * factor
        required_landing_m = self.ac['landing_distance_m'] * factor
        return required_takeoff_m, required_landing_m, factor

    def fuel_margin_hours(self, fuel_kg, cruise_burn_kgph, reserve_hours=0.5):
        endurance_hr = fuel_kg / cruise_burn_kgph
        margin_hr = endurance_hr - reserve_hours
        return endurance_hr, margin_hr

    def assess_weather_risk(self, visibility_km, ceiling_ft, wind_kts, gusts_kts=None, precip=False):
        score = 0.0
        if visibility_km is None:
            visibility_km = 10.0
        if visibility_km < 1:
            score += 40
        elif visibility_km < 5:
            score += 20
        elif visibility_km < 10:
            score += 5
        if ceiling_ft < 200:
            score += 40
        elif ceiling_ft < 500:
            score += 25
        elif ceiling_ft < 1000:
            score += 10
        if wind_kts and wind_kts >= 30:
            score += 25
        elif wind_kts and wind_kts >= 20:
            score += 12
        if gusts_kts and gusts_kts - (wind_kts or 0) >= 10:
            score += 8
        if precip:
            score += 10
        return min(100, score)

    def evaluate(self, flight):
        out = {}
        o = flight['origin']; d = flight['destination']
        pa_o, isa_o, da_o = self.density_altitude_estimate(o['elevation_ft'], o['oat_c'], o.get('qnh', 1013.25))
        pa_d, isa_d, da_d = self.density_altitude_estimate(d['elevation_ft'], d['oat_c'], d.get('qnh', 1013.25))
        out['density_altitude_origin_ft'] = round(da_o, 1)
        out['density_altitude_destination_ft'] = round(da_d, 1)
        req_to_m, req_ldg_m, factor = self.compute_takeoff_requirement(da_d)
        out['required_takeoff_m_est'] = round(req_to_m, 1)
        out['required_landing_m_est'] = round(req_ldg_m, 1)
        out['da_factor'] = round(factor, 3)
        out['destination_runway_length_m'] = d['runway_length_m']
        runway_ok = d['runway_length_m'] >= req_ldg_m * 1.15
        out['runway_suitable'] = bool(runway_ok)
        hw_o, cw_o = self.wind_components(o.get('wind_dir_deg', 0), o.get('wind_kts', 0), o.get('runway_heading_deg', 0))
        hw_d, cw_d = self.wind_components(d.get('wind_dir_deg', 0), d.get('wind_kts', 0), d.get('runway_heading_deg', 0))
        out['origin_headwind_kts'] = hw_o; out['origin_crosswind_kts'] = cw_o
        out['dest_headwind_kts'] = hw_d; out['dest_crosswind_kts'] = cw_d
        weather_risk = self.assess_weather_risk(d.get('vis_km', 10), d.get('ceil_ft', 10000), d.get('wind_kts', 0), d.get('gusts_kts', None), d.get('precip', False))
        out['destination_weather_risk'] = round(weather_risk, 1)
        endurance_hr, margin_hr = self.fuel_margin_hours(flight['fuel_kg'], self.ac['fuel_burn_kg_per_hr'], reserve_hours=flight.get('required_reserve_hr', 0.5))
        out['endurance_hr'] = round(endurance_hr, 2)
        out['fuel_margin_hr_over_reserve'] = round(margin_hr, 2)
        alternates = []
        for a in flight.get('alternates', []):
            alt = {'name': a.get('name', 'ALT'), 'runway_len_m': a['runway_length_m']}
            pa_a, isa_a, da_a = self.density_altitude_estimate(a['elevation_ft'], a['oat_c'], a.get('qnh', 1013.25))
            required_to_a, required_ldg_a, factor_a = self.compute_takeoff_requirement(da_a)
            alt['required_ldg_m'] = round(required_ldg_a, 1)
            alt['suitable_runway'] = a['runway_length_m'] >= required_ldg_a * 1.15
            if 'distance_nm' in a and self.ac['cruise_speed_kts'] > 0:
                alt['diversion_time_hr'] = round(a['distance_nm'] / self.ac['cruise_speed_kts'], 3)
            else:
                alt['diversion_time_hr'] = None
            alternates.append(alt)
        out['alternates_eval'] = alternates
        score = weather_risk
        if margin_hr < 0:
            score += 30
        elif margin_hr < 0.5:
            score += 10
        if not runway_ok:
            score += 30
        if not any(a['suitable_runway'] for a in alternates):
            score += 20
        out['aggregate_risk_score'] = min(100, round(score, 1))
        if out['aggregate_risk_score'] < 20 and margin_hr >= 0.5 and runway_ok:
            rec = "GO"
        elif out['aggregate_risk_score'] < 40 and margin_hr >= 0.25 and runway_ok:
            rec = "GO WITH CAUTION / MONITOR"
        elif out['aggregate_risk_score'] < 60 and margin_hr >= 0 and any(a['suitable_runway'] for a in alternates):
            rec = "DELAY OR PROCEED WITH ALTERNATE PLANNED"
        else:
            rec = "DIVERT / RETURN / DELAY - FURTHER CHECKS REQUIRED"
        out['recommendation'] = rec
        return out


# -------------------------
# ATIS-style number helper (e.g., 1200 -> "twelve hundred")
# -------------------------
_NUM_WORDS = {
    0: "zero",1: "one",2:"two",3:"three",4:"four",5:"five",6:"six",7:"seven",8:"eight",9:"nine",
    10:"ten",11:"eleven",12:"twelve",13:"thirteen",14:"fourteen",15:"fifteen",16:"sixteen",
    17:"seventeen",18:"eighteen",19:"nineteen",20:"twenty",30:"thirty",40:"forty",50:"fifty",
    60:"sixty",70:"seventy",80:"eighty",90:"ninety"
}

def _two_digit_to_words(n):
    if n < 20:
        return _NUM_WORDS[n]
    tens = (n // 10) * 10
    ones = n % 10
    if ones == 0:
        return _NUM_WORDS[tens]
    return f"{_NUM_WORDS[tens]} {_NUM_WORDS[ones]}"

def number_to_atis(n):
    """
    Convert number (int, typically hundreds/ thousands) to ATIS-style words:
    1200 -> "twelve hundred"
    1500 -> "fifteen hundred"
    200  -> "two hundred"
    0    -> "zero"
    For non-exact-hundred values, fall back to spoken digits groups (e.g., 1250 -> "one two five zero" fallback).
    """
    try:
        n = int(n)
    except Exception:
        return str(n)
    if n == 0:
        return "zero"
    # If exact hundreds (e.g., 1200, 1500, 200)
    if n % 100 == 0:
        hundreds_part = n // 100
        # e.g., 1200 -> hundreds_part = 12 -> "twelve hundred"
        if hundreds_part < 100:
            if hundreds_part < 100:
                # 1..99
                if hundreds_part < 20:
                    words = _NUM_WORDS.get(hundreds_part, str(hundreds_part))
                else:
                    words = _two_digit_to_words(hundreds_part)
                return f"{words} hundred"
        # fallback: spell digits groups
    # If it's thousands like 10000, try simple spoken thousands
    if n >= 1000 and n % 100 == 0:
        # e.g., 5000 -> "fifty hundred" is awkward; better "five thousand"
        if n % 1000 == 0:
            thousands = n // 1000
            if thousands < 20:
                return f"{_NUM_WORDS.get(thousands, str(thousands))} thousand"
            else:
                # 12 -> "twelve thousand"
                if thousands < 100:
                    return f"{_two_digit_to_words(thousands)} thousand"
    # fallback: speak each digit (safe)
    return " ".join(list(str(n)))


# -------------------------
# METAR -> ATIS-style decoder (modified to use number_to_atis for heights)
# -------------------------
def _signed_digit_group(token):
    token = token.strip()
    if token.startswith("M"):
        return "minus " + " ".join(list(token[1:].zfill(2)))
    return " ".join(list(token.zfill(2)))

def decode_metar_to_text(raw_metar: str) -> str:
    if not raw_metar:
        return "No METAR provided."

    r = raw_metar.strip()
    parts = []

    # Station/time
    m_time = re.match(r"^\s*([A-Z]{4})\s+(\d{6}Z)", r)
    if m_time:
        icao = m_time.group(1)
        obs = m_time.group(2)
        parts.append(f"{icao} METAR observed at {obs}.")
    else:
        m_anytime = re.search(r"(\d{6}Z)", r)
        if m_anytime:
            parts.append(f"Observed at {m_anytime.group(1)}.")

    # CAVOK
    if re.search(r"\bCAVOK\b", r):
        parts.append("CAVOK. Visibility ten kilometers or more, no clouds below one five thousand feet, no significant weather.")

    # NOSIG
    nosig = bool(re.search(r"\bNOSIG\b", r))

    # Wind
    m_wind = re.search(r"\b(VRB|\d{3})(\d{2,3})(?:G(\d{2,3}))?KT\b", r)
    if m_wind:
        dir_part = m_wind.group(1)
        spd = m_wind.group(2)
        gust = m_wind.group(3)
        if dir_part == "VRB":
            wd_text = "variable"
        else:
            # keep heading as grouped digits for clarity in ATIS
            wd_text = " ".join(list(dir_part))
            wd_text = wd_text + " degrees"
        # wind speed spoken as normal number (e.g., "twelve knots")
        spd_text = str(int(spd)) + " knots"
        wind_phrase = f"Wind {wd_text} {spd_text}"
        if gust:
            gust_text = str(int(gust)) + " knots"
            wind_phrase += f", gusting {gust_text}"
        parts.append(wind_phrase + ".")
    else:
        if re.search(r"\b00000KT\b", r):
            parts.append("Wind calm.")

    # Visibility
    vis_km = None
    m_vis_m = re.search(r"\b(\d{4})\b", r)
    if m_vis_m:
        vis_m = int(m_vis_m.group(1))
        vis_km = 10 if vis_m >= 10000 else round(vis_m / 1000.0, 1)
    else:
        msm = re.search(r"(\d+\/\d+|\d+)(?=SM)", r)
        if msm:
            try:
                sm_str = msm.group(1)
                if "/" in sm_str:
                    a, b = sm_str.split("/")
                    val = float(a) / float(b)
                else:
                    val = float(sm_str)
                vis_km = round(val * 1.852, 1)
            except Exception:
                pass
    if vis_km is not None:
        vis_phrase = f"Visibility {int(vis_km) if vis_km == int(vis_km) else vis_km} kilometers."
        parts.append(vis_phrase)

    # Weather phenomena
    wx_tokens = re.findall(r"\b(-|\+)?(TS|SH|FZ|RA|SN|DZ|PL|GR|GS|SG|IC|BC|DR|BL|FG|BR|HZ|SQ|FC|TSG)\b", r)
    if wx_tokens:
        wx_phrases = []
        for sign, code in wx_tokens:
            if sign == "-": intensity = "light"
            elif sign == "+": intensity = "heavy"
            else: intensity = ""
            code_map = {
                "TS": "thunderstorm", "SH": "showers", "FZ": "freezing", "RA": "rain", "SN": "snow",
                "DZ": "drizzle", "PL": "ice pellets", "GR": "hail", "GS": "small hail/sleet",
                "SG": "snow grains", "IC": "ice crystals", "BC": "patches", "DR": "low drifting",
                "BL": "blowing", "FG": "fog", "BR": "mist", "HZ": "haze", "SQ": "squall",
                "FC": "funnel cloud/tornado", "TSG": "thunderstorm with small hail",
            }
            desc = code_map.get(code, code)
            phrase = f"{intensity + ' ' if intensity else ''}{desc}".strip()
            wx_phrases.append(phrase)
        parts.append("Weather: " + ", ".join(wx_phrases) + ".")

    # Clouds: use number_to_atis for heights (1200 -> "twelve hundred")
    clouds = re.findall(r"\b(FEW|SCT|BKN|OVC|NSC|NCD)(\d{3})\b", r)
    if clouds:
        cl_map = {"FEW":"few","SCT":"scattered","BKN":"broken","OVC":"overcast","NSC":"no significant clouds","NCD":"no clouds detected"}
        cl_phrases = []
        for cl, ht in clouds:
            feet = int(ht) * 100
            ht_spoken = number_to_atis(feet)  # new ATIS-style
            cl_phrases.append(f"{cl_map.get(cl,cl).capitalize()} at {ht_spoken} feet")
        parts.append(". ".join(cl_phrases) + ".")

    # Temp / dewpoint: spoken as plain numbers, negative with 'minus'
    m_temp = re.search(r"\b(M?\d{1,2})/(M?\d{1,2})\b", r)
    if m_temp:
        t_raw = m_temp.group(1); d_raw = m_temp.group(2)
        def temp_to_words(tok):
            if tok.startswith("M"):
                return "minus " + str(int(tok[1:]))
            return str(int(tok))
        t_spoken = temp_to_words(t_raw); d_spoken = temp_to_words(d_raw)
        parts.append(f"Temperature {t_spoken}, dewpoint {d_spoken}.")

    # QNH (keep digit-by-digit for clarity)
    m_q = re.search(r"\bQ(\d{4})\b", r)
    if m_q:
        q = m_q.group(1)
        digit_word = {"0":"zero","1":"one","2":"two","3":"three","4":"four","5":"five","6":"six","7":"seven","8":"eight","9":"nine"}
        q_spoken_words = " ".join(digit_word[d] for d in q)
        parts.append(f"QNH {q_spoken_words} hectopascals.")

    # RVR: try to detect Rxx/ group and convert thousands/hundreds
    m_rvr = re.search(r"\bR(\d{2,4})([LCR]?)\/([MP]?\d{4})(?:FT)?\b", r)
    if m_rvr:
        # group may vary; we will attempt to speak the RVR as ATIS-style number if possible
        rvr_value = m_rvr.group(3)
        try:
            rvr_int = int(re.sub(r"[^0-9]", "", rvr_value))
            # RVR in feet; if it's multiple of 100 -> use number_to_atis
            rvr_spoken = number_to_atis(rvr_int) if rvr_int % 100 == 0 else " ".join(list(str(rvr_int)))
            parts.append(f"Runway visual range {rvr_spoken} feet.")
        except Exception:
            parts.append("Runway visual range reported.")

    if nosig:
        parts.append("No significant changes expected.")

    if not parts:
        tokens = r.split()
        readout = " ".join(tokens[:12]) + ("..." if len(tokens) > 12 else "")
        return f"METAR readout: {readout}"

    explanation = " ".join(p.rstrip(".") + "." for p in parts)
    explanation = re.sub(r"\s+", " ", explanation).strip()
    return explanation


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        def fget(prefix, key, type_fn, default):
            v = request.form.get(f"{prefix}_{key}", None)
            return type_fn(v) if v not in (None, "") else default

        origin = {
            "elevation_ft": fget("orig", "elev", float, 200.0),
            "oat_c": fget("orig", "oat", float, 25.0),
            "qnh": fget("orig", "qnh", float, 1013.25),
            "runway_length_m": fget("orig", "rlen", float, 1400.0),
            "runway_heading_deg": fget("orig", "rhd", float, 90.0),
            "wind_dir_deg": fget("orig", "wdir", float, 0.0),
            "wind_kts": fget("orig", "wspd", float, 0.0),
            "vis_km": fget("orig", "vis", float, 10.0),
            "ceil_ft": fget("orig", "ceil", float, 10000.0),
            "precip": request.form.get("orig_precip", "off") == "on",
        }

        destination = {
            "name": request.form.get("dest_name", "").strip(),
            "icao": request.form.get("dest_icao", "").strip().upper(),
            "elevation_ft": fget("dest", "elev", float, 2500.0),
            "oat_c": fget("dest", "oat", float, 15.0),
            "qnh": fget("dest", "qnh", float, 1013.25),
            "runway_length_m": fget("dest", "rlen", float, 1200.0),
            "runway_heading_deg": fget("dest", "rhd", float, 85.0),
            "wind_dir_deg": fget("dest", "wdir", float, 0.0),
            "wind_kts": fget("dest", "wspd", float, 0.0),
            "vis_km": fget("dest", "vis", float, 10.0),
            "ceil_ft": fget("dest", "ceil", float, 10000.0),
            "gusts_kts": fget("dest", "gust", float, 0.0),
            "precip": request.form.get("dest_precip", "off") == "on",
        }

        manual_metar = request.form.get("manual_metar", "").strip()

        alternates = []
        if request.form.get("alt_rlen"):
            alt = {
                "name": request.form.get("alt_name", "ALT1"),
                "elevation_ft": fget("alt", "elev", float, 1000.0),
                "oat_c": fget("alt", "oat", float, 25.0),
                "qnh": fget("alt", "qnh", float, 1013.25),
                "runway_length_m": fget("alt", "rlen", float, 1600.0),
                "distance_nm": fget("alt", "dist", float, 45.0),
            }
            alternates.append(alt)

        flight = {
            "origin": origin,
            "destination": destination,
            "alternates": alternates,
            "fuel_kg": fget("flight", "fuel", float, 200.0),
            "planned_flight_time_hr": fget("flight", "time", float, 1.5),
            "required_reserve_hr": fget("flight", "reserve", float, 0.5),
        }

        ac_perf = {
            "takeoff_run_available_m": 800.0,
            "landing_distance_m": 700.0,
            "fuel_burn_kg_per_hr": 70.0,
            "cruise_speed_kts": 120.0,
        }

        metar_result = None
        if manual_metar:
            parsed = simple_parse_metar(manual_metar)
            if parsed.get("wind_dir_deg") is not None:
                destination["wind_dir_deg"] = parsed["wind_dir_deg"]
                destination["wind_kts"] = parsed["wind_kts"]
                destination["gusts_kts"] = parsed["gusts_kts"]
            if parsed.get("vis_km") is not None:
                destination["vis_km"] = parsed["vis_km"]
            metar_result = {"icao": destination.get("icao") or "MANUAL", "raw_text": manual_metar, "observation_time": None, "source": "MANUAL", "error": None}
        elif destination.get("icao"):
            metar_result = fetch_metar_for_icao(destination["icao"], hours_before_now=3)
            if metar_result and not metar_result.get("error") and metar_result.get("raw_text"):
                parsed = simple_parse_metar(metar_result["raw_text"])
                if parsed.get("wind_dir_deg") is not None:
                    destination["wind_dir_deg"] = parsed["wind_dir_deg"]
                    destination["wind_kts"] = parsed["wind_kts"]
                    destination["gusts_kts"] = parsed["gusts_kts"]
                if parsed.get("vis_km") is not None:
                    destination["vis_km"] = parsed["vis_km"]

        fds = FlightDecisionSupport(ac_perf)
        assessment = fds.evaluate(flight)

        dest_label = destination.get("name") or destination.get("icao") or "destination"
        speech_parts = [
            f"Flight decision briefing for {dest_label}.",
            f"Recommendation: {assessment['recommendation']}.",
            f"Destination density altitude is {assessment['density_altitude_destination_ft']} feet.",
            f"Estimated required landing distance is {assessment['required_landing_m_est']} meters.",
            f"Destination runway length is {assessment['destination_runway_length_m']} meters.",
            f"Weather risk score is {assessment['destination_weather_risk']}.",
            f"Fuel endurance is {assessment['endurance_hr']} hours and fuel margin over reserve is {assessment['fuel_margin_hr_over_reserve']} hours.",
            f"Aggregate risk score is {assessment['aggregate_risk_score']}.",
        ]

        metar_audio_file = None
        metar_decoded_audio_file = None
        metar_explanation = None

        if metar_result:
            if metar_result.get("error"):
                speech_parts.append(f"METAR lookup error: {metar_result['error']}.")
            else:
                raw = metar_result.get("raw_text", "")
                obs = metar_result.get("observation_time", "") or ""
                speech_parts.append(f"Latest METAR: {raw}.")
                if gTTS and raw:
                    metar_audio_file = generate_audio_briefing(f"METAR: {raw}", filename_prefix="metar_raw")
                metar_explanation = decode_metar_to_text(raw)
                if gTTS and metar_explanation:
                    metar_decoded_audio_file = generate_audio_briefing(metar_explanation, filename_prefix="metar_decoded")
                if metar_explanation:
                    first_sentence = metar_explanation.split(".")[0].strip()
                    if first_sentence:
                        speech_parts.append(f"Decoded METAR: {first_sentence}.")

        speech_text = " ".join(speech_parts)
        audio_file = generate_audio_briefing(speech_text) if gTTS else None

        return render_template("result.html",
                               assessment=assessment,
                               flight=flight,
                               audio_file=audio_file,
                               metar_result=metar_result,
                               metar_audio_file=metar_audio_file,
                               metar_explanation=metar_explanation,
                               metar_decoded_audio_file=metar_decoded_audio_file,
                               gtts_available=(gTTS is not None))
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
