"""gta6-twitch-ccu-forecast-01 v13: unlock-aligned log-space ensemble.

Centre: three kernel members on unlock-time curves, blended 0.5/0.3/0.2
(kernel-weighted local-linear ridge, bw by size; kernel mean bw by size;
wider kernel mean bw 2.5). Target corrections from the launch's own
14-day pre-release diurnal (dip deepening, inflation shift gated by own
midday dip, early-dip echo, small-launch evening lift), plus v13:
the early-dip echo responds superlinearly to the target's own early
diurnal fall (echo x sqrt(F/0.5), F = diurnal[0] - min(diurnal[0:5])),
and launches whose pre-release diurnal is flat get a small level lift
(+0.05 x (1.8 - range)/1.8). Per-hour spread is size-adaptive:
SIGMA x 1.15 for small launches, x 1.0 for big ones, linear between
log p1 9 and 12. Standard library only, deterministic.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
# Per-hour spread (log units) of the v5 centre's residuals, x1.15 floored 0.15.
SIGMA = [0.26, 0.25, 0.3, 0.21, 0.18, 0.16, 0.18, 0.2, 0.22, 0.29, 0.32,
         0.32, 0.25, 0.24, 0.25, 0.26, 0.23, 0.17, 0.15, 0.15, 0.15, 0.15,
         0.15]

W_LOCLIN = 0.5
W_MEAN1 = 0.3
W_MEAN2 = 0.2
BW_MEAN2 = 2.5
LAM_LOCLIN = 0.6
SLOPE_DAMP = 0.3
BIG_THRESHOLD = 10.0
BW_MEAN1_BIG = 1.0
BW_MEAN1_SMALL = 1.5
BW_LOCLIN_BIG = 1.8
BW_LOCLIN_SMALL = 1.2
WEEKDAY_FACTOR = 0.75

DIP_COEF = 0.7
ECHO_COEF = 0.5
INFL_COEF = 0.02
INFL_CAP = 10.0
INFL_FALLBACK = 8.0
INFL_GATE_DENOM = 0.1
EVENING_COEF = 0.2
EVENING_REF = 11.0
EARLY_SIZE_COEF = 0.2
SIZE_REF = 11.5
EARLY_IDX = (0, 1, 2)
MID_IDX = (9, 10, 11, 12, 13)
EVENING_IDX = (15, 16, 17, 18, 19, 20, 21, 22)

# v13: convex early-dip echo response to the target's own early fall.
ECHO_F0 = 0.5
ECHO_EXP = 0.5
ECHO_CAP = 2.0
# v13: flat-diurnal level lift.
FLAT_REF = 1.8
FLAT_COEF = 0.05
# v13: size-adaptive spread multiplier (log p1 9 -> 12).
SPREAD_SMALL = 1.15
SPREAD_BIG = 1.0
SPREAD_L1 = 9.0
SPREAD_L2 = 12.0

# Simultaneous-global launches: name -> unlock hour (UTC) on release day.
SIMULTANEOUS_UNLOCK = {
    "Cyberpunk 2077": 0,
    "Black Myth: Wukong": 2,
    "Battlefield 6": 14,
    "Hollow Knight: Silksong": 14,
    "007 First Light": 14,
}


def round2(x):
    """x rounded to 2 decimals, half up; whole numbers come back as ints."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_ts(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc)


def dst_active(day):
    """US daylight saving in effect on `day` (a date). DST runs from
    the second Sunday of March to the first Sunday of November; a
    launch ON the second Sunday of March counts as DST, one ON the
    first Sunday of November does not."""
    mar1 = datetime(day.year, 3, 1, tzinfo=timezone.utc).date()
    start = mar1.replace(day=1 + ((6 - mar1.weekday()) % 7) + 7)
    nov1 = datetime(day.year, 11, 1, tzinfo=timezone.utc).date()
    end = nov1.replace(day=1 + ((6 - nov1.weekday()) % 7))
    return start <= day < end


def smooth3(values):
    """3-point moving average; edges average the two available points."""
    out = []
    n = len(values)
    for i in range(n):
        if i == 0:
            out.append((values[0] + values[1]) / 2)
        elif i == n - 1:
            out.append((values[-2] + values[-1]) / 2)
        else:
            out.append((values[i - 1] + values[i] + values[i + 1]) / 3)
    return out


def rolling_unlock(day_start):
    """US East Coast midnight unlock hour (UTC) for a rolling launch."""
    return 4 if dst_active(parse_ts(day_start).date()) else 5


def target_unlock(case):
    return rolling_unlock(case["day_start"])


def comp_unlock(comp):
    """Unlock UTC hour for a comparable launch, or None to exclude it
    from the pool (mixed console+PC-simultaneous launches, and
    simultaneous launches whose unlock hour we do not know). The
    exclusion strings are checked BEFORE the simultaneous map, so a
    mapped launch that is early-access-contaminated is still out."""
    unlock = comp["unlock"]
    if "early-access-contaminated" in unlock:
        return None
    if "Tier: weak" in unlock:
        return None
    if "Tier: target-history" in unlock:
        return None
    if comp["name"] in SIMULTANEOUS_UNLOCK:
        return SIMULTANEOUS_UNLOCK[comp["name"]]
    text = unlock.lower()
    if "simultaneous" in text or "single global" in text:
        return None
    if "mixed" in text:
        return None
    return rolling_unlock(comp["day_start"])


def build_pool(case):
    """Comparable launches usable for training: dicts with (name, log
    hour-1 print, 29-entry unlock-relative log-ratio curve, unlock
    hour, pre-release hourly points for the diurnal). The curve has
    index 0 == 5h before unlock; entry j is log(viewers[UTC offset
    clamp(unlock + j - 5, 0, 23)] / viewers[UTC offset 0]). A
    comparable is dropped if any launch-day hour 0..23 is missing or
    has fewer than 1 viewer."""
    pool = []
    for comp in case["comparables"]:
        unlock = comp_unlock(comp)
        if unlock is None:
            continue
        ds = parse_ts(comp["day_start"])
        by_offset = {}
        by_offset_ch = {}
        pre = []
        for point in comp["hourly"]:
            off = int((parse_ts(point["from"]) - ds).total_seconds()
                      // 3600)
            by_offset[off] = point["viewers"]
            if point.get("channels"):
                by_offset_ch[off] = point["channels"]
            if off < 0:
                pre.append(point)
        if any(o not in by_offset or by_offset[o] < 1
               for o in range(24)):
            continue
        v0 = max(1, by_offset[0])
        logp1 = math.log(v0)
        ch0 = max(1, by_offset_ch.get(0, 1))
        logch = math.log(ch0)
        curve = []
        for j in range(29):
            off = min(23, max(0, unlock + j - 5))
            curve.append(math.log(max(1, by_offset[off]) / v0))
        pool.append({"name": comp["name"], "logp1": logp1,
                     "logch": logch, "logvpc": logp1 - logch,
                     "curve": curve, "unlock": unlock, "pre": pre,
                     "weekend": parse_ts(comp["day_start"]).weekday() >= 5,
                     "day_start": comp["day_start"]})
    return pool


def weekday_factor(comp, target_weekend):
    """1.0 when the comparable's release day falls on the same
    weekend/weekday side as the target's, else WEEKDAY_FACTOR."""
    if target_weekend is None:
        return 1.0
    return 1.0 if comp["weekend"] == target_weekend else WEEKDAY_FACTOR


def kernel_weight(logp1, lp, bw):
    d = (logp1 - lp) / bw
    return math.exp(-0.5 * d * d)


def loclin_predict(pool, pos, lp, bw, target_weekend=None):
    """Kernel-weighted local-linear ridge of the log-ratio at curve
    position `pos` on log hour-1 print. The slope is damped outside
    the comparables' log-print range: the distance beyond the range
    is shrunk by SLOPE_DAMP, so a far bigger launch than anything
    seen does not extrapolate linearly forever."""
    if not pool:
        return 0.0
    num_w = 0.0
    mx = 0.0
    my = 0.0
    terms = []
    for comp in pool:
        w = kernel_weight(comp["logp1"], lp, bw) \
            * weekday_factor(comp, target_weekend)
        terms.append((w, comp["logp1"], comp["curve"][pos]))
        num_w += w
        mx += w * comp["logp1"]
        my += w * comp["curve"][pos]
    if num_w <= 0:
        return sum(c["curve"][pos] for c in pool) / len(pool)
    mx /= num_w
    my /= num_w
    sxx = 0.0
    sxy = 0.0
    for w, x, y in terms:
        sxx += w * (x - mx) * (x - mx)
        sxy += w * (x - mx) * (y - my)
    slope = sxy / (sxx + LAM_LOCLIN)
    lo = min(c["logp1"] for c in pool)
    hi = max(c["logp1"] for c in pool)
    clamped = min(hi, max(lo, lp))
    dx = (clamped - mx) + SLOPE_DAMP * (lp - clamped)
    return my + slope * dx


def kernel_mean_predict(pool, pos, lp, bw, target_weekend=None):
    """Kernel-weighted mean of the comparables' log-ratio at `pos`."""
    num = 0.0
    den = 0.0
    for comp in pool:
        w = kernel_weight(comp["logp1"], lp, bw) \
            * weekday_factor(comp, target_weekend)
        num += w * comp["curve"][pos]
        den += w
    if den <= 0:
        if not pool:
            return 0.0
        return sum(c["curve"][pos] for c in pool) / len(pool)
    return num / den


def ensemble_centre(pool, lp, unlock, bw_mean1, bw_loclin,
                    target_weekend=None):
    """Blended centre: 23 log-ratio values, one per forecast hour
    2..24, for a launch with hour-1 log print `lp` and unlock hour
    `unlock`. Forecast hour h uses curve position h + 4 - unlock."""
    centre = []
    for h in range(2, 25):
        pos = h + 4 - unlock
        pos = min(28, max(0, pos))
        m1 = loclin_predict(pool, pos, lp, bw_loclin, target_weekend)
        m2 = kernel_mean_predict(pool, pos, lp, bw_mean1,
                                 target_weekend)
        m3 = kernel_mean_predict(pool, pos, lp, BW_MEAN2,
                                 target_weekend)
        centre.append(W_LOCLIN * m1 + W_MEAN1 * m2 + W_MEAN2 * m3)
    return centre


def diurnal_profile(history_hourly, day_start):
    """Mean log viewers by UTC hour over the 14 pre-release days
    (offsets -336..-1). Each UTC hour needs at least 3 points or
    the profile is None. The result is passed through a 3-point
    moving average twice (a 5-tap triangular smooth)."""
    if not history_hourly:
        return None
    ds = parse_ts(day_start)
    sums = [0.0] * 24
    counts = [0] * 24
    for point in history_hourly:
        if point.get("viewers") is None:
            continue
        dt = parse_ts(point["from"])
        off = int((dt - ds).total_seconds() // 3600)
        if -336 <= off < 0:
            sums[dt.hour] += math.log(max(1, point["viewers"]))
            counts[dt.hour] += 1
    if any(c < 3 for c in counts):
        return None
    profile = [sums[h] / counts[h] for h in range(24)]
    return smooth3(smooth3(profile))


def dip_corrections(centre, logp1, diurnal, rolling=True):
    """Target-specific corrections from the launch's own pre-release
    diurnal profile. dip[i] = diurnal[i+1] - diurnal[0]. The early
    trough hours (2-4) get one-sided dip deepening plus the early
    size gate; the mid trough hours (11-15) get deepening, the
    inflation shift gated by the launch's own midday dip, and the
    early-dip echo; hours 17-24 get the small-launch evening lift.
    dip_early / dip_mid are the means of the NEGATIVE parts of the
    dips over the early / mid sets. When the diurnal is missing, the
    dips are zero, so only the size gate (early) and the evening
    lift apply; the inflation fallback then multiplies a zero gate.
    The size gate models the pre-unlock lull, so it applies only to
    rolling launches (rolling=False -> gate 0).
    v13: the early deepening responds superlinearly to the target's
    own early diurnal fall (x sqrt(F/0.5), F = diurnal[0] -
    min(diurnal[0:5])), and a flat pre-release diurnal lifts the
    whole centre a little (the pool-average dip overstates a launch
    with no dip of its own)."""
    out = list(centre)
    if diurnal is None:
        dip = [0.0] * 23
        infl = INFL_FALLBACK
        echo_mult = 1.0
        lift = 0.0
    else:
        dip = [diurnal[i + 1] - diurnal[0] for i in range(23)]
        infl = min(logp1 - diurnal[0], INFL_CAP)
        early_fall = diurnal[0] - min(diurnal[0:5])
        echo_mult = min((max(early_fall, 0.0) / ECHO_F0) ** ECHO_EXP,
                        ECHO_CAP)
        span = max(diurnal) - min(diurnal)
        lift = FLAT_COEF * max(0.0, (FLAT_REF - span) / FLAT_REF)
    dip_early = sum(min(0.0, dip[i]) for i in EARLY_IDX) / len(EARLY_IDX)
    dip_mid = sum(min(0.0, dip[i]) for i in MID_IDX) / len(MID_IDX)
    gate_mid = min(1.0, max(0.0, -dip_mid / INFL_GATE_DENOM))
    if rolling:
        size_gate = min(1.0, max(0.0, logp1 - SIZE_REF))
    else:
        size_gate = 0.0
    evening_lift = EVENING_COEF * max(0.0, EVENING_REF - logp1)
    for i in EARLY_IDX:
        out[i] += DIP_COEF * min(0.0, dip[i]) * echo_mult \
            - EARLY_SIZE_COEF * size_gate
    for i in MID_IDX:
        out[i] += DIP_COEF * min(0.0, dip[i]) \
            - INFL_COEF * infl * gate_mid + ECHO_COEF * dip_early
    for i in EVENING_IDX:
        out[i] += evening_lift
    for i in range(23):
        out[i] += lift
    return out


def full_centre(pool, logp1, unlock, pre_points, day_start,
                target_weekend=None, rolling=True,
                bw_mean1=BW_MEAN1_SMALL, bw_loclin=BW_LOCLIN_BIG):
    """Ensemble centre plus dip corrections for one launch. The
    diurnal comes from the launch's own pre-release hourly points
    (history_hourly for the target, the comparable's own pre-release
    hourly rows for a comparable)."""
    diurnal = diurnal_profile(pre_points, day_start)
    centre = ensemble_centre(pool, logp1, unlock, bw_mean1, bw_loclin,
                             target_weekend)
    return dip_corrections(centre, logp1, diurnal, rolling)


def spread_mult(logp1):
    """v13: size-adaptive spread multiplier. Small launches' curves are
    wilder, so they need wider intervals; big launches keep the base
    spread. 1.15 at log p1 <= 9, 1.0 at log p1 >= 12, linear between."""
    if logp1 <= SPREAD_L1:
        return SPREAD_SMALL
    if logp1 >= SPREAD_L2:
        return SPREAD_BIG
    return SPREAD_SMALL + (SPREAD_BIG - SPREAD_SMALL) * (
        logp1 - SPREAD_L1) / (SPREAD_L2 - SPREAD_L1)


def forecast_case(case, case_id):
    if case["levels"] != LEVELS:
        raise ValueError(
            f"{case_id} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{case_id} has no hour-1 print")
    p1 = max(1, first["viewers"])
    logp1 = math.log(p1)
    pool = build_pool(case)
    unlock = target_unlock(case)
    weekend = parse_ts(case["day_start"]).weekday() >= 5
    # A3 size-adaptive bandwidths: big launches get a narrow mean
    # bandwidth and a wide local-linear one; small launches the
    # reverse.
    big = logp1 >= BIG_THRESHOLD
    bw_mean1 = BW_MEAN1_BIG if big else BW_MEAN1_SMALL
    bw_loclin = BW_LOCLIN_BIG if big else BW_LOCLIN_SMALL
    # The early-hours size gate models the pre-unlock lull, so it
    # applies only to rolling-midnight launches. A launch named in
    # the simultaneous-unlock map unlocked everywhere at once.
    name = case.get("game", {}).get("name", "")
    rolling = name not in SIMULTANEOUS_UNLOCK
    if pool:
        centre = full_centre(pool, logp1, unlock,
                             case.get("history_hourly") or [],
                             case["day_start"], weekend, rolling,
                             bw_mean1, bw_loclin)
    else:
        centre = [0.0] * 23
    mult = spread_mult(logp1)
    rows = []
    for i, h in enumerate(range(2, 25)):
        mu = logp1 + centre[i]
        qs = [round2(math.exp(mu + Z[j] * SIGMA[i] * mult))
              for j in range(5)]
        qs = [min(1e9, max(0.01, q)) for q in qs]
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"hour": h, "q": qs})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
