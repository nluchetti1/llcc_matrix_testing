"""
ivt_maps.py — CW3E-style spatial Integrated Vapor Transport maps.

Companion to ivt.py. That module does the column integral for the point/matrix
values; this one does the 2D field: fetch, integrate on the grid, render, and hand
back a {model: {fh_key: png_path}} dict for history.json.

WHAT THE PROBES ESTABLISHED (2026-08-31, verified live -- do not re-derive):

  * NOMADS GRIB filter with `subregion` beats S3 byte-range by ~32x for this job.
    GFS SPFH+U+V over the wide domain is 1.5 MB/step against 48.4 MB/step pulling
    whole global messages off S3. Server-side subsetting wins because IVT needs many
    levels over a small footprint.

  * GFS serves SPFH on all 17 integration levels. Use it directly.

  * RAP and NAM DO NOT serve SPFH through the filter. They return HTTP 200 with
    valid GRIB containing u and v and NO moisture at all -- which would integrate to
    IVT ~= 0 and read as a bone-dry airmass. Both carry RH+TMP on all 17 levels, so
    they go through the RH route. This is why _grid_moisture() refuses to guess:
    a missing moisture field must raise, never silently yield zeros.

  * HRRR is NOT reachable here. filter_hrrr_sub.pl is the SUB-HOURLY filter and
    rejects wrfprs filenames outright ("Filename does not match legal pattern:
    hrrr.t..z.wrfsubhf...grib2$"). HRRR would have to come from HRRR_AWS_ROOT with
    no spatial subsetting. Since IVT is synoptic and HRRR tops out at 48 h, it is
    deliberately excluded rather than paid for.

  * ECMWF open data has q but NO subsetting -- ~16 MB/step global vs GFS's 1.5 MB.
    Left out of the default set; ECMWF_ENABLED below turns it on if wanted.
"""

import datetime
import logging
import math
import os
import tempfile
import time

import numpy as np
import pygrib
import requests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from ivt import G, EPSILON, IVT_TOP_HPA, cw3e_ivt_cmap

IVT_MAPS_ENABLED = True
IVT_MAPS_DIR = "./maps/ivt"

# Wide enough that IVT reads as the synoptic transport field it is. The pipeline's
# FL_DOMAIN (24.5-31N) is far too tight -- at that zoom an AR or a tropical plume
# fills the frame uniformly and the gradient, which is the whole signal, is invisible.
IVT_DOMAIN = {"left": -100.0, "right": -60.0, "top": 45.0, "bottom": 15.0}

IVT_LEVELS = [1000, 975, 950, 925, 900, 850, 800, 750, 700,
              650, 600, 550, 500, 450, 400, 350, 300]

# Per-model fetch recipe. `moisture` is "spfh" or "rh" and is NOT negotiable per the
# probe results above.
IVT_MODELS = {
    "gfs": {
        "enabled": True,
        "filter": "filter_gfs_0p25.pl",
        "file": lambda c, f: f"gfs.t{c}z.pgrb2.0p25.f{f:03d}",
        "dir": lambda d, c: f"%2Fgfs.{d}%2F{c}%2Fatmos",
        "moisture": "spfh",
        "cycles": [0, 6, 12, 18],
        "latency_h": 4,
        "steps": list(range(0, 181, 6)),   # f000-f180 6-hourly, matching CW3E
    },
    "rap": {
        "enabled": True,
        "filter": "filter_rap.pl",
        "file": lambda c, f: f"rap.t{c}z.awp130pgrbf{f:02d}.grib2",
        "dir": lambda d, c: f"%2Frap.{d}",
        "moisture": "rh",                  # SPFH silently absent -- see header
        "cycles": list(range(24)),
        "latency_h": 2,
        "steps": list(range(0, 22, 3)),
    },
    "nam": {
        "enabled": True,
        "filter": "filter_nam.pl",
        "file": lambda c, f: f"nam.t{c}z.awphys{f:02d}.tm00.grib2",
        "dir": lambda d, c: f"%2Fnam.{d}",
        "moisture": "rh",                  # SPFH silently absent -- see header
        "cycles": [0, 6, 12, 18],
        "latency_h": 3,
        "steps": list(range(0, 85, 6)),
    },
}

# NOMADS politeness. Matches the main pipeline's posture: one connection, a real
# pause between requests. A 31-step GFS set is ~31 * (1.5 + 4) = ~170 s.
IVT_REQUEST_PAUSE_S = 4.0
IVT_CONNECT_TIMEOUT = 15
IVT_READ_TIMEOUT = 90
IVT_ATTEMPTS = 3
IVT_BUDGET_S = 900          # whole-job budget; returns what it has when spent
IVT_UA = ("Mozilla/5.0 (X11; Linux x86_64; rv:120.0) "
          "Gecko/20100101 Firefox/120.0")

# Rendering
IVT_VECTOR_STRIDE = 6       # subsample factor for the transport vectors
IVT_MIN_SHADE = 250.0       # below this the field is transparent (CW3E convention)
MSLP_INTERVAL_HPA = 4


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
def _build_url(model, date_str, cycle, fh):
    cfg = IVT_MODELS[model]
    lev = "".join(f"&lev_{lv}_mb=on" for lv in IVT_LEVELS)
    if cfg["moisture"] == "spfh":
        var = "&var_SPFH=on"
    else:
        var = "&var_RH=on&var_TMP=on"
    var += "&var_UGRD=on&var_VGRD=on&var_PRES=on&var_PRMSL=on"
    # PRMSL lives on mean sea level, PRES on the surface; both need their level flags
    # or the filter returns the isobaric-only subset and the MSLP contours vanish.
    var += "&lev_mean_sea_level=on&lev_surface=on"
    region = (f"&subregion=&leftlon={IVT_DOMAIN['left']}&rightlon={IVT_DOMAIN['right']}"
              f"&toplat={IVT_DOMAIN['top']}&bottomlat={IVT_DOMAIN['bottom']}")
    return (f"https://nomads.ncep.noaa.gov/cgi-bin/{cfg['filter']}"
            f"?file={cfg['file'](cycle, fh)}{lev}{var}{region}"
            f"&dir={cfg['dir'](date_str, cycle)}")


def _fetch_grib(session, url, tag=""):
    """GET a filter URL, verify it is really GRIB, write to a temp file.

    The filter answers HTTP 200 with an HTML error page on a bad request, so status
    alone proves nothing -- the magic-bytes check is load-bearing, not defensive
    padding. Returns a path the caller must delete, or None.
    """
    for attempt in range(1, IVT_ATTEMPTS + 1):
        try:
            r = session.get(url, timeout=(IVT_CONNECT_TIMEOUT, IVT_READ_TIMEOUT))
        except Exception as e:
            logging.warning(f"[IVT] {tag} attempt {attempt} request error: {e}")
            time.sleep(IVT_REQUEST_PAUSE_S * attempt)
            continue
        if r.status_code != 200:
            logging.warning(f"[IVT] {tag} attempt {attempt} HTTP {r.status_code}")
            time.sleep(IVT_REQUEST_PAUSE_S * attempt)
            continue
        if not r.content.startswith(b"GRIB"):
            snippet = r.content[:160].decode("utf-8", "replace").replace("\n", " ")
            logging.warning(f"[IVT] {tag} 200 but not GRIB: {snippet}")
            return None       # a malformed request will not fix itself on retry
        fd, path = tempfile.mkstemp(suffix=".grib2", prefix="ivt_")
        with os.fdopen(fd, "wb") as f:
            f.write(r.content)
        return path
    return None


def determine_cycle(session, model):
    """Walk back from the most recent nominal cycle until one answers with real GRIB.
    Probes a single level of a single variable so the check costs almost nothing."""
    cfg = IVT_MODELS[model]
    now = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=cfg["latency_h"])
    for back in range(0, 5):
        t = now - datetime.timedelta(hours=6 * back)
        cyc = max([c for c in cfg["cycles"] if c <= t.hour], default=None)
        if cyc is None:
            t = t - datetime.timedelta(days=1)
            cyc = cfg["cycles"][-1]
        date_str, cycle = t.strftime("%Y%m%d"), f"{cyc:02d}"
        probe = (f"https://nomads.ncep.noaa.gov/cgi-bin/{cfg['filter']}"
                 f"?file={cfg['file'](cycle, 0)}&lev_500_mb=on&var_UGRD=on"
                 f"&subregion=&leftlon=-81&rightlon=-80&toplat=29&bottomlat=28"
                 f"&dir={cfg['dir'](date_str, cycle)}")
        try:
            r = session.get(probe, timeout=(IVT_CONNECT_TIMEOUT, 30))
            if r.status_code == 200 and r.content.startswith(b"GRIB"):
                logging.info(f"[IVT] {model}: using cycle {date_str} {cycle}z")
                return date_str, cycle
        except Exception:
            pass
        time.sleep(1.0)
    logging.error(f"[IVT] {model}: no usable cycle found")
    return None, None


# ---------------------------------------------------------------------------
# Grid integration
# ---------------------------------------------------------------------------
def _sat_vap_hpa(tc):
    return 6.112 * np.exp((17.67 * tc) / (tc + 243.5))


def _grid_moisture(fields, level, route):
    """Specific humidity (kg/kg) on the grid for one level.

    Raises on a missing field rather than returning zeros. RAP and NAM return
    perfectly valid GRIB with no moisture in it at all when SPFH is requested; a
    silent zero there produces a smooth, plausible, completely wrong dry map.
    """
    if route == "spfh":
        q = fields.get(("q", level))
        if q is None:
            raise KeyError(f"no specific humidity at {level} hPa")
        return q
    rh, t = fields.get(("r", level)), fields.get(("t", level))
    if rh is None or t is None:
        raise KeyError(f"no RH/TMP pair at {level} hPa")
    tc = t - 273.15
    e = _sat_vap_hpa(tc) * np.clip(rh, 0.0, 100.0) / 100.0
    e = np.minimum(e, level * 0.999)          # guard the e >= p singularity
    w = EPSILON * e / (level - e)
    return w / (1.0 + w)


def grid_ivt(filepath, route):
    """Integrate IVT across the whole grid.

    Returns (lons, lats, ivt_mag, ivt_u, ivt_v, mslp_hpa, levels_used).
    Vertical integration is the same trapezoid ivt.column_ivt() uses, vectorised.
    """
    fields, lats, lons, sfc_pres, mslp = {}, None, None, None, None
    grbs = pygrib.open(filepath)
    for g in grbs:
        sn = getattr(g, "shortName", "")
        tl = getattr(g, "typeOfLevel", "")
        if lats is None:
            lats, lons = g.latlons()
        if tl == "isobaricInhPa" and g.level in IVT_LEVELS:
            if sn in ("q", "r", "t", "u", "v"):
                fields[(sn, g.level)] = np.asarray(g.values, dtype=np.float64)
        elif sn in ("sp", "pres") and tl == "surface":
            sfc_pres = np.asarray(g.values, dtype=np.float64) / 100.0   # Pa -> hPa
        elif sn in ("prmsl", "msl", "mslet"):
            mslp = np.asarray(g.values, dtype=np.float64) / 100.0
    grbs.close()

    if lats is None:
        raise ValueError("no fields decoded")

    # Levels present with a complete (moisture, u, v) triple. Descending pressure.
    usable = []
    for lv in sorted(IVT_LEVELS, reverse=True):
        if ("u", lv) not in fields or ("v", lv) not in fields:
            continue
        try:
            q = _grid_moisture(fields, lv, route)
        except KeyError:
            continue
        usable.append((lv, q, fields[("u", lv)], fields[("v", lv)]))

    if len(usable) < 6:
        raise ValueError(f"only {len(usable)} complete levels; refusing to integrate")

    shape = usable[0][1].shape
    if sfc_pres is None:
        # No surface pressure: assume everything at or below 1000 hPa is valid. Over
        # the Florida/W-Atlantic domain this is nearly all ocean, so the error is
        # small, but it WILL overstate IVT over terrain.
        sfc_pres = np.full(shape, 1013.0)
        logging.warning("[IVT] no surface pressure field; below-ground levels not masked")

    acc_u = np.zeros(shape)
    acc_v = np.zeros(shape)
    for (p1, q1, u1, v1), (p2, q2, u2, v2) in zip(usable[:-1], usable[1:]):
        # Partial bottom slab: clip the lower bound at the surface so a level buried
        # underground contributes nothing and the slab straddling the surface
        # contributes only its above-ground fraction.
        p_low = np.minimum(np.full(shape, float(p1)), sfc_pres)
        dp = np.clip(p_low - p2, 0.0, None) * 100.0        # hPa -> Pa
        acc_u += 0.5 * (q1 * u1 + q2 * u2) * dp
        acc_v += 0.5 * (q1 * v1 + q2 * v2) * dp

    ivt_u, ivt_v = acc_u / G, acc_v / G
    return (lons, lats, np.hypot(ivt_u, ivt_v), ivt_u, ivt_v, mslp,
            [lv for lv, _, _, _ in usable])


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def render_ivt_map(model, lons, lats, mag, ivt_u, ivt_v, mslp,
                   cycle_label, fh, valid_str, levels_used):
    """One CW3E-style IVT panel: shaded magnitude, transport vectors, MSLP contours."""
    try:
        out_dir = os.path.join(IVT_MAPS_DIR, model)
        os.makedirs(out_dir, exist_ok=True)
        cmap, norm = cw3e_ivt_cmap()
        proj = ccrs.PlateCarree()

        fig = plt.figure(figsize=(8.0, 6.4), dpi=115)
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        ax.set_extent([IVT_DOMAIN["left"], IVT_DOMAIN["right"],
                       IVT_DOMAIN["bottom"], IVT_DOMAIN["top"]], crs=proj)

        # Light basemap, unlike the dark reflectivity maps: the CW3E ramp is a warm
        # yellow-to-purple sequence and needs a pale ground to read against.
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#eef2f7", zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#e6e6e0", zorder=0)

        plot_lons = np.where(lons > 180, lons - 360.0, lons)
        masked = np.ma.masked_less(mag, IVT_MIN_SHADE)
        mesh = ax.pcolormesh(plot_lons, lats, masked, cmap=cmap, norm=norm,
                             shading="auto", transform=proj, zorder=1)

        if mslp is not None:
            lo = math.floor(np.nanmin(mslp) / MSLP_INTERVAL_HPA) * MSLP_INTERVAL_HPA
            hi = math.ceil(np.nanmax(mslp) / MSLP_INTERVAL_HPA) * MSLP_INTERVAL_HPA
            cs = ax.contour(plot_lons, lats, mslp,
                            levels=np.arange(lo, hi + 1, MSLP_INTERVAL_HPA),
                            colors="#111827", linewidths=0.6, transform=proj, zorder=4)
            ax.clabel(cs, inline=True, fontsize=5.5, fmt="%d")

        # Vectors show the DIRECTION OF TRANSPORT, matching the CW3E convention --
        # the way the vapour is moving, not the meteorological FROM direction used
        # everywhere else in the dashboard. Subsampled or the field is unreadable.
        s = IVT_VECTOR_STRIDE
        vec_mask = mag[::s, ::s] >= IVT_MIN_SHADE
        if np.any(vec_mask):
            ax.quiver(plot_lons[::s, ::s][vec_mask], lats[::s, ::s][vec_mask],
                      ivt_u[::s, ::s][vec_mask], ivt_v[::s, ::s][vec_mask],
                      transform=proj, zorder=5, scale=12000, width=0.0022,
                      color="#1f2937", alpha=0.75)

        ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                       edgecolor="#374151", linewidth=0.7, zorder=6)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                       edgecolor="#6b7280", linewidth=0.5, zorder=6)
        ax.add_feature(cfeature.STATES.with_scale("50m"),
                       edgecolor="#9ca3af", linewidth=0.4, zorder=6)

        # The Cape, so the launch site is locatable on a domain this wide.
        ax.plot(-80.556, 28.468, marker="*", markersize=11, color="#16a34a",
                markeredgecolor="black", markeredgewidth=0.7, transform=proj, zorder=8)

        span = f"{max(levels_used)}\u2013{min(levels_used)} hPa"
        ax.set_title(f"{model.upper()} IVT (kg m\u207b\u00b9 s\u207b\u00b9), transport vectors, "
                     f"MSLP (hPa)\nInit {cycle_label}   F{fh:03d}   Valid {valid_str}   "
                     f"[{len(levels_used)} lv, {span}]",
                     fontsize=8.5, fontweight="bold", color="#0f172a")

        cbar = fig.colorbar(mesh, ax=ax, fraction=0.036, pad=0.02, extend="max")
        cbar.set_label("IVT (kg m$^{-1}$ s$^{-1}$)", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

        out_name = f"ivt_{cycle_label.replace(' ', '_')}_f{fh:03d}.png"
        out_path = os.path.join(out_dir, out_name)
        fig.savefig(out_path, format="png", dpi=115, bbox_inches="tight")
        plt.close(fig)
        return f"maps/ivt/{model}/{out_name}"
    except Exception as e:
        logging.error(f"[IVT] render failed {model} f{fh:03d}: {e}")
        try:
            plt.close("all")
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def fetch_ivt_maps(models=None, session=None):
    """Build the IVT map set.

    Returns {model: {"cycle": "YYYYMMDD HHz", "maps": {"f024": "maps/ivt/..."}}}
    ready to drop into the history.json payload. Never raises: a failure anywhere
    degrades to fewer maps, never to a failed pipeline run.
    """
    if not IVT_MAPS_ENABLED:
        return {}
    models = models or [m for m, c in IVT_MODELS.items() if c["enabled"]]
    own_session = session is None
    if own_session:
        session = requests.Session()
        session.headers.update({"User-Agent": IVT_UA})

    started = time.time()
    out = {}
    try:
        for model in models:
            cfg = IVT_MODELS[model]
            date_str, cycle = determine_cycle(session, model)
            if not date_str:
                continue
            cycle_label = f"{date_str} {cycle}z"
            init = datetime.datetime.strptime(f"{date_str}{cycle}", "%Y%m%d%H")
            maps, attempted = {}, 0

            for fh in cfg["steps"]:
                if time.time() - started > IVT_BUDGET_S:
                    logging.warning(f"[IVT] budget spent; stopping {model} at f{fh:03d}")
                    break
                attempted += 1
                path = _fetch_grib(session, _build_url(model, date_str, cycle, fh),
                                   tag=f"{model} f{fh:03d}")
                if not path:
                    time.sleep(IVT_REQUEST_PAUSE_S)
                    continue
                try:
                    lons, lats, mag, iu, iv, mslp, lv = grid_ivt(path, cfg["moisture"])
                    valid = (init + datetime.timedelta(hours=fh)).strftime("%d %b %HZ")
                    rel = render_ivt_map(model, lons, lats, mag, iu, iv, mslp,
                                         cycle_label, fh, valid, lv)
                    if rel:
                        maps[f"f{fh:03d}"] = rel
                except Exception as e:
                    logging.error(f"[IVT] {model} f{fh:03d} integrate/render: {e}")
                finally:
                    try:
                        os.unlink(path)
                    except Exception:
                        pass
                time.sleep(IVT_REQUEST_PAUSE_S)

            if maps:
                out[model] = {"cycle": cycle_label, "maps": maps}
            logging.info(f"[IVT] {model}: {len(maps)}/{attempted} steps rendered "
                         f"from {cycle_label}")
    except Exception as e:
        logging.error(f"[IVT] driver aborted: {e}")
    finally:
        if own_session:
            session.close()

    total = sum(len(v["maps"]) for v in out.values())
    logging.info(f"[IVT] complete: {total} maps across {len(out)} model(s) "
                 f"in {time.time() - started:.0f}s")
    return out


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    # Smoke test: one model, three steps, so a first live run is cheap to inspect.
    IVT_MODELS["rap"]["enabled"] = False
    IVT_MODELS["nam"]["enabled"] = False
    IVT_MODELS["gfs"]["steps"] = [0, 24, 48]
    print(json.dumps(fetch_ivt_maps(), indent=2))
