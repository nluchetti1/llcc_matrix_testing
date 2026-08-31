import datetime
import time
import json
import math
import os
import re
import requests
import concurrent.futures
import threading
import random
import logging
import pygrib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Integrated Vapor Transport. ivt.py is pure math (no I/O, unit-tested standalone);
# ivt_maps.py does the NOMADS fetch + CW3E-style rendering for the spatial panels.
from ivt import column_ivt
import ivt_maps

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
CACHE_DIR = "./workspace_cache"
HISTORY_FILE = "history.json"
STATIONS = ["kdab", "kxmr", "kmlb", "kfpr", "kpbi"]
# MODELS is the full display/column set. BUFKIT_MODELS is the subset that comes from PSU
# BUFKIT + NOMADS grib (the airport soundings and NOMADS pad columns). ECMWF is additive and
# point-extracted from ECMWF Open Data, so it must NOT be swept into the BUFKIT/NOMADS loops.
MODELS = ["gfs", "rap", "hrrr", "ecmwf"]
BUFKIT_MODELS = ["gfs", "rap", "hrrr"]

# ---- PSU BUFKIT politeness ------------------------------------------------------------
# PSU's BUFKIT server throttles hard, and when a burst of parallel requests arrives from a
# single IP (exactly what a GitHub Actions runner looks like) it stops answering altogether:
# every in-flight request read-times-out at the same instant, then subsequent ones can't even
# open a connection. Four things keep us under its radar:
#   1. LOW CONCURRENCY, enforced by a semaphore rather than just a small pool, so the cap
#      holds no matter how the executor is sized.
#   2. A BROWSER USER-AGENT. The default python-requests UA gets dropped on the floor.
#   3. A JITTERED STAGGER before each request, so 15 tasks don't align into a burst.
#   4. EXPONENTIAL BACKOFF WITH JITTER, so retries don't fire in lockstep and re-trigger
#      the same throttle that caused the first failure.
BUFKIT_MAX_CONCURRENCY = 2      # simultaneous connections to PSU. Do not raise casually.
BUFKIT_ATTEMPTS = 4             # total tries per station-model
BUFKIT_CONNECT_TIMEOUT = 12     # seconds to establish the TCP connection
BUFKIT_READ_TIMEOUT = 45        # seconds to receive the body once connected
BUFKIT_STAGGER_S = (0.4, 1.8)   # random pre-request pause, seconds
BUFKIT_BACKOFF_BASE_S = 4.0     # first retry waits ~4 s, then ~8, ~16 (x0.6-1.4 jitter)
BUFKIT_USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64; rv:120.0) "
                     "Gecko/20100101 Firefox/120.0")
# Hard gate shared by every worker thread.
_BUFKIT_GATE = threading.Semaphore(BUFKIT_MAX_CONCURRENCY)

# ---- ECMWF Open Data (IFS HRES 0.25°, CC-BY-4.0) additive global column ----
ECMWF_ENABLED = True
ECMWF_SOURCE = "ecmwf"    # ecmwf-opendata source: ecmwf | aws | azure | google
ECMWF_MAX_FH = 144        # forecast hours to ingest. IFS open-data is 3-hourly out to 144 h (then
                          # 6-hourly, which this step list would silently miss), so 144 is the
                          # natural stop. ~49 steps: bigger download + slower retrieve than 48 h,
                          # traded for ECMWF reaching day 6 in the matrix and the 10Z panel.
ECMWF_LEVELS_HPA = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100]

# The whole 49-step retrieve used to be ONE client.retrieve() call wrapped in the shared
# 300 s SOURCE_DEADLINE_S. That is an all-or-nothing bet: if the portal is slow, or 429s
# and ecmwf-opendata starts its own retry sleep, the deadline fires, the thread is
# abandoned, and the column comes back COMPLETELY EMPTY — which is exactly the "ECMWF is
# no longer an option" symptom, because an empty dict never reaches the skew-T export or
# the 10Z panel. So the retrieve is now chunked and given its own internal wall-clock
# budget: each chunk that lands is parsed and kept, and running out of time costs forecast
# HOURS instead of the whole model.
ECMWF_STEP_CHUNK = 8       # steps per retrieve call. Smaller = more requests, finer salvage.
ECMWF_BUDGET_S = 480       # internal budget; returns whatever has been parsed when spent.
ECMWF_DEADLINE_S = 660     # outer backstop (must exceed ECMWF_BUDGET_S, else it pre-empts it)
# Tried in order until one yields a chunk. The portal is normally fastest, but it is also
# the one that 429s under load, and a mirror that works beats a fast source that refuses.
ECMWF_SOURCE_FALLBACKS = ["ecmwf", "azure", "aws"]

# ---- ECMWF ENS ensemble column for the 10Z panel (IFS ENS 0.25 deg, open data) --------------
# Panel-only, exactly like GEFS: a second global ensemble to read against GEFS at day 3+.
#
# THREE THINGS THE LIVE PROBE ESTABLISHED, all of which shape the settings below:
#   1. The CONTROL member (type="cf") has no pressure-level entries in the open-data tier —
#      a cf request errors with "Cannot find index entries". So this is 10 PERTURBED members
#      (pf 1..10) with no control. For airmass indices that's statistically fine; it just
#      differs from GEFS, which is c00 + p01..p14.
#   2. Every one of t/r/u/v is published on the levels we need, so nothing is approximated.
#   3. The ECMWF portal was FASTER than both cloud mirrors (AWS returned 503 Slow Down at
#      0.2 MB/s vs 2.4 MB/s direct), so this deliberately reuses ECMWF_SOURCE.
ECMWF_ENS_ENABLED = True
ECMWF_ENS_MEMBERS = 10        # perturbed members pf 1..N (no control; see above)
# Capped to match the HRES column. Also the exact limit of the 06/18Z ENS runs, so every
# cycle is usable rather than only the deep 00/12Z ones.
ECMWF_ENS_MAX_FH = 144
# Six levels, no upper set. That drops the 300-150 mb anvil flow for this column (the panel
# renders it as an em dash) and cuts the fetch ~40%. Everything else the panel shows —
# Thompson, PWAT, 700-500 RH, 1000-700 mean flow, regime, Cizek lightning — is unaffected.
ECMWF_ENS_LEVELS_HPA = [1000, 925, 850, 700, 600, 500]
ECMWF_ENS_PARAMS = ["t", "r", "u", "v"]   # gh omitted; heights fall back to barometric
# ENS advances every 6 h while this pipeline runs hourly, so rows are cached against the
# cycle and only refetched when a new one posts (~0.9 GB / 6 min when it does).
ECMWF_ENS_CACHE_ENABLED = True
# The measured cold-cycle fetch is ~940 MB in ~285 s. Against the shared 300 s
# SOURCE_DEADLINE_S that is a coin flip, and losing it does more damage than losing one
# run: _run_with_deadline returns ({}, None), the None cycle key makes the payload write
# ecens_cache=None, the cache is DESTROYED, and the next hourly run re-downloads the same
# 940 MB into the same 300 s deadline. That is the failure loop behind "ECMWF ENS vanished
# from the 10Z table". Own deadline here, plus cache carry-forward at the call site.
ECMWF_ENS_DEADLINE_S = 600

# ---- Skew-T sounding export -----------------------------------------------------------
# Ships the KXMR profiles the matrix was already built from into a SEPARATE soundings.json,
# so the frontend can draw an interactive skew-T per model per forecast hour. Kept out of
# history.json deliberately: that file carries five runs of history, and folding ~290 full
# profiles into each snapshot would multiply its size for data only ever wanted for the
# current run.
SKEWT_ENABLED = True
SKEWT_SITE = "kxmr"
# Per-model cap on the number of PROFILES exported (not forecast hours — an hourly model
# gets 1 profile per hour, ECMWF at 3-hourly gets 1 per 3 hours, so the same number buys a
# very different horizon). At 96 an hourly model reaches f96 and ECMWF exports its full
# 3-hourly range. Cost is roughly 0.7 KB and one MetPy parcel calculation per profile, so
# raising this grows soundings.json and the run time in a straight line.
#
# Set to 200 so it is a RUNAWAY GUARD rather than a limit anyone meets: at 96 it was silently
# truncating GFS, whose BUFKIT column carries more steps than that. Every model now exports
# its full range and the number only exists to stop a malformed feed producing a huge file.
SKEWT_MAX_PROFILES = 200
SKEWT_MAX_HOURS = SKEWT_MAX_PROFILES   # back-compat alias
SKEWT_TOP_HPA = 100        # drop levels above this; nothing in the panel needs the stratosphere
SKEWT_FILE = "soundings.json"

# Columns that must never reach the skew-T, whatever else they are good for.
#
# REFS is here because the NOMADS ensemble-MEAN product publishes only FIVE pressure levels
# (925/850/700/500/250). That is enough for a layer-mean wind and nothing else: a skew-T
# drawn through five points is four straight segments, the parcel curve is meaningless, and
# the derived indices are nonsense in a way that LOOKS authoritative (a measured KXMR 25/10
# REFS profile gave Thompson -13.8 and 0 J/kg CAPE on a day the CAMs had 1800-2200 J/kg).
# RRFS is unaffected: it comes off the full 21-level isobaric set and stays in the panel.
SKEWT_EXCLUDE_MODELS = {"refs"}
# Belt and braces for the same problem: any profile thinner than this is not a sounding.
# 8 clears every real column (RRFS 21, ECMWF 12, GEFS 11, BUFKIT ~40) and stops a future
# thin product sneaking in the way REFS did.
SKEWT_MIN_LEVELS = 8
# Run the Cizek RF on every exported profile, not just the 10Z panel rows. See the caveat
# in build_soundings_export: the forest was trained on 10Z soundings only.
SKEWT_LIGHTNING = True

# ---- Convective (cumulus) mask for the Thick Cloud Layer / Max Layer Thickness LLCC ----
# The Thick Cloud Layer rule targets STRATIFORM decks; cumulus is governed by its own LLCC rule.
# We use HRRR composite reflectivity as a convection detector: where a convective core sits within
# ~5 nm of a site, those hours are tagged 'convective' so the frontend marks the thick-layer /
# max-thickness fields as convective rather than flagging a stratiform bust. The threshold is set
# CONSERVATIVELY (40 dBZ) on purpose: in an LLCC context, hiding a real stratiform violation
# (over-masking) is worse than leaving a cumulus core flagged (under-masking, which is safe and
# redundant with the Cumulus rule). Raw thickness values are preserved; only the label changes.
# NOTE: reflectivity catches convective *cumulus cores* well, but a thin/detached anvil is low-
# reflectivity ice and won't trip this — anvil detection is a separate problem (high cold cloud +
# upstream convection) not solved here. The ideal discriminator is derived reflectivity at the
# -10C isotherm (above the melting-layer bright band); composite is used as a robust first cut.
CONVECTIVE_MASK_ENABLED = True
CONVECTIVE_DBZ = 40.0     # composite reflectivity (dBZ) at/above which a column is "convective"
CONVECTIVE_NBR_NM = 5.0   # neighborhood radius (nautical miles) sampled around each site

# ---- Anvil mask for the Thick Cloud Layer rule (attached/detached anvil exclusion) ----
# The Thick Cloud Layer rule ALSO does not apply to anvil clouds (governed by the anvil LLCC rules).
# An anvil is high glaciated cloud streaming downwind from a convective core, so a site-hour is
# tagged 'anvil' when three things line up: (1) the model's own thick deck is glaciated — its top is
# above the -20C level; (2) the point itself is non-convective (else it's the CB, already caught by
# the convective mask); and (3) a >=40 dBZ core sits UPSTREAM along that model's 300-150 mb anvil
# flow, out to the advection reach. HRRR REFC is sampled by compass sector so each model checks the
# sector its own anvil streams from. Raw thickness values are preserved; only the label changes.
ANVIL_MASK_ENABLED = True
ANVIL_SRC_DBZ = 40.0        # upstream core reflectivity (dBZ) that can seed an anvil
ANVIL_NEAR_NM = 12.0        # inner radius (nm): beyond the immediate CB neighborhood
ANVIL_ADVECT_NM = 100.0     # outer radius (nm): how far an anvil can stream from its parent core
ANVIL_TOP_MARGIN_KFT = 2.0  # thick-deck top must be this far above the -20C height to count as ice
# Diagnostic mode: also evaluate the anvil test on cells that are NOT would-be thick-layer
# violations, recording the result as 'anvil_diag' (hover popup only — no badge, no suppression).
# This exists purely so the detector can be validated against GOES on a real anvil day without
# needing a thick-cloud violation to coincide. Set False to skip the extra bookkeeping.
ANVIL_DIAG_MODE = True

# ---- REFS Cumulus Cloud LLCC probabilities (durable ensemble-product path) ----
# Uses the pre-computed REFS echo-top exceedance probabilities P(echo top > {6096,9144,10668,
# 12192,15240} m) from the ensemble prob file, neighborhood-reduced within the LLCC radii and
# interpolated at the REFS-mean isotherm heights, to express the Cumulus Cloud STANDOFF rules:
#   Rule a: P(top >= -20C altitude) within 10 NM
#   Rule b: P(top >= -10C altitude) within  5 NM
# The flight-through rules (-5C / +5C) sit BELOW the lowest echo-top threshold (20 kft), so they
# cannot be resolved from this product (only floored) and would need the individual members.
REFS_CUMULUS_ENABLED = True
REFS_ECHOTOP_THRESHOLDS_M = [6096, 9144, 10668, 12192, 15240]  # 20/30/35/40/50 kft
REFS_CUMULUS_RADII_NM = {"neg10": 5.0, "neg20": 10.0}

# How to collapse the echo-top exceedance probability over the standoff box. The RETOP prob grid
# is ALREADY a REFS ensemble exceedance probability P(top > H) per gridpoint (member fraction),
# so a spatial MAX does NOT give "probability within the radius" — it returns the single hottest
# cell in the box, which saturates to ~100% on any convective Florida afternoon (this was the
# "values far too high" bug). Options:
# How to collapse the echo-top exceedance probability over the standoff RING (a true circular
# radius, not a square box — see _ring_reduce). The RETOP prob grid is a REFS ensemble exceedance
# probability P(top > H) per gridpoint; the LLCC Cumulus rule cares whether a qualifying cloud
# top exists ANYWHERE within the standoff radius, so 'max' is the literal operation. (An earlier
# worry that 'max' saturates to ~100% did not hold up: the field is well-behaved — cross-checked
# against NOAA/GSL DESI REFS-CONUS echo-top prob, the Cape reads nil-to-low when it should, and
# the in-ring max peaked near 58% on a convective afternoon, not 100%.) Options:
#   "max"       : highest P(top > H) anywhere in the ring — the LCC "within X nm" operation
#   "point"     : pad gridpoint only (ignores the radius; use only to match a point viewer)
#   "mean"      : areal-average exceedance in the ring (less conservative than a go/no-go wants)
#   "p90"/"p75" : percentile in the ring (reads above the pad point on convective days)
# NOTE: the enspost 'prob' RETOP field is confirmed (by DESI slider comparison) to be a ~40 km
# NEIGHBORHOOD-MAX ensemble probability, NOT a point field — our debug map matched DESI at 40 km,
# not 10 km. The 40 km smoothing is baked into the data, so every reducer below inherits it; none
# can recover the 5/10 nm LCC standoff. Until we replace this source with a member-based NMEP at
# the true radii, 'point' is the honest read: the pad value already means "within 40 km", so an
# extra ring-max would just double-count. (max/p90/mean kept for reference only.)
REFS_CUMULUS_NBR_REDUCER = "point"

# When True, render the raw P(echo top > 20 kft) grid we ingest to maps/refs_debug/ (one PNG per
# forecast hour) AND surface them as hover popups on the REFS cumulus cells. Now a feature, not a
# one-off diagnostic — leave True or the popups go blank.
REFS_CUMULUS_DEBUG_MAPS = True

# When True, one-shot: list the REFS S3 prefix for the resolved cycle and log the file-name
# patterns present, so we can locate the member-level RETOP files needed to build a real 5/10 nm
# neighborhood-max ensemble probability (replacing the 40 km enspost 'prob' product). Diagnostic.
REFS_MEMBER_PROBE = True

# Compute the Cumulus echo-top standoff probability as a TRUE neighborhood-max ensemble
# probability (NMEP) from the individual RRFS ensemble members at the real 5/10 nm radius,
# instead of the 40 km enspost 'prob' product. Falls back to enspost if members are unavailable.
REFS_MEMBER_NMEP_ENABLED = True
# Time-lag the prior cycle's members (+6 h, valid-time aligned) to double the ensemble size
# (5 -> 10 members), taking probability granularity from 20% steps to 10%.
REFS_MEMBER_TLE = True
# How many prior cycles to fold in when REFS_MEMBER_TLE is on. 1 = the -6 h cycle only (10
# members, the validated sweet spot). 2 = also the -12 h cycle (15 members, ~6.7% steps) — more
# sampling but the oldest members are stale for convective *placement* and can smear the sharp
# 5/10 nm signal toward climatology. It degrades gracefully if an older cycle isn't posted deep
# enough (needs f(24 + 6*k)), so a too-large value simply uses as many cycles as are available.
REFS_MEMBER_LAG_CYCLES = 1
# Forecast-hour cap for the member ensemble. Match the REFS/RRFS sounding depth (60 h) so the
# Cumulus NMEP carries as far as the isotherm columns do. The fetch auto-discovers how deep the
# cycle actually posted and fetches out to the deepest hour at or below this cap; deep hours the
# +6 h lag cycle doesn't reach fall back to 5 members.
REFS_MEMBER_WINDOW_FH = 60

# Per-member REFS launch-thermo (KXMR panel): pull each ensemble member's full isobaric sounding at
# XMR for the 10Z hours, compute the indices per member, and average the RESULTS — the valid way to
# get an ensemble TI/PWAT (never from the ensemble-MEAN sounding). Costs a handful of member GRIB
# range-reads per run (only the ~2-3 forecast hours that land on 10Z), so it's gated here.
# Individual REFS members are NOT published on the NOMADS parallel feed — only derived
# ensemble products (mean, sprd, pmmn, lpmm, avrg, prob, eas). The per-member sweep this flag
# controls has nothing to read, so it stays off.
#
# WHAT THAT COSTS, plainly: the 10Z panel's REFS row is now computed FROM the ensemble mean
# sounding rather than being the mean of indices computed per member. Those are different
# quantities. Averaging soundings smooths the moisture profile, so the mean sounding is drier
# wherever members disagreed, and its Thompson / PWAT / K-Index read systematically less
# unstable than the ensemble actually is. Still a useful airmass signal; not an ensemble index,
# and it carries no spread. Restore this the moment members reappear.
REFS_MEMBER_THERMO_ENABLED = False

# Show the REFS ensemble MEAN in the 10Z panel now that per-member soundings are gone.
#
# This is a compromise, not a free win. Airmass indices from a MEAN SOUNDING are biased:
# averaging relative humidity across members smooths away the moisture structure, so PWAT
# and K-Index read drier and Lifted Index reads more stable than the ensemble's own average
# of those quantities would. The bias grows with member spread — it is smallest on a
# well-agreed airmass and worst exactly when the members disagree, which is when you would
# most want the number.
#
# The WIND fields do not suffer this: averaging u/v across members is linear, so the
# 1000-700 mb mean flow, the regime and the anvil flow are as valid from the mean as from
# the members. That is most of why the row is worth having at all.
#
# Set False to go back to omitting REFS from the panel entirely.
REFS_MEAN_IN_PANEL = True

# Minimum number of levels carrying humidity before the moisture-driven indices are trusted.
#
# Measured on 2026-08-24: refs.tHHz.mean carries 6 isobaric levels and exactly ONE with RH.
# Thompson, K-Index, Lifted Index, PWAT and the 700-500 RH all need vertical moisture
# structure, and computing them from a single humidity level produces numbers that look
# ordinary and mean nothing. The WIND fields are unaffected — mean flow, regime and anvil
# flow only need u/v, which REFS does publish on several levels.
#
# So the gate is on the DATA, not on the model name: any column thin on humidity gets a
# flow-only row. If REFS later publishes a fuller mean, it starts working with no code change.
#
# RAISED 5 -> 6 on 2026-08-25. The pre-install REFS path now returns FIVE humidity levels
# rather than one, which cleared the old gate by exactly one level and let the indices
# through. They should not have gone through: 925/850/700/500/250 mb gives a 250 mb gap
# straddling the whole mid-troposphere, and the 25/10Z KXMR mean produced Thompson -13.8,
# 700-500 RH 28%, 0 J/kg CAPE and a Cizek probability of 19.7% on a morning the CAMs had
# Thompson +26 to +34 and 50-76%. Those numbers are not a low-end forecast, they are an
# artefact of the vertical gap, and they were being plotted on the same axis as real ones.
#
# 6 is chosen against the columns that must survive: ECMWF ENS publishes 6 humidity levels
# (1000/925/850/700/600/500) and is unaffected, as are GEFS (11), ECMWF HRES (12), RRFS (21)
# and BUFKIT (~40). Set back to 5 to restore the old behaviour in one edit.
PANEL_MIN_RH_LEVELS = 6

# ---- GEFS ensemble column for the 10Z panel (global 0.5 deg, AWS mirror) ---------------------
# Panel-only: GEFS is far coarser than the mesoscale columns and would add nothing to the hourly
# matrix, but it gives a genuine global-ensemble read on the daily airmass out to a week.
# NOTE ON FILES: pgrb2a (the "primary" half-degree file) carries TMP and RH at ONLY 1000/925/850
# mb, so it alone cannot produce K-Index, Lifted Index or 700-500 RH. The mid/upper temperature
# and moisture live in pgrb2b, so BOTH files are byte-ranged and concatenated per member.
GEFS_ENABLED = True
GEFS_AWS_ROOT = "https://noaa-gefs-pds.s3.amazonaws.com"
GEFS_MEMBERS = 15          # c00 control + p01..p14; spread converges quickly for airmass indices
GEFS_MAX_FH = 168          # 3-hourly output; 168 h = 7 forecast days
# GEFS cycles every 6 h but this pipeline runs hourly, so the fetched rows are cached against the
# cycle and only refetched when a new cycle appears (saves ~5 of every 6 runs).
GEFS_CACHE_ENABLED = True
# S3 has no burst limits, but keep a tiny pause as a courtesy / connection-reuse aid.
GEFS_REQUEST_PAUSE_S = 0.05
# GEFS 0.5-deg carries a REDUCED isobaric set (no 975/950/900/800/750/650/550/450/350 mb). Only
# ask for levels it actually publishes; anything else is a wasted lookup.
GEFS_LEVELS_HPA = [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100]
# The panel thermo needs only pressure/T/dewpoint/wind - geopotential height is never read, so
# HGT is deliberately NOT fetched (that alone is ~20% of the bytes).
GEFS_VARS = ("TMP", "RH", "UGRD", "VGRD")
# Minimum members required before a forecast day is allowed into the panel. A row built from
# one or two members has no meaningful spread — its min/max collapse toward the mean, which
# reads as high confidence when it actually means "almost no data". Rows under this are dropped.
GEFS_MIN_MEMBERS_PER_ROW = 8


STN_COORDS = {
    "kxmr": {"lat": 28.468, "lon": -80.556},
    "kdab": {"lat": 29.180, "lon": -81.058},
    "kmlb": {"lat": 28.103, "lon": -80.645},
    "kfpr": {"lat": 27.498, "lon": -80.373},
    "kpbi": {"lat": 26.683, "lon": -80.095}
}

# Cape Canaveral / KSC launch pads. These are derived from raw model isobaric GRIB2
# (GFS/RAP/HRRR) rather than BUFKIT, since the pads have no dedicated BUFKIT profiles.
# KTTS (KSC Shuttle Landing Facility) and KCOF (Patrick SFB) are airfields handled the
# same GRIB way; they inherit the KXMR HREF-lightning/CT proxy in the frontend.
LAUNCH_PADS = {
    "lc39a": {"lat": 28.608, "lon": -80.604, "label": "LC-39A (KSC)"},
    "lc39b": {"lat": 28.627, "lon": -80.621, "label": "LC-39B (KSC)"},
    "lc37":  {"lat": 28.532, "lon": -80.565, "label": "LC-37B (CCSFS)"},
    "slc40": {"lat": 28.562, "lon": -80.577, "label": "SLC-40 (CCSFS)"},
    "slc41": {"lat": 28.583, "lon": -80.583, "label": "SLC-41 (CCSFS)"},
    "lc36":  {"lat": 28.470, "lon": -80.538, "label": "LC-36 (CCSFS)"},
    "ktts":  {"lat": 28.615, "lon": -80.695, "label": "KTTS (KSC SLF)"},
    "kcof":  {"lat": 28.235, "lon": -80.610, "label": "KCOF (Patrick SFB)"},
}

THRESHOLD_MAP = {25: "p25", 50: "p50", 100: "p100", 200: "p200"}
MSG_INDEX_THRESHOLDS = {1: "p25", 2: "p50", 3: "p100", 4: "p200"}

# ---- RRFS / REFS configuration -------------------------------------------------
# RRFS (deterministic) and REFS (ensemble) are pre-operational until 2026-08-31 12z.
# We pull from the public AWS Open-Data bucket (no auth, HTTP range-request friendly)
# rather than NOMADS, using each file's .idx sidecar to byte-range only the ~21 isobaric
# levels we need instead of downloading the whole ~40MB CONUS file.
#
# The rrfs_public/ tree is the "operationally-representative" set per the AWS registry:
#   rrfs_public/rrfs.YYYYMMDD/CC/rrfs.tCCz.prslev.3km.fFFF.conus.grib2      (deterministic)
#   rrfs_public/refs.YYYYMMDD/CC/ensprod/ ...                              (ensemble products)
RRFS_ENABLED = True          # master switch for the RRFS deterministic pad column
REFS_ENABLED = True          # master switch for the REFS ensemble-average pad column
# NOMADS parallel feed. RRFS/REFS moved off the AWS prototype bucket (SCN 26-48); the
# rrfs_a/ tree that carried individual ensemble members is gone, so REFS is now the published
# ENSEMBLE MEAN only — see REFS_MEMBER_THERMO_ENABLED for what that costs.
RRFS_NOMADS_ROOT = "https://nomads.ncep.noaa.gov/pub/data/nccf/com"
RRFS_AWS_ROOT = "https://noaa-rrfs-pds.s3.amazonaws.com"   # legacy, retained for reference

# RRFS runs EVERY hour. Only the synoptic cycles run to full length; the off-hour runs are
# short, so the fetch depth follows the cycle actually chosen instead of always asking for 60
# and firing 40 pointless probes an hour.
RRFS_CYCLE_HOURS = list(range(24))
RRFS_SYNOPTIC_CYCLES = [0, 6, 12, 18]
RRFS_MAX_FH = 60             # synoptic cycles
RRFS_SHORT_FH = 18           # off-hour cycles
# THIS is why an off-hour RRFS column ends cleanly ~18-24 h out with nothing beyond, even on
# a run where every hour it did fetch came back complete. A cycle outside RRFS_SYNOPTIC_CYCLES
# is capped at RRFS_SHORT_FH and the sweep never PROBES past it — so the tail is not missing
# data, it is data that was never asked for. Distinguishing that from the budget cutoff
# matters: one is fixed by raising a number, the other by spending more wall clock.
#
# Rather than hardcode a new guess, probe. After the short cap is exhausted, ask the server
# whether the cycle actually goes further; if the .idx exists, extend to RRFS_MAX_FH. Costs
# one request on off-hour cycles and adapts on its own if NCEP lengthens the parallel runs.
RRFS_PROBE_BEYOND_SHORT = True
RRFS_PROBE_FH = 24           # the hour probed to decide whether an off-hour cycle runs long
RRFS_LATENCY_H = 2           # cycle directories are still filling ~1.5-2 h after cycle time

# REFS is 6-hourly and sits one directory deeper, under ensprod/.
REFS_CYCLE_HOURS = [0, 6, 12, 18]
REFS_MAX_FH = 60
REFS_LATENCY_H = 4

# NOMADS is not S3. It asks for restraint and will treat a burst as an attack, the same way
# PSU did. This concurrency is deliberately low and is NOT a knob to turn up casually.
# NOMADS throttles by REDIRECT, not by 403.
#
# Measured 2026-08-25: the first .idx request per model returned 200 and every subsequent one
# returned 302 — RRFS f011 fine, f001-f018 all bounced; REFS f014 fine, f001-f060 all bounced.
# f011 cannot exist while f001 does not, so this was never file availability. NOMADS says so
# in its own docs: "include a 10 second wait between fetches ... the server may mistake
# excessive requests as denial-of-service attack and block the user."
#
# So: one connection, and a pause between requests. This is the PSU lesson again — the fetch
# is deliberately slow because the alternative is being served nothing at all.
NOMADS_MAX_CONCURRENCY = 1
# Seconds between NOMADS requests. Their guidance is 10 s; each RRFS byte-range already takes
# ~20 s to transfer, so the pause below is on top of a naturally slow request and the
# effective spacing is well past their floor. REFS files are small and genuinely need it.
NOMADS_REQUEST_PAUSE_S = 4.0
# Pause between the byte-range GETs WITHIN one file. This must be tiny and separate from the
# between-files pause: a single RRFS hour needs ~100 range requests, so charging the 4 s
# file-pause for each of them cost 6.7 minutes of pure sleeping per forecast hour and turned
# a 20-second download into a 7-minute one. The ranges are one logical transfer of one file,
# not 100 requests for new data.
NOMADS_RANGE_PAUSE_S = 0.15
# Hard wall-clock budget per model kind. Without this a slow or throttled source has no way
# to end: the last run sat for 36 minutes and was cancelled with nothing committed. Better a
# short column and a finished run than a hung job.
NOMADS_KIND_BUDGET_S = 420
# Per-kind override of the budget above.
#
# One number for all three kinds was the direct cause of the ragged RRFS column. RRFS is
# HOURLY out to f060 and REFS is hourly to f060 as well, but a REFS hour is a small file
# while an RRFS hour is a full CONUS isobaric set; measured 2026-08-25 11:10Z, RRFS managed
# 26 of 60 hours in the 420 s it was given (~16 s/hour) and stopped mid-column. The hours it
# HAD reached were f001-f016 plus the reserved 10Z islands on days 2 and 3, which is why the
# matrix showed values, then a block of blanks, then values again. Nothing was random and
# nothing failed — the sweep simply ran out of clock, and the panel reservation made the
# leftover look scattered.
#
# 900 s covers all 60 RRFS hours at the measured rate with margin. Raise/lower against the
# "RRFS: N hours in Ms (X.Xs/hour)" line the sweep now logs — that number is the one to
# budget against, and it moves with NOMADS load.
NOMADS_KIND_BUDGET_OVERRIDE = {"rrfs": 900, "refs": 480, "hrrr": 300}


def _nomads_kind_budget(kind):
    return NOMADS_KIND_BUDGET_OVERRIDE.get(kind, NOMADS_KIND_BUDGET_S)


# Wall-clock ceiling for any single upstream fetch that manages its own retries.
SOURCE_DEADLINE_S = 300
# Give up on a model entirely after this many throttled requests in a row.
#
# Measured 2026-08-25: NOMADS served the first forecast hour (~105 range requests) and then
# began answering 302 with NO Location header — a brush-off, not a redirect. Once that starts
# it does not stop, so retrying burns the whole budget and the JOB GETS CANCELLED WITH
# NOTHING COMMITTED. That is the worst outcome: a missing RRFS column is survivable, an
# empty dashboard is not. Bail out fast, let the rest of the pipeline finish, publish.
NOMADS_THROTTLE_GIVEUP = 6
# Merge byte ranges separated by less than this many bytes into one request.
#
# The download issued ONE REQUEST PER GRIB MESSAGE — ~105 per forecast hour, ~1900 per run.
# No pause makes that acceptable to NOMADS; the count has to come down, not the rate. GRIB
# orders messages by variable, so wanted levels sit in runs with unwanted ones between them;
# merging across those gaps trades a little wasted download for an order of magnitude fewer
# requests. Each 3 km message is ~1.3 MB, so this spans roughly six skipped messages.
NOMADS_RANGE_MERGE_GAP = 8_000_000
# Retry a redirected (throttled) request this many times, backing off each time.
NOMADS_THROTTLE_RETRIES = 3

# Forecast hours whose VALID time lands near the 10Z assessment hour are fetched FIRST,
# ahead of the sequential f001, f002, ... sweep.
#
# Reason: a 3 km CONUS message is ~1.3 MB and byte-range cannot subset spatially, so one
# forecast hour costs ~133 MB and a 60-hour sweep is ~8 GB. When that runs out of time the
# sweep truncates at whatever hour it reached — and on 2026-08-24 that was f010, six hours
# short of the f016 the 10Z panel needed. The panel then showed no RRFS row at all, which
# looked like a panel bug and was actually a download budget running out.
#
# Fetching the handful of panel-critical hours first makes the panel robust to truncation:
# worst case the MATRIX is short, which is visible and expected, rather than the panel
# silently losing a model.
RRFS_PRIORITISE_PANEL_HOURS = True

# The TRUE member-based cumulus NMEP read individual RRFS ensemble members from the AWS
# prototype tree (rrfs_a/rrfsens.DATE/CC/mNNN/). That tree went away with SCN 26-48 and NOMADS
# publishes no members, so the member path can only fail and fall back. Left OFF so the run
# does not spend a probe sweep every hour proving that.
#
# The cumulus columns still populate, via the enspost/ensprod 'prob' exceedance-curve method
# (fetch_refs_echotop_probs). That is a ~40 km neighbourhood product rather than a true 5/10 nm
# member count, so the probabilities are SMOOTHER and less peaked than the member version was —
# the column is labelled with its provenance so the two are not confused. Flip back to True the
# day members are published again.
RRFS_MEMBER_NMEP_ENABLED = False

# HRRR pressure-level GRIB2 on AWS (byte-range friendly via .idx, no bot-blocking). HRRR
# only reaches f48 on the 00/06/12/18z "extended" cycles; other cycles stop at f18. We pull
# HRRR pads through this AWS path (same idx machinery as RRFS) rather than the flaky NOMADS
# grib-filter, which was silently failing the cycle probe.
HRRR_AWS_ROOT = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
HRRR_EXTENDED_CYCLES = [0, 6, 12, 18]
# Use EVERY hourly HRRR cycle, not only the extended ones.
#
# Restricting to 00/06/12/18z guaranteed an f48 column, but at a cost that is wrong for an
# aviation board: with the latency below, the HRRR could be up to ~9 h old just before the
# next extended cycle cleared — by which point seven newer HRRR runs existed and the column
# was the stalest thing on the page. HRRR's whole value is that it is 3 km and hourly.
#
# With this on, the newest cycle is taken and its natural depth accepted: f48 on the extended
# cycles, f18 on the others. The long range is already covered by GFS, ECMWF, RRFS and REFS,
# so nothing is really lost, and the near-term column is never more than ~2-3 h old.
# Set False to restore the old always-f48 behaviour.
HRRR_ALL_CYCLES = True
# Hours before a cycle is considered complete enough to read. HRRR f18 lands roughly 50-60 min
# after cycle time and f48 nearer 110 min, so 2 h clears both.
HRRR_LATENCY_H = 2

# The exact REFS ensemble-mean filename ordering has drifted across the pre-op feed. We probe
# these candidate patterns (formatted with cycle `c` and forecast-hour ints) once per run and
# cache whichever resolves, so every subsequent hour reuses the confirmed pattern.
# (REFS_FILENAME_CANDIDATES removed: the NOMADS layout is documented and stable, so the
#  old filename search is no longer needed.)




# Bounding box — zoomed into the Space Coast launch corridor rather than all of FL
FL_DOMAIN = {"lat_min": 24.5, "lat_max": 31.0, "lon_min": -84.5, "lon_max": -79.0}

# Spatial plot PNGs are written here (relative path, served alongside index.html)
MAPS_DIR = "./maps"

# Global cache for static grid indices to maximize ThreadPool performance
_GRID_INDEX_CACHE = {}

# ---- Model run (cycle) registry -------------------------------------------------------
# Which model CYCLE produced each column, keyed (site, model) -> "YYYYMMDDHH". Every fetch
# path records here as it resolves its cycle, and the payload carries the result so the
# frontend can label a column "HRRR (12Z)" rather than leaving the user to guess whether
# they are looking at the 12Z run or a six-hour-old one. Populated per site because the same
# model name can come from different sources at different sites — HRRR is BUFKIT at the
# airports and AWS at the pads, and those are not always the same cycle.
_MODEL_CYCLES = {}


def _record_cycle(site, model, cycle_key):
    """Register the cycle behind one (site, model) column. cycle_key is 'YYYYMMDDHH'."""
    if not site or not model or not cycle_key:
        return
    _MODEL_CYCLES[(str(site).lower(), str(model).lower())] = str(cycle_key)


def _cycles_payload():
    """Reshape the registry into {site: {model: cycle}} for history.json."""
    out = {}
    for (site, model), cyc in _MODEL_CYCLES.items():
        out.setdefault(site, {})[model] = cyc
    return out


def _bufkit_init_cycle(bufkit_text):
    """Model initialisation time from a BUFKIT file, as 'YYYYMMDDHH'.

    BUFKIT carries one block per forecast hour, each stamped with its VALID time; the file
    has no explicit init field, so the earliest valid time is f000 and therefore the cycle.
    Two-digit years are resolved against the current century.
    """
    stamps = re.findall(r"TIME\s*=\s*(\d{6})/(\d{4})", bufkit_text or "")
    best = None
    for d, t in stamps:
        try:
            yy, mm, dd = int(d[0:2]), int(d[2:4]), int(d[4:6])
            hh = int(t[0:2])
            dt = datetime.datetime(2000 + yy, mm, dd, hh, tzinfo=datetime.timezone.utc)
        except Exception:
            continue
        if best is None or dt < best:
            best = dt
    return best.strftime("%Y%m%d%H") if best else None


def purge_workspace(cache_dir=CACHE_DIR):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    else:
        for f in os.listdir(cache_dir):
            try:
                os.unlink(os.path.join(cache_dir, f))
            except Exception:
                pass
    return cache_dir


def _collect_map_paths(*containers):
    """Recursively pull every non-empty relative map path out of any mix of nested dicts
    ({row: {thresh: path}}), flat dicts ({row: path}), lists, or bare strings. Used to
    build the 'keep' set for pruning WITHOUT the lossy `{**a, **b}` merge that previously
    let the CT maps clobber the density maps (and ct4 clobber ct1) on shared row keys."""
    paths = set()

    def walk(x):
        if x is None:
            return
        if isinstance(x, str):
            if x:
                paths.add(x)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple, set)):
            for v in x:
                walk(v)

    for c in containers:
        walk(c)
    return paths


def prune_stale_maps(referenced_paths, maps_dir=MAPS_DIR):
    """Deletes spatial-map PNGs on disk that aren't in `referenced_paths` (the maps/ folder
    only ever needs to hold the latest run's images plus the blank basemap fallback).
    `referenced_paths` is any iterable of relative paths like 'maps/xyz.png'."""
    if not os.path.exists(maps_dir):
        return

    keep = {os.path.basename(p) for p in referenced_paths if p}

    for f in os.listdir(maps_dir):
        if f not in keep:
            try:
                os.unlink(os.path.join(maps_dir, f))
            except Exception:
                pass


def pressure_to_height_ft(pres_hpa):
    return 145366.45 * (1.0 - (pres_hpa / 1013.25) ** 0.190284)


def extract_lightning_from_file(filepath, lat, lon, stn):
    """
    Extracts 4 lightning strike probability thresholds from SPC HREF GRIB2.
    Uses string representation tokens to bypass version incompatibilities with eccodes.
    """
    global _GRID_INDEX_CACHE
    threshold_results = {"p25": 0, "p50": 0, "p100": 0, "p200": 0}

    try:
        grbs = pygrib.open(filepath)

        # Grid Point Index Optimization
        if stn in _GRID_INDEX_CACHE:
            y_idx, x_idx = _GRID_INDEX_CACHE[stn]
        else:
            sample = grbs[1]
            lats, lons = sample.latlons()
            lons_normalized = np.where(lons > 180, lons - 360.0, lons)

            dist = (lats - lat) ** 2 + (lons_normalized - lon) ** 2
            y_idx, x_idx = np.unravel_index(dist.argmin(), dist.shape)
            _GRID_INDEX_CACHE[stn] = (y_idx, x_idx)
            logging.info(f"Verified grid index for {stn.upper()} at [y={y_idx}, x={x_idx}]")

        grbs.seek(0)
        for msg_idx, grb in enumerate(grbs, start=1):
            grid = grb.values

            # Determine fraction-vs-percent from the WHOLE grid's max, exactly as the spatial
            # map does — a per-cell test can't distinguish 0.6 ("0.6%" percent-stored) from
            # 0.6 ("60%" fraction-stored). Also sanitize masked/fill/NaN before taking the max.
            try:
                arr = np.ma.filled(np.ma.masked_invalid(np.ma.asarray(grid, dtype=float)), 0.0)
                arr = np.where((arr > 1e19) | (arr < 0), 0.0, arr)
            except (TypeError, ValueError):
                arr = np.zeros_like(grid, dtype=float)

            scale = 100.0 if arr.max() <= 1.0 else 1.0

            # Sample a small neighborhood around the station index and take the MEDIAN, not
            # the single cell. A lone fill/edge artifact at one grid point (which produced the
            # spurious 100% readings) gets rejected by the median of its neighbors.
            y0 = max(0, y_idx - 1); y1 = min(arr.shape[0], y_idx + 2)
            x0 = max(0, x_idx - 1); x1 = min(arr.shape[1], x_idx + 2)
            neighborhood = arr[y0:y1, x0:x1]
            raw_cell = float(np.median(neighborhood)) if neighborhood.size else float(arr[y_idx, x_idx])

            pixel_value = raw_cell * scale
            pixel_value = max(0.0, min(100.0, pixel_value))
            val = int(round(pixel_value))

            msg_str = str(grb).lower()

            # Diagnostic: when a suspiciously high value appears, dump exactly what produced
            # it so a false 100% can be traced rather than guessed at. Gated to >=90%.
            if val >= 90:
                logging.info(f"[LTG DIAG] {stn.upper()} msg#{msg_idx}: median_cell={raw_cell:.6g} "
                             f"single_cell={float(arr[y_idx, x_idx]):.6g} grid_max={arr.max():.6g} "
                             f"scale={scale:g} -> {val}% | {str(grb)[:100]}")

            if "upperlimit=25" in msg_str or "prob > 0.25" in msg_str or "probability=25" in msg_str:
                threshold_results["p25"] = val
            elif "upperlimit=50" in msg_str or "prob > 0.50" in msg_str or "probability=50" in msg_str:
                threshold_results["p50"] = val
            elif "upperlimit=100" in msg_str or "prob > 1.0" in msg_str or "probability=100" in msg_str:
                threshold_results["p100"] = val
            elif "upperlimit=200" in msg_str or "prob > 2.0" in msg_str or "probability=200" in msg_str:
                threshold_results["p200"] = val
            else:
                pos_key = MSG_INDEX_THRESHOLDS.get(msg_idx)
                if pos_key:
                    threshold_results[pos_key] = val

            if val > 0:
                logging.debug(f"  [HIT] {stn.upper()} scored {val}% for {msg_str.split(':')[0]}")

        grbs.close()
    except Exception as e:
        logging.error(f"Pygrib extraction failed for {stn.upper()}: {e}")

    return threshold_results


def _fig_to_png_file(fig, filename):
    os.makedirs(MAPS_DIR, exist_ok=True)
    out_path = os.path.join(MAPS_DIR, filename)
    fig.savefig(out_path, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    # Relative path for use directly as an <img src="..."> in index.html
    return f"maps/{filename}"


def generate_blank_basemap():
    """
    Renders a single 'no data available' Florida basemap (coastlines, counties, state
    borders, station markers, no overlay) used as a fallback whenever a given forecast
    hour/threshold has no spatial map yet (download failure, hour outside the HREF
    0-48h window, etc). Overwritten each run; lives at maps/blank_basemap.png.
    """
    try:
        proj = ccrs.PlateCarree()
        states_provinces = cfeature.NaturalEarthFeature(
            category="cultural", name="admin_1_states_provinces_lines",
            scale="50m", facecolor="none"
        )
        counties = cfeature.NaturalEarthFeature(
            category="cultural", name="admin_2_counties",
            scale="10m", facecolor="none"
        )

        fig = plt.figure(figsize=(5.5, 5.8), dpi=120)
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        ax.set_extent(
            [FL_DOMAIN["lon_min"], FL_DOMAIN["lon_max"], FL_DOMAIN["lat_min"], FL_DOMAIN["lat_max"]],
            crs=proj
        )

        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#dbeafe", zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f1f5f9", zorder=0)
        ax.add_feature(counties, edgecolor="#cbd5e1", linewidth=0.35, zorder=1)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#1e293b", linewidth=0.9, zorder=3)
        ax.add_feature(states_provinces, edgecolor="#475569", linewidth=0.8, zorder=3)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#1e293b", linewidth=0.8, zorder=3)

        for stn_id, coords in STN_COORDS.items():
            ax.plot(
                coords["lon"], coords["lat"], marker="^", markersize=6,
                color="#2563eb", markeredgecolor="white", markeredgewidth=0.8,
                transform=proj, zorder=5
            )
            ax.text(
                coords["lon"] + 0.06, coords["lat"] + 0.05,
                stn_id.upper(), fontsize=6, fontweight="bold", color="#1e3a5f",
                transform=proj, zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.6, edgecolor="none")
            )

        ax.gridlines(draw_labels=False, linewidth=0.4, color="#94a3b8", alpha=0.5, linestyle="--")
        ax.set_title("No Active Signal", fontsize=9, fontweight="bold", color="#94a3b8")

        out_path = os.path.join(MAPS_DIR, "blank_basemap.png")
        os.makedirs(MAPS_DIR, exist_ok=True)
        fig.savefig(out_path, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        return "maps/blank_basemap.png"
    except Exception as e:
        logging.error(f"Blank basemap generation failed: {e}")
        return None


def generate_spatial_threshold_maps(filepath, file_prefix):
    """
    Builds a Florida-domain spatial plot (PNG, written to MAPS_DIR) for each of the 4
    HREF lightning exceedance thresholds (p25/p50/p100/p200) from a single GRIB2 file.
    `file_prefix` should uniquely identify the run/forecast-hour, e.g. "20260630_00z_f012".
    Returns a dict like {"p25": "maps/20260630_00z_f012_p25.png", ...} (missing keys if a
    given threshold message wasn't found or plotting failed).
    """
    maps = {}
    try:
        grbs = pygrib.open(filepath)
        sample = grbs[1]
        lats, lons = sample.latlons()
        lons_n = np.where(lons > 180, lons - 360.0, lons)

        domain_mask = (
            (lats >= FL_DOMAIN["lat_min"]) & (lats <= FL_DOMAIN["lat_max"]) &
            (lons_n >= FL_DOMAIN["lon_min"]) & (lons_n <= FL_DOMAIN["lon_max"])
        )
        ys, xs = np.where(domain_mask)
        if len(ys) == 0:
            grbs.close()
            return maps

        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        sub_lats = lats[y0:y1 + 1, x0:x1 + 1]
        sub_lons = lons_n[y0:y1 + 1, x0:x1 + 1]

        proj = ccrs.PlateCarree()
        states_provinces = cfeature.NaturalEarthFeature(
            category="cultural", name="admin_1_states_provinces_lines",
            scale="50m", facecolor="none"
        )
        counties = cfeature.NaturalEarthFeature(
            category="cultural", name="admin_2_counties",
            scale="10m", facecolor="none"
        )

        grbs.seek(0)
        for msg_idx, grb in enumerate(grbs, start=1):
            msg_str = str(grb).lower()

            if "upperlimit=25" in msg_str or "prob > 0.25" in msg_str or "probability=25" in msg_str:
                thresh_key = "p25"
            elif "upperlimit=50" in msg_str or "prob > 0.50" in msg_str or "probability=50" in msg_str:
                thresh_key = "p50"
            elif "upperlimit=100" in msg_str or "prob > 1.0" in msg_str or "probability=100" in msg_str:
                thresh_key = "p100"
            elif "upperlimit=200" in msg_str or "prob > 2.0" in msg_str or "probability=200" in msg_str:
                thresh_key = "p200"
            else:
                thresh_key = MSG_INDEX_THRESHOLDS.get(msg_idx)

            if not thresh_key or thresh_key in maps:
                continue

            try:
                raw_vals = grb.values[y0:y1 + 1, x0:x1 + 1]
                vals = np.nan_to_num(np.asarray(raw_vals, dtype=float), nan=0.0)
                # Normalize fractional probabilities (0-1) up to percent (0-100)
                if vals.max() <= 1.0:
                    vals = vals * 100.0

                fig = plt.figure(figsize=(5.5, 5.8), dpi=120)
                ax = fig.add_subplot(1, 1, 1, projection=proj)
                ax.set_extent(
                    [FL_DOMAIN["lon_min"], FL_DOMAIN["lon_max"], FL_DOMAIN["lat_min"], FL_DOMAIN["lat_max"]],
                    crs=proj
                )

                # Base map styling
                ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#dbeafe", zorder=0)
                ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f1f5f9", zorder=0)
                ax.add_feature(counties, edgecolor="#cbd5e1", linewidth=0.35, zorder=1)
                ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#1e293b", linewidth=0.9, zorder=3)
                ax.add_feature(states_provinces, edgecolor="#475569", linewidth=0.8, zorder=3)
                ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#1e293b", linewidth=0.8, zorder=3)

                masked_vals = np.ma.masked_less_equal(vals, 0.0)
                mesh = ax.pcolormesh(
                    sub_lons, sub_lats, masked_vals, cmap="hot_r", vmin=0, vmax=100,
                    shading="auto", transform=proj, zorder=2, alpha=0.85
                )

                # Station markers + labels for quick orientation
                for stn_id, coords in STN_COORDS.items():
                    ax.plot(
                        coords["lon"], coords["lat"], marker="^", markersize=6,
                        color="#2563eb", markeredgecolor="white", markeredgewidth=0.8,
                        transform=proj, zorder=5
                    )
                    ax.text(
                        coords["lon"] + 0.06, coords["lat"] + 0.05,
                        stn_id.upper(), fontsize=6, fontweight="bold", color="#1e3a5f",
                        transform=proj, zorder=6,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.6, edgecolor="none")
                    )

                ax.gridlines(draw_labels=False, linewidth=0.4, color="#94a3b8", alpha=0.5, linestyle="--")

                ax.set_title(f"HREF ≥ {thresh_key[1:]} Flash Density", fontsize=9, fontweight="bold", color="#1e293b")
                cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
                cbar.set_label("Exceedance Probability (%)", fontsize=7)
                cbar.ax.tick_params(labelsize=6)

                maps[thresh_key] = _fig_to_png_file(fig, f"{file_prefix}_{thresh_key}.png")
            except Exception as plot_err:
                logging.error(f"Spatial plot render failed for {thresh_key}: {plot_err}")

        grbs.close()
    except Exception as e:
        logging.error(f"Spatial map extraction failed for {filepath}: {e}")

    return maps


def extract_hrefct_from_file(filepath, lat, lon, stn):
    """Extract the single HREF Calibrated Thunder (HREFCT) probability value at a point.
    HREFCT is one field per file (probability of >=1 CG flash within 20 km), unlike the
    density product's four flash-count thresholds. Returns an int percent 0-100."""
    global _GRID_INDEX_CACHE
    result = 0
    try:
        grbs = pygrib.open(filepath)
        cache_key = f"ct_{stn}"
        if cache_key in _GRID_INDEX_CACHE:
            y_idx, x_idx = _GRID_INDEX_CACHE[cache_key]
        else:
            sample = grbs[1]
            lats, lons = sample.latlons()
            lons_n = np.where(lons > 180, lons - 360.0, lons)
            dist = (lats - lat) ** 2 + (lons_n - lon) ** 2
            y_idx, x_idx = np.unravel_index(dist.argmin(), dist.shape)
            _GRID_INDEX_CACHE[cache_key] = (y_idx, x_idx)

        grbs.seek(0)
        grb = grbs[1]  # single-field product; first message is the calibrated probability
        grid = grb.values
        arr = np.ma.filled(np.ma.masked_invalid(np.ma.asarray(grid, dtype=float)), 0.0)
        arr = np.where((arr > 1e19) | (arr < 0), 0.0, arr)
        scale = 100.0 if arr.max() <= 1.0 else 1.0
        pv = float(arr[y_idx, x_idx]) * scale
        result = int(round(max(0.0, min(100.0, pv))))
        grbs.close()
    except Exception as e:
        logging.error(f"HREFCT extraction failed for {stn.upper()}: {e}")
    return result


def generate_hrefct_map(filepath, file_prefix, window_label):
    """Render a single Florida-domain calibrated-thunder probability map (PNG) from one
    HREFCT GRIB2 file. Returns the relative path, or None on failure."""
    try:
        grbs = pygrib.open(filepath)
        sample = grbs[1]
        lats, lons = sample.latlons()
        lons_n = np.where(lons > 180, lons - 360.0, lons)

        domain_mask = (
            (lats >= FL_DOMAIN["lat_min"]) & (lats <= FL_DOMAIN["lat_max"]) &
            (lons_n >= FL_DOMAIN["lon_min"]) & (lons_n <= FL_DOMAIN["lon_max"])
        )
        ys, xs = np.where(domain_mask)
        if len(ys) == 0:
            grbs.close()
            return None
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        sub_lats = lats[y0:y1 + 1, x0:x1 + 1]
        sub_lons = lons_n[y0:y1 + 1, x0:x1 + 1]

        grbs.seek(0)
        grb = grbs[1]
        raw_vals = grb.values[y0:y1 + 1, x0:x1 + 1]
        vals = np.nan_to_num(np.asarray(raw_vals, dtype=float), nan=0.0)
        vals = np.where((vals > 1e19) | (vals < 0), 0.0, vals)
        if vals.max() <= 1.0:
            vals = vals * 100.0
        grbs.close()

        proj = ccrs.PlateCarree()
        states_provinces = cfeature.NaturalEarthFeature(
            category="cultural", name="admin_1_states_provinces_lines", scale="50m", facecolor="none")
        counties = cfeature.NaturalEarthFeature(
            category="cultural", name="admin_2_counties", scale="10m", facecolor="none")

        fig = plt.figure(figsize=(5.5, 5.8), dpi=120)
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        ax.set_extent([FL_DOMAIN["lon_min"], FL_DOMAIN["lon_max"],
                       FL_DOMAIN["lat_min"], FL_DOMAIN["lat_max"]], crs=proj)
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#dbeafe", zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f1f5f9", zorder=0)
        ax.add_feature(counties, edgecolor="#cbd5e1", linewidth=0.35, zorder=1)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#1e293b", linewidth=0.9, zorder=3)
        ax.add_feature(states_provinces, edgecolor="#475569", linewidth=0.8, zorder=3)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#1e293b", linewidth=0.8, zorder=3)

        masked_vals = np.ma.masked_less_equal(vals, 0.0)
        # Distinct colormap from the density product (which uses hot_r) so the two are
        # visually separable at a glance.
        mesh = ax.pcolormesh(sub_lons, sub_lats, masked_vals, cmap="YlGnBu", vmin=0, vmax=100,
                             shading="auto", transform=proj, zorder=2, alpha=0.85)

        for stn_id, coords in STN_COORDS.items():
            ax.plot(coords["lon"], coords["lat"], marker="^", markersize=6, color="#b91c1c",
                    markeredgecolor="white", markeredgewidth=0.8, transform=proj, zorder=5)
            ax.text(coords["lon"] + 0.06, coords["lat"] + 0.05, stn_id.upper(), fontsize=6,
                    fontweight="bold", color="#7f1d1d", transform=proj, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.6, edgecolor="none"))

        ax.gridlines(draw_labels=False, linewidth=0.4, color="#94a3b8", alpha=0.5, linestyle="--")
        ax.set_title(f"HREF Calibrated Thunder ({window_label})", fontsize=9, fontweight="bold", color="#1e293b")
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("Probability of Lightning (%)", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

        return _fig_to_png_file(fig, f"{file_prefix}.png")
    except Exception as e:
        logging.error(f"HREFCT map render failed for {filepath}: {e}")
        return None


def fetch_href_spatial_map(session, date_str, cycle, f_hour_int):
    """Downloads the HREF lightning GRIB2 once per forecast hour (domain-wide, not
    station-specific) and renders the Florida spatial threshold maps from it."""
    base_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/spc_post/prod/spc_post.{date_str}/ltgdensity"
    file_name = f"spc_post.t{cycle}z.hrefld_4hr.f{f_hour_int:03d}.grib2"
    url = f"{base_url}/{file_name}"
    local_path = os.path.join(CACHE_DIR, f"spatial_{file_name}")

    try:
        with session.get(url, timeout=10, stream=True) as r:
            if r.status_code == 200:
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

                file_prefix = f"{date_str}_{cycle}z_f{f_hour_int:03d}"
                maps = generate_spatial_threshold_maps(local_path, file_prefix)

                if os.path.exists(local_path):
                    os.remove(local_path)
                return maps
    except Exception as e:
        logging.debug(f"Spatial map download break for {file_name}: {e}")

    if os.path.exists(local_path):
        try: os.remove(local_path)
        except Exception: pass
    return {}


def fetch_href_lightning_point(session, stn, lat, lon, date_str, cycle, f_hour_int):
    base_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/spc_post/prod/spc_post.{date_str}/ltgdensity"
    file_name = f"spc_post.t{cycle}z.hrefld_4hr.f{f_hour_int:03d}.grib2"
    url = f"{base_url}/{file_name}"
    local_path = os.path.join(CACHE_DIR, f"{stn}_{file_name}")

    try:
        with session.get(url, timeout=7, stream=True) as r:
            if r.status_code == 200:
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

                vals = extract_lightning_from_file(local_path, lat, lon, stn)

                if os.path.exists(local_path):
                    os.remove(local_path)
                return stn, vals
    except Exception as e:
        logging.debug(f"Download break for {file_name}: {e}")

    if os.path.exists(local_path):
        try: os.remove(local_path)
        except Exception: pass
    return stn, {"p25": 0, "p50": 0, "p100": 0, "p200": 0}


def extract_ct_point(filepath, y_idx, x_idx):
    """Extract the single calibrated-thunder probability (%) at a grid index from an
    HREFCT GRIB2 file. Returns an int percent, or 0 on any failure. Uses the same
    whole-grid fraction-vs-percent normalization and fill-value sanitizing as the
    lightning-density extractor."""
    try:
        grbs = pygrib.open(filepath)
        grbs.seek(0)
        val = 0
        for grb in grbs:
            grid = grb.values
            try:
                arr = np.ma.filled(np.ma.masked_invalid(np.ma.asarray(grid, dtype=float)), 0.0)
                arr = np.where((arr > 1e19) | (arr < 0), 0.0, arr)
            except (TypeError, ValueError):
                continue
            scale = 100.0 if arr.max() <= 1.0 else 1.0
            y0 = max(0, y_idx - 1); y1 = min(arr.shape[0], y_idx + 2)
            x0 = max(0, x_idx - 1); x1 = min(arr.shape[1], x_idx + 2)
            neigh = arr[y0:y1, x0:x1]
            cell = float(np.median(neigh)) if neigh.size else float(arr[y_idx, x_idx])
            pv = max(0.0, min(100.0, cell * scale))
            val = int(round(pv))
            break  # HREFCT files carry a single probability message
        grbs.close()
        return val
    except Exception as e:
        logging.error(f"HREFCT extraction failed: {e}")
        return 0


def generate_ct_map(filepath, out_filename):
    """Render a single Florida-domain spatial map of the calibrated-thunder probability
    field (0-100%). Returns the relative maps/ path, or None on failure."""
    try:
        grbs = pygrib.open(filepath)
        grb = grbs[1]
        lats, lons = grb.latlons()
        lons_n = np.where(lons > 180, lons - 360.0, lons)
        vals = np.ma.filled(np.ma.masked_invalid(np.ma.asarray(grb.values, dtype=float)), 0.0)
        vals = np.where((vals > 1e19) | (vals < 0), 0.0, vals)
        if vals.max() <= 1.0:
            vals = vals * 100.0
        grbs.close()

        domain_mask = (
            (lats >= FL_DOMAIN["lat_min"]) & (lats <= FL_DOMAIN["lat_max"]) &
            (lons_n >= FL_DOMAIN["lon_min"]) & (lons_n <= FL_DOMAIN["lon_max"])
        )
        ys, xs = np.where(domain_mask)
        if len(ys) == 0:
            return None
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        sub_lats = lats[y0:y1 + 1, x0:x1 + 1]
        sub_lons = lons_n[y0:y1 + 1, x0:x1 + 1]
        sub_vals = vals[y0:y1 + 1, x0:x1 + 1]

        proj = ccrs.PlateCarree()
        states = cfeature.NaturalEarthFeature(category="cultural",
                    name="admin_1_states_provinces_lines", scale="50m", facecolor="none")
        counties = cfeature.NaturalEarthFeature(category="cultural",
                    name="admin_2_counties", scale="10m", facecolor="none")

        fig = plt.figure(figsize=(5.5, 5.8), dpi=120)
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        ax.set_extent([FL_DOMAIN["lon_min"], FL_DOMAIN["lon_max"],
                       FL_DOMAIN["lat_min"], FL_DOMAIN["lat_max"]], crs=proj)
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#dbeafe", zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f1f5f9", zorder=0)
        ax.add_feature(counties, edgecolor="#cbd5e1", linewidth=0.35, zorder=1)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#1e293b", linewidth=0.9, zorder=3)
        ax.add_feature(states, edgecolor="#475569", linewidth=0.8, zorder=3)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#1e293b", linewidth=0.8, zorder=3)

        masked = np.ma.masked_less_equal(sub_vals, 0.0)
        mesh = ax.pcolormesh(sub_lons, sub_lats, masked, cmap="plasma_r", vmin=0, vmax=100,
                             shading="auto", transform=proj, zorder=2, alpha=0.85)
        for sid, c in STN_COORDS.items():
            ax.plot(c["lon"], c["lat"], marker="^", markersize=6, color="#2563eb",
                    markeredgecolor="white", markeredgewidth=0.8, transform=proj, zorder=5)
            ax.text(c["lon"] + 0.06, c["lat"] + 0.05, sid.upper(), fontsize=6,
                    fontweight="bold", color="#1e3a5f", transform=proj, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.6, edgecolor="none"))
        ax.gridlines(draw_labels=False, linewidth=0.4, color="#94a3b8", alpha=0.5, linestyle="--")
        ax.set_title("HREF Calibrated Thunder Probability", fontsize=9, fontweight="bold", color="#1e293b")
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("Thunder Probability (%)", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

        os.makedirs(MAPS_DIR, exist_ok=True)
        out_path = os.path.join(MAPS_DIR, out_filename)
        fig.savefig(out_path, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        return f"maps/{out_filename}"
    except Exception as e:
        logging.error(f"HREFCT map render failed: {e}")
        return None


def _sanitize_grid(grid):
    """masked / NaN / huge-fill (>1e19) / negative values -> 0.0. Returns a float ndarray."""
    try:
        arr = np.ma.filled(np.ma.masked_invalid(np.ma.asarray(grid, dtype=float)), 0.0)
    except (TypeError, ValueError):
        arr = np.zeros_like(np.asarray(grid, dtype=float))
    return np.where((arr > 1e19) | (arr < 0), 0.0, arr)


def _render_ct_domain_map(sub_lons, sub_lats, sub_vals_pct, out_filename, window_label):
    """Render an FL-domain calibrated-thunder map from an ALREADY percent-scaled (0-100)
    subgrid. Kept separate from extraction so the exact same run-level scale drives both the
    table numbers and the map colors. Returns 'maps/<out_filename>' or None."""
    try:
        proj = ccrs.PlateCarree()
        states = cfeature.NaturalEarthFeature(category="cultural",
                    name="admin_1_states_provinces_lines", scale="50m", facecolor="none")
        counties = cfeature.NaturalEarthFeature(category="cultural",
                    name="admin_2_counties", scale="10m", facecolor="none")

        fig = plt.figure(figsize=(5.5, 5.8), dpi=120)
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        ax.set_extent([FL_DOMAIN["lon_min"], FL_DOMAIN["lon_max"],
                       FL_DOMAIN["lat_min"], FL_DOMAIN["lat_max"]], crs=proj)
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#dbeafe", zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f1f5f9", zorder=0)
        ax.add_feature(counties, edgecolor="#cbd5e1", linewidth=0.35, zorder=1)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#1e293b", linewidth=0.9, zorder=3)
        ax.add_feature(states, edgecolor="#475569", linewidth=0.8, zorder=3)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#1e293b", linewidth=0.8, zorder=3)

        masked = np.ma.masked_less_equal(sub_vals_pct, 0.0)
        mesh = ax.pcolormesh(sub_lons, sub_lats, masked, cmap="YlGnBu", vmin=0, vmax=100,
                             shading="auto", transform=proj, zorder=2, alpha=0.85)
        for sid, c in STN_COORDS.items():
            ax.plot(c["lon"], c["lat"], marker="^", markersize=6, color="#b91c1c",
                    markeredgecolor="white", markeredgewidth=0.8, transform=proj, zorder=5)
            ax.text(c["lon"] + 0.06, c["lat"] + 0.05, sid.upper(), fontsize=6,
                    fontweight="bold", color="#7f1d1d", transform=proj, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.6, edgecolor="none"))
        ax.gridlines(draw_labels=False, linewidth=0.4, color="#94a3b8", alpha=0.5, linestyle="--")
        ax.set_title(f"HREF Calibrated Thunder ({window_label})", fontsize=9, fontweight="bold", color="#1e293b")
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("Probability of Lightning (%)", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

        return _fig_to_png_file(fig, out_filename)
    except Exception as e:
        logging.error(f"CT domain map render failed: {e}")
        return None


def fetch_calibrated_thunder(window="4hr"):
    """Fetch HREF Calibrated Thunder (HREFCT) for the given accumulation window ('1hr' or
    '4hr') across the 1-48h forecast range. Returns (ct_points, ct_maps):
      ct_points: {stn: {row_key: prob_pct}}
      ct_maps:   {row_key: 'maps/....png' | None}
    The product is a single ML-calibrated probability of >=1 CG flash within 20 km."""
    ct_points = {stn: {} for stn in STATIONS}
    ct_maps = {}

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(pool_connections=50, pool_maxsize=50, max_retries=3))

    # Robust cycle discovery. SPC HREFCT initializes at 00Z and 12Z. NOMADS frequently throttles
    # a GitHub-Actions IP right after the HREF-lightning download burst that runs just before this,
    # so a single 3-second HEAD is fragile: a throttled probe times out and the run looks "absent"
    # even when it's on disk — and because every probe (today AND the older fallbacks) times out
    # together, the whole product silently drops. Fix: browser User-Agent, longer timeout, retries
    # with backoff, and a newest-first walk across 3 days so a valid fallback is always found.
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"
    })

    def _probe_exists(url):
        for attempt in range(3):
            try:
                r = session.head(url, timeout=12, allow_redirects=True)
                if r.status_code == 200:
                    return True
                if r.status_code == 404:
                    return False  # definitively absent — don't burn retries on it
            except Exception:
                pass
            time.sleep(1.5 * (attempt + 1))  # brief backoff to ride out throttling
        return False

    active_cycle = active_date_str = None
    candidates = []
    for days_back in [0, 1, 2]:
        d = (now_utc - datetime.timedelta(days=days_back)).strftime("%Y%m%d")
        for cyc in ["12", "00"]:
            init = datetime.datetime.strptime(f"{d}{cyc}", "%Y%m%d%H").replace(tzinfo=datetime.timezone.utc)
            if init <= now_utc:  # skip cycles that haven't run yet
                candidates.append((init, d, cyc))
    candidates.sort(key=lambda x: x[0], reverse=True)  # newest available cycle first

    for _init, d, cyc in candidates:
        # Probe f004: the first forecast hour valid for BOTH the 1-hr and 4-hr windows (a 4-hr
        # accumulation can't end at f001), so it reliably signals "this cycle exists".
        probe = (f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/spc_post/prod/"
                 f"spc_post.{d}/thunder/spc_post.t{cyc}z.hrefct_{window}.f004.grib2")
        if _probe_exists(probe):
            active_cycle, active_date_str = cyc, d
            break

    if not active_cycle:
        logging.warning(f"No active HREFCT {window} cycle found on NOMADS.")
        return ct_points, ct_maps

    logging.info(f"HREFCT {window}: targeting {active_date_str} {active_cycle}z")
    cycle_init = datetime.datetime.strptime(f"{active_date_str}{active_cycle}", "%Y%m%d%H").replace(tzinfo=datetime.timezone.utc)

    window_label = "1-hr" if window == "1hr" else "4-hr"

    # ---- PHASE A: download every hour, pull RAW (unscaled) FL-domain subgrid + point values.
    # The fraction-vs-percent decision is deliberately NOT made per file. A quiet hour whose
    # entire domain is < 1.0 (a genuine 0.8% field) is indistinguishable from a 0-1 fraction
    # when looked at in isolation — that per-file guess is exactly what turned a real ~1% into
    # a bogus 100%. We instead gather the raw maximum across ALL forecast hours and the whole
    # CONUS grid, then decide the scale ONCE below.
    def _dl_worker(f_hour_int, row_key):
        base = (f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/spc_post/prod/"
                f"spc_post.{active_date_str}/thunder")
        fname = f"spc_post.t{active_cycle}z.hrefct_{window}.f{f_hour_int:03d}.grib2"
        url = f"{base}/{fname}"
        local_path = os.path.join(CACHE_DIR, f"ct_{window}_{fname}")
        try:
            with session.get(url, timeout=10, stream=True) as r:
                if r.status_code != 200:
                    return row_key, None
                with open(local_path, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=8192):
                        fh.write(chunk)

            grbs = pygrib.open(local_path)
            grb = grbs[1]  # single-field product: message 1 is the calibrated probability
            lats, lons = grb.latlons()
            lons_n = np.where(lons > 180, lons - 360.0, lons)
            arr = _sanitize_grid(grb.values)
            grbs.close()

            raw_max = float(arr.max()) if arr.size else 0.0

            # FL-domain subset (indices depend only on the static grid geometry).
            domain_mask = (
                (lats >= FL_DOMAIN["lat_min"]) & (lats <= FL_DOMAIN["lat_max"]) &
                (lons_n >= FL_DOMAIN["lon_min"]) & (lons_n <= FL_DOMAIN["lon_max"])
            )
            ys, xs = np.where(domain_mask)
            sub_lats = sub_lons = sub_vals = None
            if len(ys) > 0:
                y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
                sub_lats = lats[y0:y1 + 1, x0:x1 + 1]
                sub_lons = lons_n[y0:y1 + 1, x0:x1 + 1]
                sub_vals = arr[y0:y1 + 1, x0:x1 + 1]

            # Per-station RAW point value; neighborhood median rejects lone fill artifacts.
            pts = {}
            for stn, c in STN_COORDS.items():
                cache_key = f"ct_{stn}"
                if cache_key in _GRID_INDEX_CACHE:
                    yi, xi = _GRID_INDEX_CACHE[cache_key]
                else:
                    dist = (lats - c["lat"]) ** 2 + (lons_n - c["lon"]) ** 2
                    yi, xi = np.unravel_index(dist.argmin(), dist.shape)
                    _GRID_INDEX_CACHE[cache_key] = (yi, xi)
                yy0 = max(0, yi - 1); yy1 = min(arr.shape[0], yi + 2)
                xx0 = max(0, xi - 1); xx1 = min(arr.shape[1], xi + 2)
                neigh = arr[yy0:yy1, xx0:xx1]
                pts[stn] = float(np.median(neigh)) if neigh.size else float(arr[yi, xi])

            return row_key, {"f": f_hour_int, "raw_max": raw_max, "points": pts,
                             "sub_lats": sub_lats, "sub_lons": sub_lons, "sub_vals": sub_vals}
        except Exception as e:
            logging.debug(f"HREFCT {window} f{f_hour_int:03d} break: {e}")
            return row_key, None
        finally:
            if os.path.exists(local_path):
                try: os.remove(local_path)
                except Exception: pass

    records = {}
    global_raw_max = 0.0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for f_hour_int in range(1, 49):
            valid_dt = cycle_init + datetime.timedelta(hours=f_hour_int)
            row_key = f"{valid_dt.day:02d}/{valid_dt.hour:02d}"
            futures.append(executor.submit(_dl_worker, f_hour_int, row_key))
        for fut in concurrent.futures.as_completed(futures):
            try:
                row_key, rec = fut.result()
            except Exception:
                continue
            if rec is None:
                continue
            records[row_key] = rec
            if rec["raw_max"] > global_raw_max:
                global_raw_max = rec["raw_max"]

    # ---- Decide the scale ONCE for the whole run.
    # If the raw max never exceeds 1.0 across the entire CONUS grid AND all 48 forecast hours,
    # the product is a 0-1 fraction -> x100. Otherwise it is already stored as percent (0-100)
    # and must NOT be rescaled. A true percent field essentially always tops 1% somewhere over
    # 48h, so this cleanly separates the two encodings and kills the per-file 100% misfire.
    run_scale = 100.0 if global_raw_max <= 1.0 else 1.0
    logging.info(f"HREFCT {window}: global raw max={global_raw_max:.4g} -> applying x{run_scale:g}")

    # ---- PHASE B: apply the single scale, populate points, and render maps.
    render_futs = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as rex:
        for row_key, rec in records.items():
            for stn in STATIONS:
                raw = rec["points"].get(stn, 0.0)
                ct_points[stn][row_key] = int(round(max(0.0, min(100.0, raw * run_scale))))
            if rec["sub_vals"] is not None:
                sub_pct = np.clip(rec["sub_vals"] * run_scale, 0.0, 100.0)
                out_name = f"ct_{window}_{active_date_str}_{active_cycle}z_f{rec['f']:03d}.png"
                fut = rex.submit(_render_ct_domain_map, rec["sub_lons"], rec["sub_lats"],
                                 sub_pct, out_name, window_label)
                render_futs[fut] = row_key
            else:
                ct_maps[row_key] = None
        for fut in concurrent.futures.as_completed(render_futs):
            rk = render_futs[fut]
            try:
                ct_maps[rk] = fut.result()
            except Exception:
                ct_maps[rk] = None

    n_maps = sum(1 for v in ct_maps.values() if v)
    logging.info(f"HREFCT {window}: points for {len(records)} hours, {n_maps} maps rendered.")
    return ct_points, ct_maps


def fetch_href_lightning(time_keys):
    # href_data is initialized empty and populated only for the HREF 1-48h window below,
    # NOT pre-seeded from time_keys (which spans the full multi-day sounding range and would
    # otherwise leak far-future zero-value keys into the lightning slider).
    href_data = {stn: {} for stn in STATIONS}

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=50, pool_maxsize=50, max_retries=3)
    session.mount("https://", adapter)

    active_cycle = None
    active_date_str = None

    # HREF Lightning density files ONLY initialize for 00Z and 12Z cycles
    for days_back in [0, 1]:
        check_date = now_utc - datetime.timedelta(days=days_back)
        date_str = check_date.strftime("%Y%m%d")

        for cycle in ["12", "00"]:
            test_url = (
                f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/spc_post/prod/"
                f"spc_post.{date_str}/ltgdensity/spc_post.t{cycle}z.hrefld_4hr.f004.grib2"
            )
            try:
                if session.head(test_url, timeout=3).status_code == 200:
                    active_cycle = cycle
                    active_date_str = date_str
                    break
            except Exception:
                continue
        if active_cycle:
            break

    if not active_cycle:
        logging.warning("No active 00Z or 12Z SPC HREF lightning directories found on NOMADS.")
        return href_data, {}

    logging.info(f"Targeting Valid NOMADS Initialization Run: {active_date_str} at {active_cycle}Z")

    cycle_init_utc = datetime.datetime.strptime(f"{active_date_str}{active_cycle}", "%Y%m%d%H").replace(tzinfo=datetime.timezone.utc)

    # Build the full 1-48h valid-time list directly from the cycle init, independent of
    # the sounding matrix time keys (which only cover what the models happen to provide).
    # This guarantees HREF data always spans the full 48h window including tomorrow's
    # diurnal maximum, not just whatever hours the soundings happened to cover.
    all_href_time_keys = {}  # row_key -> f_hour_int
    for f_hour_int in range(1, 49):
        valid_dt = cycle_init_utc + datetime.timedelta(hours=f_hour_int)
        # Zero-pad the day to exactly match the sounding-matrix key format ("%d/%H"),
        # otherwise "1/06" and "01/06" collide as two separate rows for the same hour.
        row_key = f"{valid_dt.day:02d}/{valid_dt.hour:02d}"
        all_href_time_keys[row_key] = f_hour_int

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        futures_map = {}
        for row_key, f_hour_int in all_href_time_keys.items():
            for stn, coords in STN_COORDS.items():
                future = executor.submit(
                    fetch_href_lightning_point,
                    session, stn, coords["lat"], coords["lon"],
                    active_date_str, active_cycle, f_hour_int
                )
                futures_map[future] = (stn, row_key)

        for future in concurrent.futures.as_completed(futures_map):
            stn, row_key = futures_map[future]
            try:
                _, vals = future.result()
                href_data[stn][row_key] = vals
            except Exception:
                pass

    # NOTE: the HREF flash-density spatial PLOTS were retired (the calibrated-thunder maps
    # replaced them). We keep the density POINT audit below for situational awareness, but no
    # longer render or ship the multi-threshold density map PNGs.
    href_maps = {}

    # Streamlined runner-safe Audit Log
    logging.info("=============================================")
    logging.info("         HREF LIGHTNING AUDIT LOG            ")
    logging.info("=============================================")
    total_signals = 0
    for stn in STATIONS:
        stn_hits = 0
        max_p25 = 0
        for r_key, thresh_vals in href_data[stn].items():
            if any(v > 0 for v in thresh_vals.values()):
                stn_hits += 1
                total_signals += 1
                max_p25 = max(max_p25, thresh_vals["p25"])

        if stn_hits > 0:
            logging.info(f"  {stn.upper()} -> Processed {stn_hits} intervals with active lightning signals (Max p25: {max_p25}%)")
        else:
            logging.info(f"  {stn.upper()} -> All 48 forecast intervals returned flat 0%")

    logging.info(f"Audit Complete: Cleanly tracked {total_signals} total non-zero cell vectors.")
    logging.info("=============================================")

    return href_data, href_maps


def _interp_logp(layers, target_p, key):
    """Linear-in-ln(p) interpolation of `key` to target pressure (mb)."""
    below = above = None
    for L in layers:
        if L.get(key) is None:
            continue
        if L["pres"] >= target_p and (below is None or L["pres"] < below["pres"]):
            below = L
        if L["pres"] <= target_p and (above is None or L["pres"] > above["pres"]):
            above = L
    if below is None or above is None:
        return None
    if below["pres"] == above["pres"]:
        return below[key]
    f = (math.log(target_p) - math.log(below["pres"])) / (math.log(above["pres"]) - math.log(below["pres"]))
    return below[key] + f * (above[key] - below[key])


def _sat_vap(tc):
    """Saturation vapor pressure (hPa) over water, Bolton 1980; tc in C."""
    return 6.112 * math.exp(17.67 * tc / (tc + 243.5))


def _mixing_ratio_gkg(td_c, p_mb):
    e = _sat_vap(td_c)
    return 621.97 * e / (p_mb - e)


def _theta_e(tk, tdk, p):
    """Bolton 1980 eq 43 equivalent potential temperature. tk,tdk in K, p in hPa."""
    e = _sat_vap(tdk - 273.15)
    r = 0.62197 * e / (p - e)
    tlcl = 1.0 / (1.0 / (tdk - 56.0) + math.log(tk / tdk) / 800.0) + 56.0
    return tk * (1000.0 / p) ** (0.2854 * (1.0 - 0.28 * r)) * \
        math.exp((3.376 / tlcl - 0.00254) * r * 1000.0 * (1.0 + 0.81 * r))


try:
    import metpy.calc as _mpcalc
    from metpy.units import units as _mpunits
    _HAVE_METPY = True
except Exception:
    _HAVE_METPY = False

_ML_DEPTH_HPA = 100.0  # mixed-layer parcel depth (lowest 100 hPa) for the Lifted Index


def _layer_mean_rh(layers, p_bot, p_top):
    """Mean relative humidity (%) over [p_top, p_bot] mb. Uses each layer's RH via _layer_rh, which
    is native GRIB RH on the pad/GRIB paths and Magnus-derived from T/Td on BUFKIT."""
    vals = []
    for L in layers:
        if p_top <= L.get("pres", 0) <= p_bot:
            rh = _layer_rh(L) if "_layer_rh" in globals() else L.get("rh")
            if rh is not None:
                vals.append(rh)
    return sum(vals) / len(vals) if vals else None


def _layer_mean_flow(layers, p_bot, p_top, prefix):
    """Vector-mean wind over [p_top, p_bot] mb: FROM-direction, speed (kt), 8-pt compass regime, and
    the mean u/v components (kept so an ensemble can be averaged in component space). Keys are
    prefixed (e.g. 'mf' -> mf_dir/mf_spd/..., 'av' -> av_dir/...)."""
    us = [L["u"] for L in layers if p_top <= L["pres"] <= p_bot and L.get("u") is not None]
    vs = [L["v"] for L in layers if p_top <= L["pres"] <= p_bot and L.get("v") is not None]
    if not us:
        return {}
    um, vm = sum(us) / len(us), sum(vs) / len(vs)
    frm = math.degrees(math.atan2(-um, -vm)) % 360.0
    compass = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int((frm + 22.5) // 45) % 8]
    return {f"{prefix}_dir": round(frm), f"{prefix}_spd": round(math.hypot(um, vm), 1),
            f"{prefix}_regime": compass, f"{prefix}_u": round(um, 3), f"{prefix}_v": round(vm, 3)}


def _layer_mean_rh(layers, p_bot, p_top):
    """Mean relative humidity (%) over [p_top, p_bot] mb. Uses native GRIB RH where present and
    Magnus-derived RH (from T/Td) otherwise — consistent with the cloud logic."""
    rhs = []
    for L in layers:
        if not (p_top <= L.get("pres", -1) <= p_bot):
            continue
        rh = L.get("rh")
        if rh is None and L.get("tmpc") is not None and L.get("dwpt") is not None:
            a, b = 17.625, 243.04
            es = lambda x: math.exp((a * x) / (b + x))
            rh = max(0.0, min(100.0, 100.0 * es(L["dwpt"]) / es(L["tmpc"])))
        if rh is not None:
            rhs.append(rh)
    return sum(rhs) / len(rhs) if rhs else None


# ---- Lightning probability (coworker's RandomForest, exported to a dependency-free numpy file) ----
# Features, in the model's own order: [Thompson_Index, 1000-700mb Average U-Wind Component (kt,
# eastward +), 700-500mb Average RH (%)]. The .sav was trained under scikit-learn 1.3.2; rather than
# pin that (it conflicts with the pipeline's numpy 2.x), the 500 trees were extracted to
# LIGHTNING_MODEL_PATH and are evaluated here in pure numpy. Verified bit-for-bit (max abs diff 0.0)
# against sklearn 1.3.2 predict_proba, so this reproduces the coworker's tool exactly.
LIGHTNING_ENABLED = True
LIGHTNING_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lightning_rf_10Z.npz")
_LTG_RF = None            # cached tree arrays
_LTG_RF_TRIED = False


def _load_lightning_rf():
    global _LTG_RF, _LTG_RF_TRIED
    if _LTG_RF_TRIED:
        return _LTG_RF
    _LTG_RF_TRIED = True
    try:
        d = np.load(LIGHTNING_MODEL_PATH, allow_pickle=True)
        _LTG_RF = {
            "offsets": d["offsets"], "cl": d["children_left"], "cr": d["children_right"],
            "feat": d["feature"], "thr": d["threshold"], "val": d["value"],
            "n_trees": int(d["n_trees"][0]),
            "names": [str(x) for x in d["feature_names"]],
        }
        logging.info(f"Lightning RF loaded: {_LTG_RF['n_trees']} trees, features={_LTG_RF['names']}.")
    except Exception as e:
        logging.warning(f"Lightning RF unavailable ({e}); lightning column will be blank.")
        _LTG_RF = None
    return _LTG_RF


def lightning_probability(thompson, mf_dir, mf_spd, rh_700_500):
    """P(lightning) in %, from the coworker's RandomForest. U-wind is rebuilt from the 1000-700 mb
    mean flow with the coworker's own convention: speed * cos(270 - dir) == eastward component (kt).
    Returns None if any feature is missing or the model isn't loaded."""
    if not LIGHTNING_ENABLED:
        return None
    if thompson is None or mf_dir is None or mf_spd is None or rh_700_500 is None:
        return None
    rf = _load_lightning_rf()
    if rf is None:
        return None
    try:
        u = mf_spd * math.cos(math.radians(270.0 - mf_dir))          # eastward component, kt
        x = np.array([float(thompson), float(u), float(rh_700_500)], dtype=float)
        offs, cl, cr, feat, thr, val = (rf["offsets"], rf["cl"], rf["cr"],
                                        rf["feat"], rf["thr"], rf["val"])
        acc = 0.0
        for ti in range(rf["n_trees"]):
            a = offs[ti]
            node = 0
            while cl[a + node] != -1:                                # -1 == leaf (TREE_LEAF)
                node = (cl[a + node] if x[feat[a + node]] <= thr[a + node] else cr[a + node])
            leaf = val[a + node]                                     # [count_class0, count_class1]
            s = leaf[0] + leaf[1]
            if s > 0:
                acc += leaf[1] / s
        return round(acc / rf["n_trees"] * 100.0, 1)
    except Exception as e:
        logging.debug(f"lightning_probability failed: {e}")
        return None


def _thermo_metpy(layers):
    """K-Index, PWAT, and MIXED-LAYER Lifted Index (+ Thompson = KI-LI) via MetPy."""
    P, T, D = [], [], []
    for L in layers:
        if None in (L.get("pres"), L.get("tmpc"), L.get("dwpt")):
            continue
        P.append(L["pres"]); T.append(L["tmpc"]); D.append(min(L["dwpt"], L["tmpc"]))
    if len(P) < 4:
        return {}
    p = np.array(P) * _mpunits.hPa
    Tq = np.array(T) * _mpunits.degC
    Tdq = np.array(D) * _mpunits.degC

    def _scal(q):
        return float(np.atleast_1d(q.magnitude)[0])

    out = {}
    ki = _scal(_mpcalc.k_index(p, Tq, Tdq))
    out["k_index"] = round(ki, 1)
    pw_in = _scal(_mpcalc.precipitable_water(p, Tdq).to("inch"))
    out["pwat_in"] = round(pw_in, 2)
    out["pwat_mm"] = round(pw_in * 25.4, 1)
    # Mixed-layer parcel over the lowest _ML_DEPTH_HPA, lifted to 500 mb
    _, mp_T, mp_Td = _mpcalc.mixed_parcel(p, Tq, Tdq, depth=_ML_DEPTH_HPA * _mpunits.hPa)
    prof = _mpcalc.parcel_profile(p, mp_T, mp_Td).to("degC")
    li = _scal(_mpcalc.lifted_index(p, Tq, prof))
    out["lifted_index"] = round(li, 1)
    out["thompson"] = round(ki - li, 1)
    out["parcel"] = "mixed-layer"
    out["engine"] = "metpy"
    return out


def _thermo_numpy(layers):
    """Fallback for when MetPy isn't installed: KI, PWAT, and a MIXED-LAYER Lifted Index built by
    mixing potential temperature and mixing ratio over the lowest _ML_DEPTH_HPA, then lifting via
    theta-e conservation to 500 mb."""
    out = {}
    t850 = _interp_logp(layers, 850, "tmpc"); td850 = _interp_logp(layers, 850, "dwpt")
    t700 = _interp_logp(layers, 700, "tmpc"); td700 = _interp_logp(layers, 700, "dwpt")
    t500 = _interp_logp(layers, 500, "tmpc")
    ki = None
    if None not in (t850, td850, t700, td700, t500):
        ki = (t850 - t500) + td850 - (t700 - td700)
        out["k_index"] = round(ki, 1)
    li = None
    p_sfc = layers[0]["pres"]
    ml = [L for L in layers if L.get("tmpc") is not None and L.get("dwpt") is not None
          and (p_sfc - L["pres"]) <= _ML_DEPTH_HPA]
    if t500 is not None and ml:
        theta_ml = sum((L["tmpc"] + 273.15) * (1000.0 / L["pres"]) ** 0.2854 for L in ml) / len(ml)
        w_ml = sum(_mixing_ratio_gkg(L["dwpt"], L["pres"]) / 1000.0 for L in ml) / len(ml)
        ml_tk = theta_ml * (p_sfc / 1000.0) ** 0.2854
        e = max(w_ml * p_sfc / (0.62197 + w_ml), 1e-6)
        ml_tdc = 243.5 * math.log(e / 6.112) / (17.67 - math.log(e / 6.112))
        thetae_parcel = _theta_e(ml_tk, ml_tdc + 273.15, p_sfc)
        lo, hi = 200.0, 320.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if _theta_e(mid, mid, 500.0) < thetae_parcel:
                lo = mid
            else:
                hi = mid
        li = t500 - (0.5 * (lo + hi) - 273.15)
        out["lifted_index"] = round(li, 1)
    if ki is not None and li is not None:
        out["thompson"] = round(ki - li, 1)
    wl = [L for L in layers if L.get("dwpt") is not None]
    if len(wl) >= 3:
        tot = 0.0
        for i in range(len(wl) - 1):
            p1, p2 = wl[i]["pres"], wl[i + 1]["pres"]
            w1 = _mixing_ratio_gkg(wl[i]["dwpt"], p1) / 1000.0
            w2 = _mixing_ratio_gkg(wl[i + 1]["dwpt"], p2) / 1000.0
            tot += 0.5 * (w1 + w2) * ((p1 - p2) * 100.0) / 9.81
        out["pwat_mm"] = round(tot, 1)
        out["pwat_in"] = round(tot / 25.4, 2)
    out["parcel"] = "mixed-layer"
    out["engine"] = "numpy"
    return out


def _rh_of_layer(l):
    """Relative humidity (%) for a layer: native GRIB 'rh' when present, else Magnus-derived from
    T/Td (BUFKIT path). Mirrors the _layer_rh helper used for cloud decks."""
    rh = l.get("rh")
    if rh is not None:
        return rh
    t, td = l.get("tmpc"), l.get("dwpt")
    if t is None or td is None:
        return None
    a, b = 17.625, 243.04
    es = lambda x: math.exp((a * x) / (b + x))
    return max(1.0, min(100.0, 100.0 * es(min(td, t)) / es(t)))


def _layer_mean_rh(layers, p_bot, p_top):
    """Mean RH (%) through [p_top, p_bot] mb."""
    vals = [_rh_of_layer(l) for l in layers if p_top <= l["pres"] <= p_bot]
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 1) if vals else None


# ---- Cizek lightning-probability random forest (KSC/CCSFS, 10Z sounding) --------------------
# Source model: cyclonecizek/LightningProbabilityTool, RFC_model_limited_depth_10Z_updated.sav
# (RandomForestClassifier, 500 trees, max_depth 6, scikit-learn 1.3.2).
#
# That pickle requires numpy<2 and cannot be loaded alongside this pipeline's numpy 2.x, so the
# forest was exported to a plain .npz of per-tree arrays and is evaluated here in pure numpy. The
# extraction was verified to reproduce sklearn's predict_proba EXACTLY (max abs diff 0.0) over a
# validation grid, so this is a re-implementation of the same model, not an approximation.
#
# Features, in this exact order (model.feature_names_in_):
#   Thompson_Index                        -- K-Index minus Lifted Index
#   1000-700mb_Average_U-Wind_Component   -- NOTE: the upstream tool defines this as
#                                            speed_kt * cos(deg2rad(270 - direction)), i.e. a
#                                            WESTERLY-POSITIVE component in KNOTS off the
#                                            meteorological FROM-direction. This is NOT the
#                                            standard math-convention u, so it is rebuilt with the
#                                            upstream formula rather than reusing mf_u.
#   700-500mb_Average_RH                  -- percent
RF_LIGHTNING_ENABLED = True
RF_LIGHTNING_NPZ = "rf_lightning_10Z.npz"
_RF_LTG = None
_RF_LTG_TRIED = False


def _rf_lightning_load():
    """Load the exported forest once. Returns the arrays dict, or None if unavailable (in which
    case the column simply shows '-' — a missing model must never fail the run)."""
    global _RF_LTG, _RF_LTG_TRIED
    if _RF_LTG_TRIED:
        return _RF_LTG
    _RF_LTG_TRIED = True
    if not RF_LIGHTNING_ENABLED:
        return None
    try:
        path = RF_LIGHTNING_NPZ
        if not os.path.exists(path):
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), RF_LIGHTNING_NPZ)
        d = np.load(path, allow_pickle=True)
        _RF_LTG = {k: d[k] for k in d.files}
        ntrees = len(_RF_LTG["tree_offsets"]) - 1
        logging.info(f"Cizek lightning RF loaded: {ntrees} trees, "
                     f"{int(_RF_LTG['tree_offsets'][-1])} nodes.")
    except Exception as e:
        logging.warning(f"Cizek lightning RF unavailable ({e}); probability column will be blank.")
        _RF_LTG = None
    return _RF_LTG


def rf_lightning_u_wind(mf_dir, mf_spd):
    """Upstream tool's '1000-700mb Average U-Wind Component': westerly-positive knots."""
    if mf_dir is None or mf_spd is None:
        return None
    return mf_spd * math.cos(math.radians(270.0 - mf_dir))


def rf_lightning_prob(thompson, u_wind, rh_700_500):
    """P(lightning) in percent for one 10Z environment, or None if a feature is missing."""
    r = _rf_lightning_load()
    if r is None or thompson is None or u_wind is None or rh_700_500 is None:
        return None
    try:
        X = np.array([[float(thompson), float(u_wind), float(rh_700_500)]], dtype=float)
        cl, cr = r["children_left"], r["children_right"]
        fe, th, va, off = r["feature"], r["threshold"], r["value"], r["tree_offsets"]
        acc = np.zeros((1, va.shape[1]), dtype=float)
        for t in range(len(off) - 1):
            a, b = int(off[t]), int(off[t + 1])
            tcl, tcr, tfe, tth = cl[a:b], cr[a:b], fe[a:b], th[a:b]
            node = 0
            while tcl[node] != -1:                     # walk to a leaf
                node = tcl[node] if X[0, tfe[node]] <= tth[node] else tcr[node]
            acc += va[a:b][node]
        acc /= (len(off) - 1)
        c1 = int(np.where(r["classes"] == 1.0)[0][0])
        return round(float(acc[0, c1]) * 100.0, 1)
    except Exception as e:
        logging.debug(f"Cizek lightning RF predict failed: {e}")
        return None


def compute_launch_thermo(profile_layers):
    """1000-700 mb mean flow (+ regime), K-Index, MIXED-LAYER Lifted Index, Thompson Index (KI-LI),
    and PWAT. Uses MetPy when available and falls back to an equivalent numpy implementation.
    NOTE: this is now called on demand for the launch-thermo panel only (not per profile), since the
    MetPy path is ~17 ms/profile. Returns {} if the profile is too thin."""
    try:
        layers = sorted([L for L in profile_layers if L.get("pres") is not None],
                        key=lambda x: -x["pres"])
        if len(layers) < 4:
            return {}
        out = dict(_layer_mean_flow(layers, 1000.0, 700.0, "mf"))
        out.update(_layer_mean_flow(layers, 300.0, 150.0, "av"))
        rh75 = _layer_mean_rh(layers, 700.0, 500.0)   # Cizek RF feature 3
        if rh75 is not None:
            out["rh_700_500"] = rh75
        core = _thermo_metpy(layers) if _HAVE_METPY else {}
        if not core:
            core = _thermo_numpy(layers)
        out.update(core)
        # IVT rides along here because every model column in the panel passes through
        # this function, so one call covers GFS/RAP/HRRR/ECMWF/RRFS/REFS/GEFS/ECENS.
        # column_ivt returns None on a column too thin or too shallow to integrate,
        # in which case the keys are simply absent and the panel renders a blank.
        _ivt = column_ivt(layers)
        if _ivt:
            out["ivt"] = _ivt["ivt"]
            out["ivt_dir"] = _ivt["ivt_dir_to"]
        return out
    except Exception:
        return {}


def compute_profile_variables(profile_layers):
    """
    Given a list of profile layers (each a dict with pres/hght/tmpc/dwpt/depr/sknt/u/v),
    compute the full set of aviation + launch variables. Shared by both the BUFKIT
    station path and the raw-GRIB launch-pad path so the math stays identical.
    Returns the per-hour data dict, or None if the profile is unusable.
    """
    if not profile_layers:
        return None
    profile_layers = sorted(profile_layers, key=lambda x: x["pres"], reverse=True)

    def get_height_of_isotherm(target_temp):
        for i in range(len(profile_layers) - 1):
            t1, t2 = profile_layers[i]["tmpc"], profile_layers[i + 1]["tmpc"]
            h1, h2 = profile_layers[i]["hght"], profile_layers[i + 1]["hght"]
            if (t1 >= target_temp >= t2) or (t1 <= target_temp <= t2):
                if t1 == t2:
                    return h1 / 1000.0
                fraction = (target_temp - t1) / (t2 - t1)
                return (h1 + fraction * (h2 - h1)) / 1000.0
        return profile_layers[-1]["hght"] / 1000.0

    def get_wind_component_at_agl(target_agl_ft, sfc_hght):
        """Linearly interpolate u/v wind components to a target height (ft AGL)."""
        for i in range(len(profile_layers) - 1):
            l1, l2 = profile_layers[i], profile_layers[i + 1]
            if l1["u"] is None or l1["v"] is None or l2["u"] is None or l2["v"] is None:
                continue
            agl1 = l1["hght"] - sfc_hght
            agl2 = l2["hght"] - sfc_hght
            if (agl1 <= target_agl_ft <= agl2) or (agl2 <= target_agl_ft <= agl1):
                if agl1 == agl2:
                    return l1["u"], l1["v"]
                fraction = (target_agl_ft - agl1) / (agl2 - agl1)
                u = l1["u"] + fraction * (l2["u"] - l1["u"])
                v = l1["v"] + fraction * (l2["v"] - l1["v"])
                return u, v
        for layer in reversed(profile_layers):
            if layer["u"] is not None and layer["v"] is not None:
                return layer["u"], layer["v"]
        return None, None

    def calc_shear_0_6km():
        """0-6 km AGL bulk shear magnitude (kt), using true wind vector components."""
        sfc_layer = profile_layers[0]
        if sfc_layer["u"] is None or sfc_layer["v"] is None:
            return None
        sfc_hght = sfc_layer["hght"]
        u_sfc, v_sfc = sfc_layer["u"], sfc_layer["v"]
        target_agl_ft = 6000.0 * 3.280839895  # 6 km converted to feet
        u_6km, v_6km = get_wind_component_at_agl(target_agl_ft, sfc_hght)
        if u_6km is None or v_6km is None:
            return None
        return round(math.hypot(u_6km - u_sfc, v_6km - v_sfc), 1)

    def _layer_rh(layer):
        """Relative humidity (%) for a layer. Uses stored 'rh' if present (raw GRIB paths),
        otherwise derives it from temperature/dewpoint (BUFKIT path) via the Magnus formula."""
        rh = layer.get("rh")
        if rh is not None:
            return rh
        t = layer.get("tmpc")
        td = layer.get("dwpt")
        if t is None or td is None:
            return None
        a, b = 17.625, 243.04
        try:
            gt = (a * t) / (b + t)
            gd = (a * td) / (b + td)
            return 100.0 * math.exp(gd - gt)
        except Exception:
            return None

    def _group_cloud_layers(is_cloud_fn):
        """Walk the profile bottom-up, grouping contiguous 'in cloud' levels into decks.
        is_cloud_fn(layer) -> bool decides membership. To avoid a coarse (mandatory-level)
        GRIB column merging widely-separated moist levels into one impossibly-thick deck,
        two consecutive in-cloud levels are only joined if the vertical gap between them is
        at most MAX_LEVEL_GAP_FT; a larger jump breaks the deck (we can't confirm the air
        between two sparse levels is actually cloudy)."""
        MAX_LEVEL_GAP_FT = 5000.0
        decks = []
        active = None
        prev_hght = None
        for layer in profile_layers:
            if is_cloud_fn(layer):
                if active is None:
                    active = {"base": layer["hght"], "top": layer["hght"]}
                elif (layer["hght"] - prev_hght) <= MAX_LEVEL_GAP_FT:
                    active["top"] = layer["hght"]
                else:
                    # Gap too large to trust as a single deck; close current, start new.
                    decks.append(active)
                    active = {"base": layer["hght"], "top": layer["hght"]}
                prev_hght = layer["hght"]
            elif active is not None:
                decks.append(active)
                active = None
                prev_hght = None
        if active:
            decks.append(active)
        return decks

    # Ceiling uses the stricter RH >= 95% criterion: a discrete "is there a solid deck here"
    # test that avoids over-calling MVFR. Cloud TOP and THICKNESS use the more permissive
    # dewpoint-depression <= 2C criterion, which captures the fuller vertical extent of the
    # moist/cloudy layer (RH >= 95% clips the deck edges and undercounts thickness).
    RH_CLOUD_THRESHOLD = 95.0
    # ...but the depr<=2C test is only physical in the water/mixed part of the column. At cirrus
    # temperatures the Magnus depression collapses (at -68C, RH~85% already gives depr~1.3C), so a
    # model with a moist near-tropopause bias — notably GFS on the pad grids — invents a phantom
    # cloud top at 40-47 kft. So above CLOUD_GLACIATION_C we drop depr and demand genuine near-
    # saturation (RH>=95%, or a very small depr when RH is absent, e.g. BUFKIT), which admits real
    # cirrus/anvil but rejects the artifact.
    CLOUD_GLACIATION_C = -40.0

    def _extent_cloud(l):
        t, d = l.get("tmpc"), l.get("depr")
        if t is None or d is None:
            return False
        if t >= CLOUD_GLACIATION_C:
            return d <= 2.0                      # water/mixed cloud: full-extent depression test
        rh = _layer_rh(l)                        # glaciated: require true near-saturation
        return rh >= RH_CLOUD_THRESHOLD if rh is not None else d <= 1.0

    ceiling_decks = _group_cloud_layers(
        lambda l: (_layer_rh(l) is not None and _layer_rh(l) >= RH_CLOUD_THRESHOLD)
    )
    extent_decks = _group_cloud_layers(_extent_cloud)

    # --- Mixed-layer momentum (BUFKIT-style) -------------------------------------
    # Both PBL Mom Mean (transport-style mean wind through the mixed layer) and PBL Mom Max
    # (gust/mixing potential = strongest wind within the mixed layer) are evaluated over the
    # DIAGNOSED mixed-layer depth, not a fixed 850 hPa slab. The mixed-layer top is found by
    # walking up from the surface until potential temperature (theta) rises more than
    # THETA_DELTA_K above the surface value — the classic well-mixed-layer criterion.
    def _theta_k(layer):
        t_k = layer["tmpc"] + 273.15
        return t_k * (1000.0 / layer["pres"]) ** 0.286

    THETA_DELTA_K = 1.5  # K above surface theta that marks the mixed-layer top
    MIN_ML_TOP_FT = 1000.0   # floor so a strong nocturnal inversion still yields a usable layer
    MAX_ML_TOP_FT = 12000.0  # ceiling guard against runaway deep-convective profiles

    sfc_theta = _theta_k(profile_layers[0])
    sfc_hght = profile_layers[0]["hght"]
    ml_top_ft = None
    for layer in profile_layers[1:]:
        if _theta_k(layer) - sfc_theta > THETA_DELTA_K:
            ml_top_ft = layer["hght"]
            break
    if ml_top_ft is None:
        ml_top_ft = profile_layers[-1]["hght"]
    # Clamp the diagnosed top into a sane AGL band
    ml_top_ft = max(sfc_hght + MIN_ML_TOP_FT, min(ml_top_ft, sfc_hght + MAX_ML_TOP_FT))

    ml_winds = [l["sknt"] for l in profile_layers if l["hght"] <= ml_top_ft]
    if not ml_winds:                       # degenerate guard: at least use the surface layer
        ml_winds = [profile_layers[0]["sknt"]]
    mean_wind = sum(ml_winds) / len(ml_winds)
    max_pbl = max(ml_winds)

    sfc_depr = profile_layers[0]["depr"] if profile_layers else 10.0
    vis = 0.25 if sfc_depr <= 0.5 else (1.0 if sfc_depr <= 1.0 else (3.0 if sfc_depr <= 2.0 else 10.0))
    valid_ceilings = [c for c in ceiling_decks if c["base"] >= 100.0]
    ceiling_val = round(valid_ceilings[0]["base"]) if valid_ceilings else 24000.0

    # Thick Cloud Layer LLCC (rule #6): do not fly through a cloud layer >= 4,500 ft thick
    # where any part lies in the 0C to -20C charging band. For each dewpoint-depression deck
    # we test TWO ways it can violate:
    #   (a) the deck itself is >= 4,500 ft thick AND overlaps the [0C, -20C] band, or
    #   (b) the portion of the deck that falls *inside* the band is itself >= 4,500 ft thick
    #       (catches deep clouds that pass through the band even if grouped with layers below).
    h0c_ft = get_height_of_isotherm(0.0) * 1000.0
    h20c_ft = get_height_of_isotherm(-20.0) * 1000.0
    band_lo, band_hi = min(h0c_ft, h20c_ft), max(h0c_ft, h20c_ft)  # 0C is lower, -20C higher
    thick_layer_violated = False
    thickest_in_band_ft = 0.0
    for d in extent_decks:
        depth = max(0.0, d["top"] - d["base"])
        overlaps_band = (d["top"] >= band_lo) and (d["base"] <= band_hi)
        # In-band portion of this deck
        in_band_depth = max(0.0, min(d["top"], band_hi) - max(d["base"], band_lo))
        if (depth >= 4500.0 and overlaps_band) or (in_band_depth >= 4500.0):
            thick_layer_violated = True
            thickest_in_band_ft = max(thickest_in_band_ft, in_band_depth if in_band_depth > 0 else depth)

    return {
        "mom_mean": round(mean_wind, 1),
        "mom_max": round(max_pbl, 1),
        "shear": calc_shear_0_6km(),
        "vis": vis,
        "ceiling": ceiling_val,
        "hght_p5c": round(get_height_of_isotherm(5.0), 1),
        "hght_0c": round(get_height_of_isotherm(0.0), 1),
        "hght_5c": round(get_height_of_isotherm(-5.0), 1),
        "hght_10c": round(get_height_of_isotherm(-10.0), 1),
        "hght_15c": round(get_height_of_isotherm(-15.0), 1),
        "hght_20c": round(get_height_of_isotherm(-20.0), 1),
        "cloud_top": round(max([c["top"] for c in extent_decks], default=0.0) / 1000.0, 1),
        "cloud_thick": round(max([max(0.0, c["top"] - c["base"]) for c in extent_decks], default=0.0) / 1000.0, 1),
        "thick_layer": 1 if thick_layer_violated else 0,
        "thick_layer_ft": round(thickest_in_band_ft),
        **_layer_mean_flow(profile_layers, 300.0, 150.0, "av"),
        "_layers": profile_layers,
    }


def parse_time_series_bufkit(bufkit_text):
    hourly_data = {}
    blocks = bufkit_text.split("STID = ")

    for block in blocks:
        if not block.strip():
            continue
        time_match = re.search(r"TIME\s*=\s*(\d{6})/(\d{4})", block)
        if not time_match:
            continue

        date_part, time_part = time_match.groups()
        try:
            valid_hour_key = f"{int(date_part[4:6]):02d}/{int(time_part[0:2]):02d}"
        except (ValueError, IndexError):
            continue

        lines = block.splitlines()
        profile_layers = []
        pres_idx, tmpc_idx, dwpt_idx, sknt_idx, drct_idx = 0, 1, 3, 5, 4
        header_names = []
        in_profile = False

        for line in lines:
            cleaned = line.strip()
            if not cleaned:
                continue
            if "PRES" in cleaned or "TMPC" in cleaned or "SKNT" in cleaned:
                in_profile = True
                header_names.extend(cleaned.split())
                try:
                    if "PRES" in header_names: pres_idx = header_names.index("PRES")
                    if "TMPC" in header_names: tmpc_idx = header_names.index("TMPC")
                    if "DWPT" in header_names: dwpt_idx = header_names.index("DWPT")
                    if "SKNT" in header_names: sknt_idx = header_names.index("SKNT")
                    if "DRCT" in header_names: drct_idx = header_names.index("DRCT")
                except ValueError:
                    pass
                continue

            if in_profile:
                if "STID" in cleaned or "STNM" in cleaned:
                    break
                parts = cleaned.split()
                if len(parts) > max(pres_idx, tmpc_idx, dwpt_idx, sknt_idx, drct_idx):
                    try:
                        if not parts[0].replace(".", "", 1).replace("-", "", 1).isdigit():
                            continue
                        pres = float(parts[pres_idx])
                        tmpc = float(parts[tmpc_idx])
                        dwpt = float(parts[dwpt_idx])
                        sknt = float(parts[sknt_idx])
                        try:
                            drct = float(parts[drct_idx])
                        except (ValueError, IndexError):
                            drct = None
                        if 100.0 <= pres <= 1050.0:
                            # Meteorological wind vector components (u: east+, v: north+).
                            # "FROM" direction convention -> components point opposite the heading.
                            if drct is not None and 0.0 <= drct <= 360.0:
                                u_comp = -sknt * math.sin(math.radians(drct))
                                v_comp = -sknt * math.cos(math.radians(drct))
                            else:
                                u_comp, v_comp = None, None
                            profile_layers.append({
                                "pres": pres,
                                "hght": pressure_to_height_ft(pres),
                                "tmpc": tmpc,
                                "dwpt": dwpt,
                                "depr": tmpc - dwpt,
                                "sknt": sknt,
                                "drct": drct,
                                "u": u_comp,
                                "v": v_comp,
                            })
                    except (ValueError, IndexError):
                        continue

        if not profile_layers:
            continue
        profile_layers.sort(key=lambda x: x["pres"], reverse=True)

        result = compute_profile_variables(profile_layers)
        if result is not None:
            hourly_data[valid_hour_key] = result
    return hourly_data


def fetch_station_model(session, stn, model):
    """Pull one station-model BUFKIT profile from PSU, politely.

    Returns (stn, model, hourly_data). An empty dict means the fetch failed or the file
    wasn't posted; run_pipeline will try to carry the previous run's column forward rather
    than render a blank column.
    """
    download_id = "xmr" if stn == "kxmr" else stn
    model_prefix = "gfs3" if model == "gfs" else model
    # https, not http: PSU redirects anyway, and the redirect costs an extra round trip
    # against a server that is already rate-limiting us.
    url = (f"https://www.meteo.psu.edu/bufkit/data/{model.upper()}/latest/"
           f"{model_prefix}_{download_id}.buf")
    headers = {
        "User-Agent": BUFKIT_USER_AGENT,
        "Accept": "text/plain,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    last = "no attempt made"
    with _BUFKIT_GATE:
        # Stagger inside the gate so the two permitted slots don't fire simultaneously.
        time.sleep(random.uniform(*BUFKIT_STAGGER_S))
        for attempt in range(BUFKIT_ATTEMPTS):
            try:
                r = session.get(url, headers=headers,
                                timeout=(BUFKIT_CONNECT_TIMEOUT, BUFKIT_READ_TIMEOUT))
                if r.status_code == 200:
                    body = r.text
                    if body and "STID" in body:
                        data = parse_time_series_bufkit(body)
                        if data:
                            _record_cycle(stn, model, _bufkit_init_cycle(body))
                            return stn, model, data
                        last = f"200 OK ({len(body)} B) but no parseable profiles"
                    else:
                        # A throttle page or truncated body can still come back 200.
                        last = f"200 OK but body is not BUFKIT ({len(body or '')} B)"
                elif r.status_code == 404:
                    # Genuinely not posted (some models skip some sites). Don't burn retries.
                    logging.warning(f"BUFKIT {stn}/{model}: 404, file not posted.")
                    return stn, model, {}
                elif r.status_code in (403, 429, 500, 502, 503, 504):
                    # These are exactly the throttle codes worth backing off on — the old
                    # code `break`-ed here and never retried them.
                    last = f"HTTP {r.status_code} (throttled)"
                else:
                    last = f"HTTP {r.status_code}"
            except Exception as e:
                last = f"{type(e).__name__}"

            if attempt < BUFKIT_ATTEMPTS - 1:
                wait = BUFKIT_BACKOFF_BASE_S * (2 ** attempt) * random.uniform(0.6, 1.4)
                logging.info(f"BUFKIT {stn}/{model}: {last}; retry "
                             f"{attempt + 2}/{BUFKIT_ATTEMPTS} in {wait:.1f}s")
                time.sleep(wait)

    logging.error(f"BUFKIT {stn}/{model} failed after {BUFKIT_ATTEMPTS} attempts ({last}).")
    return stn, model, {}


def _row_is_future(row_key, now_utc):
    """True when a 'DD/HH' row key is at or after the current hour (same wrap-safe rule
    run_pipeline uses to trim the live BUFKIT rows)."""
    try:
        d, h = map(int, row_key.split("/"))
    except Exception:
        return True
    if d < now_utc.day and now_utc.day - d < 25:
        return False
    if d == now_utc.day and h < now_utc.hour:
        return False
    return True


def _prior_run_station_data():
    """Newest stored run's station block from history.json, as (data, timestamp)."""
    try:
        with open(HISTORY_FILE, "r") as f:
            payload = json.load(f)
        runs = payload.get("runs", []) if isinstance(payload, dict) else payload
        for r in (runs or []):
            d = (r or {}).get("data") or {}
            if d:
                return d, r.get("timestamp")
    except Exception:
        pass
    return {}, None


def carry_forward_missing(sounding_matrix, models_to_check=None):
    """When PSU throttles us out of an entire airport column, reuse the newest stored rows
    for that (station, model) instead of rendering a blank column. Only EMPTY columns are
    filled — a partial fetch is never overwritten — and only forecast hours still in the
    future are carried, so nothing rots into the past. Each carried profile is tagged with
    the run it came from, which the frontend surfaces as a stale marker.

    A carried BUFKIT column is a genuinely older forecast, not a nowcast: treat it as the
    last known good run, and note that RRFS/REFS/ECMWF in the same row are current."""
    prior, ts = _prior_run_station_data()
    if not prior:
        return 0
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    filled = 0
    for stn, mdls in sounding_matrix.items():
        for mdl in (models_to_check or list(mdls.keys())):
            if mdls.get(mdl):
                continue  # this column fetched fine
            old = ((prior.get(stn) or {}).get(mdl)) or {}
            carried = {}
            for rk, prof in old.items():
                if isinstance(prof, dict) and _row_is_future(rk, now_utc):
                    p = dict(prof)
                    p["stale"] = p.get("stale") or ts or "a previous run"
                    carried[rk] = p
            if carried:
                mdls[mdl] = carried
                filled += 1
                logging.warning(f"Carried forward {stn}/{mdl}: {len(carried)} future hours "
                                f"from {ts} (BUFKIT fetch returned nothing this run).")
    if filled:
        logging.warning(f"{filled} BUFKIT column(s) carried forward and flagged stale.")
    return filled


# ---------------------------------------------------------------------------
# Launch-pad soundings derived from raw isobaric GRIB2 (GFS / RAP / HRRR)
# ---------------------------------------------------------------------------

# Isobaric levels to request from the NOMADS GRIB filter, in hPa. GFS carries the
# full mandatory+standard set; RAP/HRRR carry 25 hPa spacing but we request the same
# nominal list and just use whatever comes back.
PAD_LEVELS_HPA = [1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600,
                  550, 500, 450, 400, 350, 300, 250, 200, 150, 100]


def _rh_to_dewpoint_c(temp_c, rh_pct):
    """Magnus-formula dewpoint (°C) from temperature (°C) and relative humidity (%)."""
    if rh_pct is None or rh_pct <= 0:
        return temp_c - 30.0  # very dry fallback
    rh = max(1.0, min(100.0, rh_pct))
    a, b = 17.625, 243.04
    gamma = math.log(rh / 100.0) + (a * temp_c) / (b + temp_c)
    return (b * gamma) / (a - gamma)


def _nomads_grib_url(model, date_str, cycle, f_hour_int):
    """Build a NOMADS GRIB-filter URL that subsets to isobaric T/RH/HGT/UGRD/VGRD +
    surface pressure over a small Cape Canaveral bounding box (keeps downloads tiny)."""
    lev_params = "".join(f"&lev_{lv}_mb=on" for lv in PAD_LEVELS_HPA)
    var_params = "&var_TMP=on&var_RH=on&var_HGT=on&var_UGRD=on&var_VGRD=on&var_PRES=on"
    region = "&subregion=&leftlon=-81.2&rightlon=-80.0&toplat=29.2&bottomlat=28.0"

    if model == "hrrr":
        # Use the pressure-level HRRR filter (filter_hrrr_2d.pl is SURFACE fields only and
        # cannot serve the wrfprs 3D isobaric file we need for a sounding).
        base = "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_sub.pl"
        f_name = f"hrrr.t{cycle}z.wrfprsf{f_hour_int:02d}.grib2"
        dir_part = f"&dir=%2Fhrrr.{date_str}%2Fconus"
    elif model == "rap":
        base = "https://nomads.ncep.noaa.gov/cgi-bin/filter_rap.pl"
        f_name = f"rap.t{cycle}z.awp130pgrbf{f_hour_int:02d}.grib2"
        dir_part = f"&dir=%2Frap.{date_str}"
    else:  # gfs
        base = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
        f_name = f"gfs.t{cycle}z.pgrb2.0p25.f{f_hour_int:03d}"
        dir_part = f"&dir=%2Fgfs.{date_str}%2F{cycle}%2Fatmos"

    return f"{base}?file={f_name}{lev_params}{var_params}{region}{dir_part}"


def _grib_levels_to_layers(levels):
    """Convert a {pressure_hPa: {field: value}} dict (fields decoded from raw isobaric GRIB2:
    t, rh, hgt, u, v) into the profile_layers schema consumed by compute_profile_variables().
    Shared by the launch-pad NOMADS path and the ECMWF Open Data path so the math is identical."""
    layers = []
    for pres, f in levels.items():
        if "t" not in f:
            continue
        tmpc = f["t"] - 273.15 if f["t"] > 100 else f["t"]  # K -> C guard
        rh = f.get("rh")
        dwpt = _rh_to_dewpoint_c(tmpc, rh)
        u = f.get("u")
        v = f.get("v")
        sknt = math.hypot(u, v) * 1.943844 if (u is not None and v is not None) else 0.0
        # Meteorological FROM-direction, recovered from the components. Every consumer of the
        # matrix reads u/v directly (LLWS, mean flow, anvil flow), so this stayed None for
        # years without anyone noticing — until the skew-T tried to draw wind barbs off a
        # GRIB column and every barb silently vanished.
        drct = (math.degrees(math.atan2(-u, -v)) % 360.0) if (u is not None and v is not None) else None
        # GRIB geopotential height (gpm) -> feet if present, else barometric fallback.
        hght_ft = f["hgt"] * 3.280839895 if "hgt" in f else pressure_to_height_ft(pres)
        layers.append({
            "pres": pres,
            "hght": hght_ft,
            "tmpc": tmpc,
            "dwpt": dwpt,
            "depr": tmpc - dwpt,
            "rh": rh,  # native GRIB RH (%), used directly for RH>=95% cloud detection
            "sknt": sknt,
            "drct": drct,
            "u": u * 1.943844 if u is not None else None,  # m/s -> kt
            "v": v * 1.943844 if v is not None else None,
        })
    return layers


def build_pad_profiles_from_grib(filepath, pad_coords, debug=False):
    """
    Extract a vertical column at each pad's nearest grid cell from a raw isobaric
    GRIB2 file and assemble profile_layers dicts (matching the BUFKIT schema) so the
    shared compute_profile_variables() can run on them.
    Returns {pad_id: profile_layers_list}. When debug=True, logs a summary of every
    distinct (shortName, typeOfLevel) seen and how many isobaric fields matched, so a
    first live run reveals exactly what NOMADS returned vs what the parser expects.
    """
    per_pad_levels = {pid: {} for pid in pad_coords}
    seen_short_types = {}   # (shortName, typeOfLevel) -> count      [debug]
    matched_counts = {"t": 0, "rh": 0, "hgt": 0, "u": 0, "v": 0}   # [debug]
    isobaric_levels_seen = set()                                    # [debug]
    total_msgs = 0                                                  # [debug]
    try:
        grbs = pygrib.open(filepath)
        # Cache lat/lon grid + nearest-cell index per pad from the first message.
        grid_lats, grid_lons = None, None
        pad_ij = {}

        for grb in grbs:
            total_msgs += 1
            try:
                level = grb.level
                short = getattr(grb, "shortName", "")
                type_lvl = getattr(grb, "typeOfLevel", "")
            except Exception:
                continue

            if debug:
                key = (short, type_lvl)
                seen_short_types[key] = seen_short_types.get(key, 0) + 1

            if type_lvl != "isobaricInhPa" or level not in PAD_LEVELS_HPA:
                continue

            if debug:
                isobaric_levels_seen.add(level)

            if grid_lats is None:
                grid_lats, grid_lons = grb.latlons()
                glons = np.where(grid_lons > 180, grid_lons - 360.0, grid_lons)
                for pid, c in pad_coords.items():
                    dist = (grid_lats - c["lat"]) ** 2 + (glons - c["lon"]) ** 2
                    pad_ij[pid] = np.unravel_index(np.argmin(dist), dist.shape)

            vals = grb.values
            field = None
            if short in ("t", "TMP"): field = "t"
            elif short in ("r", "RH"): field = "rh"
            elif short in ("gh", "HGT"): field = "hgt"
            elif short in ("u", "UGRD", "10u"): field = "u"
            elif short in ("v", "VGRD", "10v"): field = "v"
            if field is None:
                continue

            if debug:
                matched_counts[field] += 1

            for pid, (iy, ix) in pad_ij.items():
                per_pad_levels[pid].setdefault(level, {})[field] = float(vals[iy, ix])

        grbs.close()
    except Exception as e:
        logging.error(f"Pad GRIB parse failed for {filepath}: {e}")
        return {}

    if debug:
        logging.info(f"[PAD DEBUG] {os.path.basename(filepath)}: {total_msgs} total GRIB messages")
        logging.info(f"[PAD DEBUG]   distinct (shortName, typeOfLevel) seen: "
                     + ", ".join(f"{k[0]}/{k[1]}={v}" for k, v in sorted(seen_short_types.items())))
        logging.info(f"[PAD DEBUG]   isobaric levels matched (hPa): {sorted(isobaric_levels_seen, reverse=True)}")
        logging.info(f"[PAD DEBUG]   fields matched to parser: {matched_counts}")
        if sum(matched_counts.values()) == 0:
            logging.warning("[PAD DEBUG]   >>> ZERO fields matched. shortNames above don't match the "
                            "parser's expected set (t/r/gh/u/v). Update the field mapping to match.")

    pad_profiles = {}
    for pid, levels in per_pad_levels.items():
        layers = _grib_levels_to_layers(levels)
        if layers:
            pad_profiles[pid] = layers
    return pad_profiles


def fetch_pad_model(session, model, date_str, cycle, f_hour_int, row_key, debug=False):
    """Download one raw GRIB2 subset and build pad profiles/variables for a single
    model forecast hour. Returns (row_key, model, {pad_id: variables_dict}).
    When debug=True, logs the request URL, HTTP status, and downloaded byte size."""
    url = _nomads_grib_url(model, date_str, cycle, f_hour_int)
    local_path = os.path.join(CACHE_DIR, f"pad_{model}_{cycle}z_f{f_hour_int:03d}.grib2")
    out = {}
    try:
        with session.get(url, timeout=25, stream=True) as r:
            if debug:
                logging.info(f"[PAD DEBUG] {model.upper()} f{f_hour_int:03d} HTTP {r.status_code}")
                logging.info(f"[PAD DEBUG]   URL: {url}")
            if r.status_code != 200:
                if debug:
                    logging.warning(f"[PAD DEBUG]   >>> Non-200 status. Check the NOMADS filter path/"
                                    f"filename for {model.upper()}. First 300 chars of body:")
                    try:
                        logging.warning(f"[PAD DEBUG]   {r.text[:300]}")
                    except Exception:
                        pass
                return row_key, model, out
            with open(local_path, "wb") as fh:
                for chunk in r.iter_content(chunk_size=16384):
                    fh.write(chunk)

        if debug:
            sz = os.path.getsize(local_path) if os.path.exists(local_path) else 0
            logging.info(f"[PAD DEBUG]   downloaded {sz} bytes")

        pad_profiles = build_pad_profiles_from_grib(local_path, LAUNCH_PADS, debug=debug)
        for pid, layers in pad_profiles.items():
            result = compute_profile_variables(layers)
            if result is not None:
                out[pid] = result
        if debug:
            sample_pid = next(iter(pad_profiles), None)
            n_layers = len(pad_profiles[sample_pid]) if sample_pid else 0
            logging.info(f"[PAD DEBUG]   built {len(pad_profiles)} pad profiles, "
                         f"~{n_layers} levels each, {len(out)} produced variable sets")
    except Exception as e:
        logging.debug(f"Pad fetch break {model} f{f_hour_int:03d}: {e}")
        if debug:
            logging.warning(f"[PAD DEBUG]   >>> Exception during {model.upper()} f{f_hour_int:03d}: {e}")
    finally:
        if os.path.exists(local_path):
            try: os.remove(local_path)
            except Exception: pass
    return row_key, model, out


def determine_model_cycle(session, model):
    """Find the most recent available cycle for a given model on NOMADS by probing
    directory listings for the last few candidate cycles."""
    now = datetime.datetime.now(datetime.timezone.utc)
    if model == "gfs":
        cycle_hours, latency_h = [0, 6, 12, 18], 5
    else:  # rap, hrrr are hourly
        cycle_hours, latency_h = list(range(24)), 2

    for back in range(0, 30):
        cand = now - datetime.timedelta(hours=back)
        if cand.hour not in cycle_hours:
            continue
        if (now - cand).total_seconds() / 3600.0 < latency_h:
            continue
        date_str = cand.strftime("%Y%m%d")
        cycle = f"{cand.hour:02d}"
        # Probe one representative file
        probe_url = _nomads_grib_url(model, date_str, cycle, 1)
        try:
            resp = session.head(probe_url, timeout=8)
            if resp.status_code == 200:
                return date_str, cycle
            resp = session.get(probe_url, timeout=8, stream=True)
            if resp.status_code == 200:
                resp.close()
                return date_str, cycle
        except Exception:
            continue
    return None, None


def fetch_all_pad_soundings():
    """Build the pad sounding matrix {pad_id: {model: {row_key: variables}}} from NOMADS.
    GFS and RAP are pulled here via the NOMADS grib-filter; HRRR is intentionally skipped
    (its NOMADS filter probe was unreliable) and instead sourced from AWS in the RRFS pass."""
    pad_matrix = {pid: {m: {} for m in MODELS} for pid in LAUNCH_PADS}
    nomads_models = [m for m in BUFKIT_MODELS if m != "hrrr"]  # gfs, rap (HRRR + ECMWF fetched elsewhere)

    with requests.Session() as session:
        for model in nomads_models:
            date_str, cycle = determine_model_cycle(session, model)
            if not cycle:
                logging.warning(f"No available {model.upper()} cycle found for pad soundings.")
                continue
            cycle_init = datetime.datetime.strptime(f"{date_str}{cycle}", "%Y%m%d%H").replace(tzinfo=datetime.timezone.utc)

            # All three are requested hourly across the 48h window. GFS carries hourly
            # native output through f120 on NOMADS (3-hourly only kicks in after f120),
            # so within 48h we get a full hourly series that matches the BUFKIT airports
            # and avoids sparse every-third-row gaps in the merged table.
            max_fh = 48
            step = 1
            f_hours = list(range(step, max_fh + 1, step))

            logging.info(f"Fetching {model.upper()} pad columns: {date_str} {cycle}z, {len(f_hours)} hours")
            for _pid in LAUNCH_PADS:
                _record_cycle(_pid, model, f"{date_str}{cycle}")
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                futures = []
                for idx, fh in enumerate(f_hours):
                    valid_dt = cycle_init + datetime.timedelta(hours=fh)
                    row_key = f"{valid_dt.day:02d}/{valid_dt.hour:02d}"
                    # Emit verbose diagnostics only on the first forecast hour of each
                    # model so the log shows exactly what NOMADS returned without spam.
                    dbg = (idx == 0)
                    futures.append(executor.submit(
                        fetch_pad_model, session, model, date_str, cycle, fh, row_key, dbg
                    ))
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        row_key, mdl, pad_vals = fut.result()
                        for pid, vars_dict in pad_vals.items():
                            pad_matrix[pid][mdl][row_key] = vars_dict
                    except Exception:
                        pass

            # Per-model summary: how many forecast hours produced usable pad data.
            sample_pad = next(iter(LAUNCH_PADS))
            hours_ok = len(pad_matrix[sample_pad].get(model, {}))
            if hours_ok == 0:
                logging.warning(f"[PAD DEBUG] {model.upper()} produced ZERO usable pad-hours — "
                                f"see the [PAD DEBUG] lines above for HTTP status / shortName mismatch.")
            else:
                logging.info(f"{model.upper()} pad soundings: {hours_ok}/{len(f_hours)} forecast hours produced data.")

    return pad_matrix


# ---------------------------------------------------------------------------
# RRFS (deterministic) + REFS (ensemble mean) pad columns via AWS Open Data
# ---------------------------------------------------------------------------

def _parse_grib_idx(idx_text):
    """Parse a GRIB2 .idx sidecar into a list of (msg_num, byte_start, shortName, level).
    Each idx line looks like: '1:0:d=2026070100:REFC:entire atmosphere:...'
    We only need the byte offsets so we can range-request specific messages."""
    entries = []
    lines = [ln for ln in idx_text.splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        parts = ln.split(":")
        if len(parts) < 5:
            continue
        try:
            msg_num = int(parts[0])
            byte_start = int(parts[1])
        except ValueError:
            continue
        short = parts[3].strip()
        level = parts[4].strip()
        # Byte end = start of next message - 1 (or EOF for the last message)
        byte_end = None
        if i + 1 < len(lines):
            nxt = lines[i + 1].split(":")
            try:
                byte_end = int(nxt[1]) - 1
            except (ValueError, IndexError):
                byte_end = None
        entries.append({"msg": msg_num, "start": byte_start, "end": byte_end,
                        "short": short, "level": level})
    return entries


def _range_download_grib(session, grib_url, idx_entries, wanted_levels_hpa, debug=False):
    """Given parsed idx entries, byte-range download only the isobaric TMP/RH/HGT/UGRD/VGRD
    messages at the wanted levels and concatenate them into a local temp GRIB2 file."""
    # Match idx level strings like "500 mb" and variable names.
    wanted_vars = ("TMP", "RH", "HGT", "UGRD", "VGRD")
    wanted_level_strs = {f"{lv} mb" for lv in wanted_levels_hpa}

    ranges = []
    for e in idx_entries:
        if e["short"] not in wanted_vars:
            continue
        if e["level"] not in wanted_level_strs:
            continue
        if e["end"] is None:
            ranges.append((e["start"], ""))  # open-ended to EOF
        else:
            ranges.append((e["start"], e["end"]))

    if not ranges:
        if debug:
            logging.warning("[RRFS DEBUG]   idx parsed but no matching isobaric TMP/RH/HGT/U/V "
                            "messages at wanted levels — check idx var/level naming.")
        return None

    # Collapse adjacent/near-adjacent ranges into single requests. One request per message
    # was ~105 per forecast hour, which NOMADS refuses outright after the first file.
    _n_before = len(ranges)
    _wanted_bytes = sum((e - s0) for s0, e in ranges if e not in (None, ""))
    _closed = sorted(((s0, e) for s0, e in ranges if e not in (None, "")), key=lambda x: x[0])
    _open = [(s0, e) for s0, e in ranges if e in (None, "")]
    _merged = []
    for s0, e in _closed:
        if _merged and (s0 - _merged[-1][1]) <= NOMADS_RANGE_MERGE_GAP:
            _merged[-1] = (_merged[-1][0], max(_merged[-1][1], e))
        else:
            _merged.append((s0, e))
    ranges = _merged + _open
    _span_bytes = sum((e - s0) for s0, e in _merged)
    if debug:
        logging.info(f"[RRFS DEBUG]   ranges merged {_n_before} -> {len(ranges)} requests; "
                     f"{_span_bytes/1e6:.0f} MB spanned vs {_wanted_bytes/1e6:.0f} MB wanted "
                     f"({_span_bytes/max(1,_wanted_bytes):.2f}x)")

    local_path = os.path.join(CACHE_DIR, f"rrfs_col_{abs(hash(grib_url)) % 10_000_000}.grib2")
    try:
        with open(local_path, "wb") as fh:
            # Group into a single multi-range request where possible; fall back to per-range.
            for start, end in ranges:
                hdr = {"Range": f"bytes={start}-{end}"}
                # Paced too: the range requests are the bulk of the traffic, so throttling
                # only the .idx probe while firing dozens of unpaced range GETs would miss
                # the thing NOMADS actually objects to.
                if "nomads.ncep.noaa.gov" in grib_url:
                    r = _nomads_get(session, grib_url, timeout=25, headers=hdr,
                                    tag="range GET", pause=NOMADS_RANGE_PAUSE_S)
                else:
                    r = session.get(grib_url, headers=hdr, timeout=25)
                if r.status_code in (200, 206):
                    fh.write(r.content)
        if os.path.getsize(local_path) == 0:
            os.remove(local_path)
            return None
        return local_path
    except Exception as e:
        if debug:
            logging.warning(f"[RRFS DEBUG]   range download failed: {e}")
        if os.path.exists(local_path):
            try: os.remove(local_path)
            except Exception: pass
        return None


def _rrfs_determine_cycle(session, model_kind):
    """Newest usable cycle on the NOMADS parallel feed, confirmed by probing .idx existence.

    model_kind: 'rrfs' (deterministic, hourly), 'refs' (ensemble mean, 6-hourly) or 'hrrr'.
    Returns (date_str, cycle).

    The old AWS version had to guess which of several REFS filenames actually carried
    isobaric temperature, because the prototype bucket kept renaming them. The NOMADS layout
    is documented and stable — refs.tHHz.mean.fFF.conus.grib2 under ensprod/ — so the probe
    is now a single existence check rather than a search.
    """
    now = datetime.datetime.now(datetime.timezone.utc)

    if model_kind == "hrrr":
        cycle_hours = list(range(24)) if HRRR_ALL_CYCLES else HRRR_EXTENDED_CYCLES
        latency_h = HRRR_LATENCY_H
    elif model_kind == "refs":
        cycle_hours, latency_h = REFS_CYCLE_HOURS, REFS_LATENCY_H
    else:
        cycle_hours, latency_h = RRFS_CYCLE_HOURS, RRFS_LATENCY_H

    for back in range(0, 36):
        cand = now - datetime.timedelta(hours=back)
        if cand.hour not in cycle_hours:
            continue
        if (now - cand).total_seconds() / 3600.0 < latency_h:
            continue
        date_str = cand.strftime("%Y%m%d")
        cycle = f"{cand.hour:02d}"

        # Probe a forecast hour that every run length reaches. f001 for RRFS/HRRR; REFS
        # ensemble products sometimes start at f01 but the mean is reliably there by f06.
        probe_fh = 6 if model_kind == "refs" else 1
        probe = _rrfs_grib_url(model_kind, date_str, cycle, probe_fh) + ".idx"
        try:
            r = session.get(probe, timeout=15)
            if r.status_code == 200 and len(r.text) > 50:
                if model_kind == "refs" and "TMP" not in r.text:
                    # A product without isobaric temperature is no use for a sounding; skip
                    # rather than silently building empty columns from it.
                    logging.warning(f"[RRFS] REFS {date_str} {cycle}z idx has no TMP; trying older cycle.")
                    continue
                return date_str, cycle
        except Exception:
            continue
    logging.warning(f"[RRFS] no usable {model_kind.upper()} cycle found on NOMADS after 36 h of probing.")
    return None, None


# Serialised, paced GET for NOMADS, with redirect-aware throttle handling.
#
# `allow_redirects=False` is deliberate: a 302 here is the throttle telling us to go away, and
# following it just fetches an HTML error page that would then fail to parse as an .idx —
# turning a clear signal into a confusing one. The Location is logged once so the assumption
# stays checkable rather than becoming folklore.
_NOMADS_LOCK = threading.Lock()
_NOMADS_LAST = [0.0]
_NOMADS_REDIRECT_LOGGED = [False]
_NOMADS_THROTTLE_STREAK = [0]


class NomadsThrottled(Exception):
    """Raised once NOMADS has refused enough requests in a row that continuing is pointless."""


def _nomads_reset_throttle():
    _NOMADS_THROTTLE_STREAK[0] = 0
    _NOMADS_REDIRECT_LOGGED[0] = False


def _nomads_get(session, url, timeout=20, tag="", stream=False, headers=None, pause=None):
    """One paced request. Retries a 302 with backoff before giving up.

    `pause` is the minimum spacing before this request. Callers fetching a NEW FILE pass the
    default (NOMADS_REQUEST_PAUSE_S); callers pulling successive byte-ranges out of a file
    they are already downloading pass NOMADS_RANGE_PAUSE_S.
    """
    pause = NOMADS_REQUEST_PAUSE_S if pause is None else pause
    for attempt in range(NOMADS_THROTTLE_RETRIES + 1):
        with _NOMADS_LOCK:
            gap = time.time() - _NOMADS_LAST[0]
            if gap < pause:
                time.sleep(pause - gap)
            _NOMADS_LAST[0] = time.time()
        r = session.get(url, timeout=timeout, allow_redirects=False,
                        stream=stream, headers=headers or {})
        if r.status_code not in (301, 302, 303, 307, 308, 429, 503):
            _NOMADS_THROTTLE_STREAK[0] = 0
            return r
        _NOMADS_THROTTLE_STREAK[0] += 1
        if _NOMADS_THROTTLE_STREAK[0] >= NOMADS_THROTTLE_GIVEUP:
            raise NomadsThrottled(
                f"{_NOMADS_THROTTLE_STREAK[0]} consecutive refusals from NOMADS "
                f"(last: HTTP {r.status_code} on {tag or url})")
        if not _NOMADS_REDIRECT_LOGGED[0]:
            _NOMADS_REDIRECT_LOGGED[0] = True
            logging.warning(f"[NOMADS] {tag or url} -> HTTP {r.status_code}; "
                            f"Location={r.headers.get('Location', '(none)')!r}. "
                            f"Treating as a throttle and backing off. If that Location looks "
                            f"like a real file move rather than an error page, the URL builder "
                            f"needs updating instead.")
        if attempt < NOMADS_THROTTLE_RETRIES:
            wait = NOMADS_REQUEST_PAUSE_S * (2 ** attempt) * random.uniform(0.8, 1.3)
            time.sleep(wait)
    return r


# Per-hour failure ledger for the RRFS/REFS sweep.
#
# The futures loop used to swallow every exception with a bare `except: pass`, so a run that
# produced 10 of 60 forecast hours gave no clue whether the other 50 were missing files, a
# throttle, or a parse error. Those need completely different fixes, and guessing between
# them wasted a cycle. Reasons are tallied here and summarised at the end of each sweep.
_RRFS_FAILS = {}


def _run_with_deadline(fn, seconds, label, default=None):
    """Run fn() but give up waiting after `seconds`.

    Some third-party clients retry forever on their own schedule — ecmwf-opendata defaults to
    500 attempts at 120 s apart, which is 16 hours, and a 429 from ECMWF's portal is enough to
    trigger it. There is no way to pass a deadline into those libraries, so the call runs on a
    daemon thread and the pipeline stops WAITING for it. The thread may still be sleeping when
    the process exits; that is fine, it is a daemon and holds nothing the run needs.

    The point is that no single upstream can decide how long the whole dashboard takes.
    """
    box = {}

    def _target():
        try:
            box["v"] = fn()
        except Exception as e:
            box["e"] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        logging.error(f"{label}: still running after {seconds}s — abandoning it and continuing "
                      f"so the rest of the run can finish and publish.")
        return default
    if "e" in box:
        logging.error(f"{label}: {type(box['e']).__name__}: {box['e']}")
        return default
    return box.get("v", default)


def _rrfs_note_fail(kind, fh, reason):
    _RRFS_FAILS.setdefault(kind, []).append((fh, reason))


def _rrfs_fail_summary(kind, attempted):
    """Log why the hours that produced nothing produced nothing."""
    fails = _RRFS_FAILS.get(kind) or []
    if not fails:
        return
    by_reason = {}
    for fh, reason in fails:
        # Collapse "idx HTTP 404" for f011..f060 into one line rather than fifty.
        key = reason.split("(")[0].strip()
        by_reason.setdefault(key, []).append(fh)
    logging.warning(f"{kind.upper()}: {len(fails)}/{attempted} forecast hours produced nothing. Reasons:")
    for reason, hrs in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
        hrs = sorted(hrs)
        span = f"f{hrs[0]:03d}" if len(hrs) == 1 else f"f{hrs[0]:03d}-f{hrs[-1]:03d}"
        logging.warning(f"    {len(hrs):>3} hour(s)  {span:<14} {reason}")
    # An unbroken tail of missing files means the run simply is not that long yet.
    idx404 = sorted(fh for fh, r in fails if r.startswith("idx HTTP 404"))
    if len(idx404) > 3 and idx404 == list(range(idx404[0], idx404[-1] + 1)):
        logging.warning(f"    -> f{idx404[0]:03d} onward are all missing on the server: this cycle "
                        f"has not been written that far yet, or does not run that long. "
                        f"Not a fetch problem.")


def _rrfs_grib_url(model_kind, date_str, cycle, f_hour_int):
    """Build the AWS S3 URL for an RRFS deterministic, REFS ensemble-mean, or HRRR
    pressure-level file. For REFS, uses the module-cached resolved filename pattern."""
    if model_kind == "refs":
        # refs/para/refs.YYYYMMDD/HH/ensprod/refs.tHHz.mean.fFF.conus.grib2
        # Note the extra ensprod/ level and the TWO-digit forecast hour; RRFS uses three.
        f_name = f"refs.t{cycle}z.mean.f{f_hour_int:02d}.conus.grib2"
        return f"{RRFS_NOMADS_ROOT}/refs/para/refs.{date_str}/{cycle}/ensprod/{f_name}"
    elif model_kind == "hrrr":
        f_name = f"hrrr.t{cycle}z.wrfprsf{f_hour_int:02d}.grib2"
        return f"{HRRR_AWS_ROOT}/hrrr.{date_str}/conus/{f_name}"
    else:
        # rrfs/para/rrfs.YYYYMMDD/HH/rrfs.tHHz.prslev.3km.fFFF.conus.grib2
        f_name = f"rrfs.t{cycle}z.prslev.3km.f{f_hour_int:03d}.conus.grib2"
        return f"{RRFS_NOMADS_ROOT}/rrfs/para/rrfs.{date_str}/{cycle}/{f_name}"


def fetch_rrfs_pad_hour(session, model_kind, date_str, cycle, f_hour_int, row_key, all_coords, debug=False):
    """Fetch one RRFS/REFS forecast hour from AWS via idx byte-range, extract columns at
    every site in all_coords (pads + airports), and compute variables.
    Returns (row_key, model_kind, {site_id: variables})."""
    grib_url = _rrfs_grib_url(model_kind, date_str, cycle, f_hour_int)
    idx_url = grib_url + ".idx"
    out = {}
    local_path = None
    try:
        idx_resp = _nomads_get(session, idx_url, timeout=15,
                               tag=f"{model_kind.upper()} f{f_hour_int:03d} idx")
        if debug:
            logging.info(f"[RRFS DEBUG] {model_kind.upper()} f{f_hour_int:03d} idx HTTP {idx_resp.status_code}")
            logging.info(f"[RRFS DEBUG]   idx URL: {idx_url}")
        if idx_resp.status_code != 200:
            if debug:
                logging.warning(f"[RRFS DEBUG]   >>> idx not found. Check {model_kind.upper()} "
                                f"path/filename. GRIB URL was: {grib_url}")
            _rrfs_note_fail(model_kind, f_hour_int, f"idx HTTP {idx_resp.status_code}")
            return row_key, model_kind, out

        idx_entries = _parse_grib_idx(idx_resp.text)
        if debug:
            uniq_vars = sorted({e["short"] for e in idx_entries})
            logging.info(f"[RRFS DEBUG]   idx has {len(idx_entries)} messages; distinct vars: {uniq_vars[:25]}")

        local_path = _range_download_grib(session, grib_url, idx_entries, PAD_LEVELS_HPA, debug=debug)
        if not local_path:
            _rrfs_note_fail(model_kind, f_hour_int, "range download returned nothing")
            return row_key, model_kind, out

        if debug:
            sz = os.path.getsize(local_path)
            logging.info(f"[RRFS DEBUG]   range-downloaded {sz} bytes of isobaric fields")

        site_profiles = build_pad_profiles_from_grib(local_path, all_coords, debug=debug)
        for sid, layers in site_profiles.items():
            result = compute_profile_variables(layers)
            if result is not None:
                out[sid] = result
        if not out:
            _rrfs_note_fail(model_kind, f_hour_int,
                            f"grib parsed but produced no usable profile "
                            f"({len(site_profiles)} site profiles)")
    except Exception as e:
        logging.debug(f"RRFS fetch break {model_kind} f{f_hour_int:03d}: {e}")
        _rrfs_note_fail(model_kind, f_hour_int, f"{type(e).__name__}: {str(e)[:80]}")
        if debug:
            logging.warning(f"[RRFS DEBUG]   >>> exception: {e}")
    finally:
        if local_path and os.path.exists(local_path):
            try: os.remove(local_path)
            except Exception: pass
    return row_key, model_kind, out


def fetch_all_rrfs_refs_soundings(include_hrrr=True):
    """Build {site_id: {'rrfs'|'refs'|'hrrr': {row_key: variables}}} from AWS, for BOTH the
    launch pads and the BUFKIT airport points (airports have no BUFKIT RRFS/REFS profiles).
    HRRR is pulled here too (via the same idx byte-range path) because the NOMADS grib-filter
    probe for HRRR was unreliable; its results replace the failed NOMADS HRRR pad column."""
    kinds = []
    if RRFS_ENABLED: kinds.append("rrfs")
    if REFS_ENABLED: kinds.append("refs")
    if include_hrrr: kinds.append("hrrr")

    # Combined site set: launch pads + the 5 airport stations, all point-extracted.
    all_coords = {}
    for pid, c in LAUNCH_PADS.items():
        all_coords[pid] = {"lat": c["lat"], "lon": c["lon"]}
    for sid, c in STN_COORDS.items():
        all_coords[sid] = {"lat": c["lat"], "lon": c["lon"]}

    if not kinds:
        return {sid: {} for sid in all_coords}

    matrix = {sid: {k: {} for k in kinds} for sid in all_coords}

    with requests.Session() as session:
        for kind in kinds:
            date_str, cycle = _rrfs_determine_cycle(session, kind)
            if not cycle:
                logging.warning(f"No available {kind.upper()} cycle found on AWS.")
                continue
            cycle_init = datetime.datetime.strptime(f"{date_str}{cycle}", "%Y%m%d%H").replace(tzinfo=datetime.timezone.utc)
            # RRFS/REFS/HRRR all provide hourly forecast output; request the full window and let
            # any missing hour 404 on its .idx probe (so exact availability is never hardcoded).
            # HRRR tops out at f48 even on extended cycles, so never chase f49-60 for it —
            # and on a non-extended cycle it stops at f18, so asking for 48 would fire 30
            # pointless .idx probes an hour. The depth follows the cycle actually chosen.
            if kind == "hrrr":
                kind_max_fh = 48 if int(cycle) in HRRR_EXTENDED_CYCLES else 18
            elif kind == "refs":
                kind_max_fh = REFS_MAX_FH
            else:
                # RRFS is hourly but only the synoptic runs go the full 60 h. Asking a 13Z run
                # for f060 would fire 40 doomed probes at NOMADS every hour.
                kind_max_fh = RRFS_MAX_FH if int(cycle) in RRFS_SYNOPTIC_CYCLES else RRFS_SHORT_FH
                if (RRFS_PROBE_BEYOND_SHORT and kind_max_fh < RRFS_MAX_FH
                        and RRFS_PROBE_FH > kind_max_fh):
                    # One .idx probe decides whether this off-hour cycle is genuinely short or
                    # whether the cap is just an assumption we inherited.
                    try:
                        _pu = _rrfs_grib_url(kind, date_str, cycle, RRFS_PROBE_FH) + ".idx"
                        _pr = _nomads_get(session, _pu, timeout=15,
                                          tag=f"{kind.upper()} f{RRFS_PROBE_FH:03d} depth probe")
                        if _pr.status_code == 200:
                            logging.info(f"{kind.upper()}: {cycle}z is an off-hour cycle (cap "
                                         f"f{kind_max_fh:03d}) but f{RRFS_PROBE_FH:03d} exists on "
                                         f"the server — extending to f{RRFS_MAX_FH:03d}.")
                            kind_max_fh = RRFS_MAX_FH
                        else:
                            logging.info(f"{kind.upper()}: {cycle}z stops at f{kind_max_fh:03d} "
                                         f"(f{RRFS_PROBE_FH:03d} probe returned HTTP "
                                         f"{_pr.status_code}). The short tail is the RUN LENGTH, "
                                         f"not a fetch failure.")
                    except Exception as _e:
                        logging.debug(f"{kind.upper()} depth probe failed, keeping cap: {_e}")
            f_hours = list(range(1, kind_max_fh + 1))
            if RRFS_PRIORITISE_PANEL_HOURS and kind in ("rrfs", "refs"):
                # Hours valid within ASSESS_HOUR_TOL of the assessment hour, on any forecast
                # day, moved to the front. Order only — nothing is added or dropped.
                def _is_panel_hour(fh):
                    v = cycle_init + datetime.timedelta(hours=fh)
                    return abs(v.hour - 10) <= ASSESS_HOUR_TOL
                prio = [h for h in f_hours if _is_panel_hour(h)]
                rest = [h for h in f_hours if not _is_panel_hour(h)]
                f_hours = prio + rest
                if prio:
                    logging.info(f"{kind.upper()}: fetching {len(prio)} panel-critical hour(s) "
                                 f"first (f{prio[0]:03d}..f{prio[-1]:03d}), then the rest.")

            logging.info(f"Fetching {kind.upper()} columns ({len(all_coords)} sites): {date_str} {cycle}z, {len(f_hours)} hours")
            for _sid in all_coords:
                _record_cycle(_sid, kind, f"{date_str}{cycle}")
            # Sequential, with a wall-clock budget. Concurrency is 1 anyway (NOMADS throttles
            # bursts), so the executor bought nothing but hid the running total — and without
            # a budget a slow source has no way to stop: the 2026-08-25 run sat for 36 minutes
            # and was cancelled with nothing written. Because the panel-critical hours are at
            # the FRONT of f_hours, running out of budget costs matrix depth, not a model.
            _nomads_reset_throttle()
            _budget = _nomads_kind_budget(kind)
            _t_start = time.time()
            _done = 0
            _reached = []          # forecast hours that actually produced data, for the horizon log
            for idx, fh in enumerate(f_hours):
                if time.time() - _t_start > _budget:
                    # Say WHICH hours are missing and why, because "budget spent" and "the run
                    # is only that long" produce identical-looking holes in the matrix and need
                    # completely different fixes.
                    _missed = sorted(set(f_hours[idx:]))
                    logging.warning(
                        f"{kind.upper()}: {_budget}s budget spent after {_done} of "
                        f"{len(f_hours)} hours ({(time.time()-_t_start)/max(1,_done):.1f}s/hour) "
                        f"— stopping here so the run can finish. Not fetched: "
                        f"f{_missed[0]:03d}-f{_missed[-1]:03d} ({len(_missed)} hours). These "
                        f"appear as BLANKS in the matrix and are a clock limit, not missing "
                        f"data upstream — raise NOMADS_KIND_BUDGET_OVERRIDE['{kind}'] "
                        f"(currently {_budget}) to close them.")
                    break
                valid_dt = cycle_init + datetime.timedelta(hours=fh)
                row_key = f"{valid_dt.day:02d}/{valid_dt.hour:02d}"
                dbg = (idx == 0)  # verbose only on the first hour
                try:
                    row_key, mk, site_vals = fetch_rrfs_pad_hour(
                        session, kind, date_str, cycle, fh, row_key, all_coords, dbg)
                    for sid, vd in site_vals.items():
                        matrix[sid][mk][row_key] = vd
                    if site_vals:
                        _done += 1
                        _reached.append(fh)
                        if _done % 10 == 0:
                            _el = time.time() - _t_start
                            logging.info(f"{kind.upper()}: {_done} hours in {_el:.0f}s "
                                         f"({_el/_done:.1f}s/hour), {_budget - _el:.0f}s of "
                                         f"budget left.")
                except NomadsThrottled as e:
                    # NOMADS has stopped serving us. Continuing just burns the budget and
                    # risks the job being cancelled before anything is committed, so stop
                    # this model and let the rest of the pipeline finish and publish.
                    logging.warning(
                        f"{kind.upper()}: abandoning after {_done} of {len(f_hours)} hours — {e}. "
                        f"The dashboard will publish without the missing {kind.upper()} hours; "
                        f"an incomplete column beats an uncommitted run.")
                    _rrfs_note_fail(kind, fh, "NOMADS throttled; model abandoned")
                    break
                except Exception as e:
                    _rrfs_note_fail(kind, fh, f"{type(e).__name__}: {str(e)[:60]}")

            # Contiguous horizon vs interior holes. A column that stops at f018 and a column
            # that has f001-f016 then islands at f026-f030 look the same in a summary count
            # and mean entirely different things at the matrix.
            if _reached:
                _r = sorted(set(_reached))
                _contig = _r[0]
                for _h in _r[1:]:
                    if _h == _contig + 1:
                        _contig = _h
                    else:
                        break
                _holes = [h for h in range(_r[0], _r[-1] + 1) if h not in set(_r)]
                _el = time.time() - _t_start
                logging.info(f"{kind.upper()}: unbroken through f{_contig:03d}, last hour "
                             f"f{_r[-1]:03d}, {len(_r)} hours in {_el:.0f}s "
                             f"({_el/len(_r):.1f}s/hour).")
                if _holes:
                    logging.warning(
                        f"{kind.upper()}: {len(_holes)} INTERIOR gap(s) between f{_contig+1:03d} "
                        f"and f{_r[-1]:03d} — these are hours the panel reservation jumped over "
                        f"before the budget ran out, so the matrix column will show blanks with "
                        f"values on both sides. Raise the budget or set "
                        f"RRFS_PRIORITISE_PANEL_HOURS=False to trade panel depth for a solid column.")

            _rrfs_fail_summary(kind, len(f_hours))
            sample = next(iter(all_coords))
            hours_ok = len(matrix[sample].get(kind, {}))
            if hours_ok == 0:
                logging.warning(f"[RRFS DEBUG] {kind.upper()} produced ZERO usable site-hours — "
                                f"see [RRFS DEBUG] lines above for the idx/URL mismatch.")
            else:
                logging.info(f"{kind.upper()} soundings: {hours_ok}/{len(f_hours)} forecast hours produced data.")

    return matrix


def fetch_all_ecmwf_soundings():
    """Add an ECMWF IFS (HRES 0.25°) column, point-extracted for every launch pad + airport,
    from the free ECMWF Open Data distribution (CC-BY-4.0). One multi-step GRIB2 file is
    pulled via the ecmwf-opendata client (byte-range subset of pressure-level t/gh/r/u/v),
    then grouped by valid time and run through the shared compute_profile_variables().

    Vertical resolution is coarse (12 tropospheric levels) vs the CAMs, so isotherm heights,
    PBL winds and shear are solid while the moisture-based LLCC fields (ceiling, cloud top,
    layer thickness) are advisory. Returns {site_id: {row_key: variables}} (empty on any
    failure — the column simply won't appear, the rest of the dashboard is unaffected)."""
    if not ECMWF_ENABLED:
        return {}
    try:
        from ecmwf.opendata import Client
    except Exception as e:
        logging.warning(f"ecmwf-opendata not installed; skipping ECMWF column ({e}).")
        return {}

    all_coords = {}
    for pid, c in LAUNCH_PADS.items():
        all_coords[pid] = {"lat": c["lat"], "lon": c["lon"]}
    for sid, c in STN_COORDS.items():
        all_coords[sid] = {"lat": c["lat"], "lon": c["lon"]}

    steps = list(range(0, ECMWF_MAX_FH + 1, 3))  # IFS open-data cadence is 3-hourly

    # Accumulators live OUTSIDE the fetch loop so a chunk that lands is kept even if a later
    # one never does. This is the whole point of the rewrite: the previous version made one
    # retrieve for all 49 steps inside a 300 s external deadline, so a slow portal or a 429
    # (ecmwf-opendata answers those with its own 120 s retry sleep, up to 500 times) produced
    # an EMPTY dict, and an empty dict is indistinguishable downstream from "no ECMWF".
    per = {}                                   # row_key -> sid -> {level: {field: val}}
    seen = {}                                  # (shortName, typeOfLevel) -> count  [debug]
    matched = {"t": 0, "rh": 0, "hgt": 0, "u": 0, "v": 0}
    decode_errors = 0
    grid_lats = grid_lons = None
    site_ij = {}
    init_dt = None
    chunks_ok = 0
    steps_ok = 0
    t_budget = time.time()

    def _parse_chunk(path):
        """Parse one chunk file into `per`. Returns messages matched."""
        nonlocal grid_lats, grid_lons, site_ij, decode_errors
        n_before = sum(matched.values())
        grbs = pygrib.open(path)
        try:
            for grb in grbs:
                try:
                    type_lvl = getattr(grb, "typeOfLevel", "")
                    short = getattr(grb, "shortName", "")
                except Exception:
                    continue
                seen[(short, type_lvl)] = seen.get((short, type_lvl), 0) + 1
                if type_lvl != "isobaricInhPa":
                    continue
                level = grb.level
                if level not in ECMWF_LEVELS_HPA:
                    continue
                field = None
                if short in ("t", "TMP"): field = "t"
                elif short in ("r", "RH"): field = "rh"
                elif short in ("gh", "HGT"): field = "hgt"
                elif short in ("u", "UGRD"): field = "u"
                elif short in ("v", "VGRD"): field = "v"
                if field is None:
                    continue
                try:
                    vd = grb.validDate  # datetime of the valid time
                except Exception:
                    continue
                row_key = f"{vd.day:02d}/{vd.hour:02d}"
                if grid_lats is None:
                    grid_lats, grid_lons = grb.latlons()
                    glons = np.where(grid_lons > 180, grid_lons - 360.0, grid_lons)
                    for sid, c in all_coords.items():
                        dist = (grid_lats - c["lat"]) ** 2 + (glons - c["lon"]) ** 2
                        site_ij[sid] = np.unravel_index(np.argmin(dist), dist.shape)
                try:
                    vals = grb.values
                except Exception as e:
                    # CCSDS decode failure surfaces here if eccodes lacks aec/libaec support.
                    decode_errors += 1
                    if decode_errors <= 3:
                        logging.error(f"ECMWF GRIB value decode failed ({short}@{level}): {e}")
                    continue
                matched[field] += 1
                for sid, (iy, ix) in site_ij.items():
                    per.setdefault(row_key, {}).setdefault(sid, {}).setdefault(level, {})[field] = float(vals[iy, ix])
        finally:
            grbs.close()
        return sum(matched.values()) - n_before

    # Chunked retrieve with source failover and a wall-clock budget.
    #
    # Sources are tried in order per chunk and the winner is remembered, so a portal that is
    # refusing today costs one failed chunk rather than the column. Every chunk is parsed and
    # deleted immediately: peak disk is one chunk, not the whole 49-step file.
    src_order = list(ECMWF_SOURCE_FALLBACKS)
    if ECMWF_SOURCE in src_order:
        src_order.remove(ECMWF_SOURCE)
    src_order.insert(0, ECMWF_SOURCE)
    good_src = None

    chunk_list = [steps[i:i + ECMWF_STEP_CHUNK] for i in range(0, len(steps), ECMWF_STEP_CHUNK)]
    logging.info(f"ECMWF IFS: {len(steps)} steps in {len(chunk_list)} chunk(s) of "
                 f"{ECMWF_STEP_CHUNK}, budget {ECMWF_BUDGET_S}s, sources {src_order}.")

    for ci, chunk in enumerate(chunk_list):
        elapsed = time.time() - t_budget
        if elapsed > ECMWF_BUDGET_S:
            logging.warning(
                f"ECMWF IFS: {ECMWF_BUDGET_S}s budget spent after {steps_ok} of {len(steps)} "
                f"steps — keeping what parsed and moving on. The column will be SHORTER, not "
                f"absent. Raise ECMWF_BUDGET_S (and ECMWF_DEADLINE_S with it) to go deeper.")
            break
        target = os.path.join(CACHE_DIR, f"ecmwf_ifs_pl_{ci:02d}.grib2")
        if os.path.exists(target):
            try: os.remove(target)
            except Exception: pass

        got = False
        for src in ([good_src] if good_src else src_order):
            t_chunk = time.time()
            try:
                client = Client(source=src)
                result = client.retrieve(
                    type="fc",
                    step=chunk,
                    levtype="pl",
                    levelist=ECMWF_LEVELS_HPA,
                    param=["t", "gh", "r", "u", "v"],
                    target=target,
                )
                if init_dt is None:
                    init_dt = getattr(result, "datetime", None)
                    if init_dt is not None:
                        _ec_cycle = init_dt.strftime("%Y%m%d%H")
                        for _sid in list(LAUNCH_PADS) + list(STATIONS):
                            _record_cycle(_sid, "ecmwf", _ec_cycle)
                n_msgs = _parse_chunk(target)
                size_kib = os.path.getsize(target) // 1024 if os.path.exists(target) else 0
                logging.info(f"ECMWF IFS chunk {ci + 1}/{len(chunk_list)} "
                             f"(f{chunk[0]:03d}-f{chunk[-1]:03d}) via {src}: {size_kib} KiB, "
                             f"{n_msgs} messages, {time.time() - t_chunk:.0f}s.")
                good_src = src
                chunks_ok += 1
                steps_ok += len(chunk)
                got = True
                break
            except Exception as e:
                logging.warning(f"ECMWF IFS chunk {ci + 1} (f{chunk[0]:03d}-f{chunk[-1]:03d}) "
                                f"via {src} failed after {time.time() - t_chunk:.0f}s: "
                                f"{type(e).__name__}: {e}")
                if good_src == src:
                    good_src = None   # stop pinning a source that just broke
            finally:
                if os.path.exists(target):
                    try: os.remove(target)
                    except Exception: pass
        if not got:
            logging.error(f"ECMWF IFS chunk {ci + 1}: every source refused. Continuing to the "
                          f"next chunk rather than abandoning the column.")

    logging.info("[ECMWF DEBUG] shortName/typeOfLevel seen: "
                 + ", ".join(f"{k[0]}/{k[1]}={v}" for k, v in sorted(seen.items())))
    logging.info(f"[ECMWF DEBUG] fields matched to parser: {matched}"
                 + (f" | {decode_errors} value-decode errors" if decode_errors else ""))
    logging.info(f"ECMWF IFS: {chunks_ok}/{len(chunk_list)} chunks, {steps_ok}/{len(steps)} "
                 f"steps, init {init_dt}, {time.time() - t_budget:.0f}s total.")
    if sum(matched.values()) == 0:
        logging.warning("[ECMWF DEBUG] ZERO isobaric fields matched. If shortNames above look right "
                        "but decode errors are nonzero, eccodes likely lacks CCSDS/aec (libaec) support. "
                        "If NO chunk succeeded at all, the failure is the retrieve, not the parse — "
                        "read the per-chunk warnings above for the reason each source gave.")
        return {}

    # Build profiles + run the shared variable computation (same engine as pads/BUFKIT).
    matrix = {sid: {} for sid in all_coords}
    for row_key, sites in per.items():
        for sid, levels in sites.items():
            layers = _grib_levels_to_layers(levels)
            vars_dict = compute_profile_variables(layers) if layers else None
            if vars_dict:
                # ECMWF's coarse 12-level grid + upper-level humidity reported relative to ICE
                # make the moisture-based cloud detection unreliable — it can false-flag ~46 kft
                # "cloud tops" from near-tropopause ice-saturation, and can't resolve low decks
                # (no levels between 1000 and 925 hPa). Blank those fields so the column shows
                # "-" instead of misleading values; isotherms, PBL winds and shear stay valid.
                for mk in ("ceiling", "cloud_top", "cloud_thick", "thick_layer", "thick_layer_ft"):
                    vars_dict[mk] = None
                matrix[sid][row_key] = vars_dict

    n = sum(len(v) for v in matrix.values())
    logging.info(f"ECMWF IFS soundings: {n} site-hours across {len(all_coords)} sites, {len(per)} valid times.")
    return matrix


def _anvil_eval(p, point, sectors):
    """Shared anvil test for one profile cell. Returns {'src','dir'} when the column looks like
    anvil debris — glaciated deck aloft, quiet point, and a convective core UPSTREAM along this
    column's own 300-150 mb anvil flow — else None. Used both for the real mask (on would-be thick-
    layer violations) and for the diagnostic pass, so the two can never drift apart."""
    ct = p.get("cloud_top")      # kft, highest cloud-deck top
    h20 = p.get("hght_20c")      # kft, -20C isotherm height
    av_reg = p.get("av_regime")  # compass of the anvil FROM-direction
    if ct is None or h20 is None or not av_reg:
        return None
    if ct < h20 + ANVIL_TOP_MARGIN_KFT:          # not glaciated -> not an anvil
        return None
    if point is not None and point >= CONVECTIVE_DBZ:  # it's the CB itself
        return None
    src = (sectors or {}).get(av_reg, 0.0)
    if src < ANVIL_SRC_DBZ:                      # no upstream source feeding it
        return None
    return {"src": src, "dir": av_reg}


def fetch_convective_reflectivity(time_keys):
    """Detect convective cores near each site from HRRR composite reflectivity (REFC), so the
    Thick Cloud Layer / Max Layer Thickness LLCC fields can be flagged as cu/anvil-governed
    (which have their own rules) rather than a stratiform bust. One tiny Cape-box REFC subset is
    pulled per forecast hour from the NOMADS HRRR 2-D grib filter; the neighborhood max within
    ~10 nm of each site is returned as {site_id: {row_key: dBZ}}. Empty on any failure (the mask
    simply won't apply — nothing else changes)."""
    if not CONVECTIVE_MASK_ENABLED:
        return {}

    all_coords = {}
    for pid, c in LAUNCH_PADS.items():
        all_coords[pid] = {"lat": c["lat"], "lon": c["lon"]}
    for sid, c in STN_COORDS.items():
        all_coords[sid] = {"lat": c["lat"], "lon": c["lon"]}

    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(pool_connections=30, pool_maxsize=30, max_retries=3))
    session.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"})
    now_utc = datetime.datetime.now(datetime.timezone.utc)

    def _s3_url(cc, date_str, fh):
        return f"{HRRR_AWS_ROOT}/hrrr.{date_str}/conus/hrrr.t{cc}z.wrfsfcf{fh:02d}.grib2"

    def _nomads_url(cc, date_str, fh):
        # Server-side subset. Kept only as a fallback: this CGI filter is the piece that has been
        # intermittently unavailable, taking the convective AND anvil masks down with it.
        return (f"https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl?file=hrrr.t{cc}z.wrfsfcf{fh:02d}.grib2"
                f"&var_REFC=on&lev_entire_atmosphere=on&subregion=&leftlon=-83.5&rightlon=-78.5"
                f"&toplat=31.0&bottomlat=25.0&dir=%2Fhrrr.{date_str}%2Fconus")

    def _s3_idx_ok(cc, date_str, fh):
        """The .idx is a few KB of text; fetching it both proves the cycle exists and gives us the
        byte offsets, so no separate HEAD probe is needed."""
        try:
            r = session.get(_s3_url(cc, date_str, fh) + ".idx", timeout=12)
            return r.status_code == 200 and "REFC" in r.text
        except Exception:
            return False

    def _probe(url):
        for attempt in range(3):
            try:
                r = session.head(url, timeout=12, allow_redirects=True)
                if r.status_code == 200: return True
                if r.status_code == 404: return False
            except Exception: pass
            time.sleep(1.0 * (attempt + 1))
        return False

    # HRRR runs hourly; find the most recent cycle with f01 posted (probe back up to 6 h).
    # Prefer AWS S3 (byte-range, no CGI dependency); fall back to the NOMADS filter if S3 is dry.
    active_date = active_cycle = None
    refc_source = None
    for back in range(0, 7):
        t = now_utc - datetime.timedelta(hours=back)
        d, cc = t.strftime("%Y%m%d"), t.strftime("%H")
        if _s3_idx_ok(cc, d, 1):
            active_date, active_cycle, refc_source = d, cc, "s3"
            break
    if not active_cycle:
        for back in range(0, 7):
            t = now_utc - datetime.timedelta(hours=back)
            d, cc = t.strftime("%Y%m%d"), t.strftime("%H")
            if _probe(_nomads_url(cc, d, 1)):
                active_date, active_cycle, refc_source = d, cc, "nomads"
                break
    if not active_cycle:
        logging.warning("Convective mask: no available HRRR REFC cycle found (tried AWS S3 and NOMADS).")
        return {}
    cycle_init = datetime.datetime.strptime(f"{active_date}{active_cycle}", "%Y%m%d%H").replace(tzinfo=datetime.timezone.utc)
    # HRRR forecast length: 48 h at the 00/06/12/18Z cycles, 18 h otherwise.
    max_fh = 48 if active_cycle in ("00", "06", "12", "18") else 18
    logging.info(f"Convective mask: HRRR REFC from {active_date} {active_cycle}z ({max_fh}-h) via {refc_source.upper()}.")

    ncells = max(1, round((CONVECTIVE_NBR_NM * 1.852) / 3.0))  # ~10 nm at HRRR 3-km spacing

    def _fetch_refc_file(fh, local):
        """Download just the REFC message to `local`. S3 path byte-ranges the single message out of
        the full wrfsfc file; NOMADS path streams the pre-subset file. Returns True on success."""
        if refc_source == "s3":
            grib_url = _s3_url(active_cycle, active_date, fh)
            r = session.get(grib_url + ".idx", timeout=20)
            if r.status_code != 200:
                return False
            ent = next((e for e in _parse_grib_idx(r.text)
                        if e["short"] == "REFC" and "entire atmosphere" in e["level"]), None)
            if not ent:
                return False
            rng = f"bytes={ent['start']}-{'' if ent['end'] is None else ent['end']}"
            rr = session.get(grib_url, headers={"Range": rng}, timeout=30)
            if rr.status_code not in (200, 206) or not rr.content:
                return False
            with open(local, "wb") as fhh:
                fhh.write(rr.content)
            return True
        with session.get(_nomads_url(active_cycle, active_date, fh), timeout=25, stream=True) as r:
            if r.status_code != 200:
                return False
            with open(local, "wb") as fhh:
                for chunk in r.iter_content(chunk_size=8192):
                    fhh.write(chunk)
        return True

    def _worker(fh, row_key):
        local = os.path.join(CACHE_DIR, f"refc_{active_cycle}z_f{fh:02d}.grib2")
        out = {}
        try:
            if not _fetch_refc_file(fh, local):
                return row_key, {}, None
            grbs = pygrib.open(local)
            refc = lats = lons = None
            for grb in grbs:
                short = getattr(grb, "shortName", "")
                nm = getattr(grb, "name", "") or ""
                if short in ("refc", "REFC") or "reflectivity" in nm.lower():
                    refc = _sanitize_grid(grb.values)
                    lats, lons = grb.latlons()
                    break
            grbs.close()
            if refc is None:
                return row_key, {}, None
            lons_n = np.where(lons > 180, lons - 360.0, lons)
            # S3 serves the full CONUS grid (no server-side subsetting), so crop to the Florida box
            # here — this keeps the per-site neighborhood and sector math on a small array.
            box = ((lats >= 25.0) & (lats <= 31.0) & (lons_n >= -83.5) & (lons_n <= -78.5))
            if np.any(box):
                ys, xs = np.where(box)
                y0b, y1b, x0b, x1b = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                refc = refc[y0b:y1b, x0b:x1b]
                lats = lats[y0b:y1b, x0b:x1b]
                lons_n = lons_n[y0b:y1b, x0b:x1b]
            sub_lons_out, sub_lats_out = lons_n, lats
            compass8 = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
            for sid, c in all_coords.items():
                dist = (lats - c["lat"]) ** 2 + (lons_n - c["lon"]) ** 2
                iy, ix = np.unravel_index(np.argmin(dist), dist.shape)
                y0, y1 = max(0, iy - ncells), min(refc.shape[0], iy + ncells + 1)
                x0, x1 = max(0, ix - ncells), min(refc.shape[1], ix + ncells + 1)
                nb = refc[y0:y1, x0:x1]
                entry = {"nbr": round(float(np.max(nb)) if nb.size else float(refc[iy, ix]), 1),
                         "point": round(float(refc[iy, ix]), 1)}
                if ANVIL_MASK_ENABLED:
                    # bearing (0=N,90=E) and great-circle-ish distance (nm) from the site to every
                    # cell; keep the max REFC per compass sector within the anvil advection annulus.
                    dlat_nm = (lats - c["lat"]) * 60.0
                    dlon_nm = (lons_n - c["lon"]) * 60.0 * math.cos(math.radians(c["lat"]))
                    dist_nm = np.sqrt(dlat_nm ** 2 + dlon_nm ** 2)
                    bearing = np.degrees(np.arctan2(dlon_nm, dlat_nm)) % 360.0
                    ring = (dist_nm >= ANVIL_NEAR_NM) & (dist_nm <= ANVIL_ADVECT_NM)
                    sectors = {}
                    for k, name in enumerate(compass8):
                        lo, hi = (k * 45 - 22.5) % 360.0, (k * 45 + 22.5) % 360.0
                        sect = ((bearing >= lo) & (bearing < hi)) if lo < hi else ((bearing >= lo) | (bearing < hi))
                        m = ring & sect
                        sectors[name] = round(float(np.max(refc[m])), 1) if np.any(m) else 0.0
                    entry["sectors"] = sectors
                out[sid] = entry
            # Hand the cropped grid back so REFC maps can be rendered AFTER the pool (matplotlib is
            # not thread-safe). Small: the Florida crop is ~1% of the CONUS grid.
            grid = (sub_lons_out, sub_lats_out, refc) if ANVIL_MASK_ENABLED else None
            return row_key, out, grid
        except Exception as e:
            logging.debug(f"REFC f{fh:02d} failed: {e}")
            return row_key, {}, None
        finally:
            if os.path.exists(local):
                try: os.remove(local)
                except Exception: pass

    matrix = {sid: {} for sid in all_coords}
    grids = {}
    tasks = []
    for fh in range(1, max_fh + 1):
        valid = cycle_init + datetime.timedelta(hours=fh)
        tasks.append((fh, f"{valid.day:02d}/{valid.hour:02d}"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(_worker, fh, rk) for fh, rk in tasks]
        for fut in concurrent.futures.as_completed(futs):
            try:
                rk, site_vals, grid = fut.result()
                for sid, v in site_vals.items():
                    matrix[sid][rk] = v
                if grid is not None:
                    grids[rk] = grid
            except Exception:
                pass

    total = sum(len(v) for v in matrix.values())
    hits = sum(1 for s in matrix.values() for cell in s.values()
               if isinstance(cell, dict) and cell.get("nbr") is not None and cell["nbr"] >= CONVECTIVE_DBZ)
    logging.info(f"Convective mask: REFC for {total} site-hours; {hits} exceed {CONVECTIVE_DBZ:.0f} dBZ "
                 f"(neighborhood {CONVECTIVE_NBR_NM:.0f} nm / {ncells} cells)"
                 f"{'; anvil sectors on' if ANVIL_MASK_ENABLED else ''}.")

    # Render the reflectivity maps single-threaded, after the pool (matplotlib is not thread-safe).
    # These back the ANVIL hover popup: the parent core shows upstream of the tagged site.
    refc_maps = {}
    if ANVIL_MASK_ENABLED and grids:
        for rk in sorted(grids.keys()):
            lo, la, va = grids[rk]
            p = _render_refc_map(rk, lo, la, va, active_cycle)
            if p:
                refc_maps[rk] = p
        logging.info(f"REFC maps: rendered {len(refc_maps)} NWS-scale reflectivity maps to maps/refc/.")
    return matrix, refc_maps


def _interp_exceedance(curve, height_m):
    """curve: {threshold_m: prob_pct}. Linear interpolation of the echo-top exceedance
    probability at height_m. Below the lowest threshold only a floor is known (the product is
    blind below ~20 kft), so we return that with a 'floor' quality flag."""
    if not curve:
        return None, "no_data"
    ths = sorted(curve.keys())
    if height_m <= ths[0]:
        return curve[ths[0]], "floor"
    if height_m >= ths[-1]:
        return curve[ths[-1]], "cap"
    for i in range(len(ths) - 1):
        lo, hi = ths[i], ths[i + 1]
        if lo <= height_m <= hi:
            f = (height_m - lo) / (hi - lo)
            return round(curve[lo] + f * (curve[hi] - curve[lo]), 1), "interp"
    return curve[ths[-1]], "cap"


def _ring_reduce(arr, lats, lons_n, clat, clon, iy, ix, radius_nm, method, half=18):
    """Collapse a probability grid over all cells within radius_nm of (clat, clon) — a TRUE
    circular ring, not a square box. This is the literal LCC "within X nm" operation: for the
    default 'max' it answers "does ANY cell within the standoff radius carry a high echo-top
    probability." A cropped (2*half+1) window around the nearest cell keeps it cheap, and the
    ring is defined by actual great-circle-ish distance so it does NOT assume a 3 km grid — if
    the enspost prob grid is a different spacing, the radius stays honest.
    method: 'max' (default LCC operation) | 'point' | 'mean' | 'p90'/'p75' | ...
    Falls back to the point value if no cell lands inside the radius."""
    y0, y1 = max(0, iy - half), min(arr.shape[0], iy + half + 1)
    x0, x1 = max(0, ix - half), min(arr.shape[1], ix + half + 1)
    sub = arr[y0:y1, x0:x1]
    slat = lats[y0:y1, x0:x1]
    slon = lons_n[y0:y1, x0:x1]
    # equirectangular km distance — accurate to well under a cell at 5-10 nm scales
    dx = (slon - clon) * 111.320 * math.cos(math.radians(clat))
    dy = (slat - clat) * 110.574
    dkm = np.sqrt(dx * dx + dy * dy)
    ring = dkm <= (radius_nm * 1.852)
    if not ring.any():
        return round(float(arr[iy, ix]), 1)
    vals = sub[ring]
    if method == "point":
        return round(float(arr[iy, ix]), 1)
    if method == "mean":
        return round(float(np.mean(vals)), 1)
    if method == "max":
        return round(float(np.max(vals)), 1)
    if method.startswith("p"):
        try:
            return round(float(np.percentile(vals, float(method[1:]))), 1)
        except (ValueError, IndexError):
            pass
    return round(float(np.max(vals)), 1)


def probe_rrfsens_member_retop(session):
    """Confirm the individual RRFS ensemble members carry RETOP (echo top) and in WHICH file, so
    we can build a true 5/10 nm member NMEP. Members live under rrfs_a/rrfsens.DATE/CC/mNNN/ — a
    DIFFERENT prefix from the 'refs.' enspost products (which is why the old probe only ever saw
    avrg/eas/prob). Logs the member list and the exact RETOP idx line. One-shot diagnostic."""
    import re as _re

    def _retop_lines(text):
        ents = _parse_grib_idx(text)
        rl = [f"{e['short']}:{e['level']}" for e in ents
              if "RETOP" in e['short'].upper() or "ETOP" in e['short'].upper()]
        return ents, rl

    now = datetime.datetime.now(datetime.timezone.utc)
    for back in range(0, 48):
        cand = now - datetime.timedelta(hours=back)
        if cand.hour not in (0, 6, 12, 18):
            continue
        d, cc = cand.strftime("%Y%m%d"), f"{cand.hour:02d}"
        base = f"rrfs_a/rrfsens.{d}/{cc}"
        # Require the full window (f024) of m001's 2dfld to be posted before accepting the cycle.
        head = f"{RRFS_AWS_ROOT}/{base}/m001/rrfs.t{cc}z.m001.2dfld.3km.f024.conus.grib2.idx"
        try:
            r = session.get(head, timeout=15)
        except Exception as e:
            logging.info(f"[RRFSENS PROBE] {d} {cc}z probe error: {e}")
            continue
        if r.status_code != 200 or len(r.text) < 50:
            continue

        # Enumerate member directories (m001, m002, ...).
        members = []
        try:
            lr = session.get(f"{RRFS_AWS_ROOT}/?list-type=2&prefix={base}/&delimiter=/&max-keys=300", timeout=20)
            members = sorted({mm.group(0) for mm in
                              (_re.search(r"m\d{3}", p) for p in _re.findall(r"<Prefix>([^<]+)</Prefix>", lr.text)) if mm})
        except Exception:
            pass

        # Locate RETOP: try 2dfld first, then subset as a fallback.
        ents, rl = _retop_lines(r.text)
        where = "2dfld"
        if not rl:
            try:
                sr = session.get(f"{RRFS_AWS_ROOT}/{base}/m001/rrfs.t{cc}z.m001.subset.3km.f024.conus.grib2.idx", timeout=15)
                if sr.status_code == 200 and len(sr.text) > 50:
                    ents2, rl2 = _retop_lines(sr.text)
                    if rl2:
                        where, rl, ents = "subset", rl2, ents2
            except Exception:
                pass

        shorts = sorted({e['short'] for e in ents}) if ents else []
        logging.info(f"[RRFSENS PROBE] cycle {d} {cc}z: members={members or ['m001?']} (n={len(members)})")
        logging.info(f"[RRFSENS PROBE] RETOP found in '{where}'? -> {rl[:2] if rl else 'NOT FOUND in 2dfld/subset'}; "
                     f"m001 {where} shorts sample={shorts[:24]}")

        # Confirm the prior cycle (6 h earlier) is posted deep enough for a time-lagged 10-member
        # ensemble: for our window f01..f24 the lagged cycle must reach f30 (= f24 + 6 h offset).
        prior = cand - datetime.timedelta(hours=6)
        dp, ccp = prior.strftime("%Y%m%d"), f"{prior.hour:02d}"
        lag_url = f"{RRFS_AWS_ROOT}/rrfs_a/rrfsens.{dp}/{ccp}/m001/rrfs.t{ccp}z.m001.2dfld.3km.f030.conus.grib2.idx"
        try:
            lr2 = session.get(lag_url, timeout=15)
            ok = (lr2.status_code == 200 and len(lr2.text) > 50)
        except Exception:
            ok = False
        logging.info(f"[RRFSENS PROBE] time-lag cycle {dp} {ccp}z f030 posted? -> {'YES' if ok else 'NO'} "
                     f"(need it for the +6 h lagged 5 members -> 10-member TLE).")
        return
    logging.info("[RRFSENS PROBE] no rrfsens cycle with m001 2dfld f024 found in the last 48h.")


def _render_refc_map(row_key, sub_lons, sub_lats, sub_vals, cycle):
    """Render one composite-reflectivity PNG over the Florida domain using the standard NWS radar
    reflectivity color scale, with the pads/stations marked. Used by the ANVIL hover popup so the
    parent convective core feeding the anvil can be seen upstream. Returns the path or None."""
    try:
        out_dir = os.path.join(MAPS_DIR, "refc")
        os.makedirs(out_dir, exist_ok=True)
        # Standard NWS reflectivity ramp: 5-75 dBZ in 5-dBZ steps.
        nws_colors = ["#04e9e7", "#019ff4", "#0300f4", "#02fd02", "#01c501", "#008e00",
                      "#fdf802", "#e5bc00", "#fd9500", "#fd0000", "#d40000", "#bc0000",
                      "#f800fd", "#9854c6", "#fdfdfd"]
        bounds = list(range(5, 85, 5))                      # 16 edges -> 15 colors
        cmap = mcolors.ListedColormap(nws_colors)
        cmap.set_under((0, 0, 0, 0))                        # <5 dBZ fully transparent
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        proj = ccrs.PlateCarree()
        counties = cfeature.NaturalEarthFeature(
            category="cultural", name="admin_2_counties", scale="10m", facecolor="none")
        states = cfeature.NaturalEarthFeature(
            category="cultural", name="admin_1_states_provinces_lines", scale="50m", facecolor="none")

        fig = plt.figure(figsize=(5.5, 5.8), dpi=120)
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        ax.set_extent([FL_DOMAIN["lon_min"], FL_DOMAIN["lon_max"],
                       FL_DOMAIN["lat_min"], FL_DOMAIN["lat_max"]], crs=proj)
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#0b1020", zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#141a24", zorder=0)
        ax.add_feature(counties, edgecolor="#334155", linewidth=0.3, zorder=1)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#94a3b8", linewidth=0.8, zorder=3)
        ax.add_feature(states, edgecolor="#94a3b8", linewidth=0.6, zorder=3)

        masked = np.ma.masked_less(sub_vals, 5.0)
        mesh = ax.pcolormesh(sub_lons, sub_lats, masked, cmap=cmap, norm=norm,
                             shading="auto", transform=proj, zorder=2)

        for pid, c in LAUNCH_PADS.items():
            ax.plot(c["lon"], c["lat"], marker="o", markersize=4, color="#38bdf8",
                    markeredgecolor="white", markeredgewidth=0.6, transform=proj, zorder=5)
        for sid, c in STN_COORDS.items():
            hot = sid.lower() == "kxmr"
            ax.plot(c["lon"], c["lat"], marker="^", markersize=7 if hot else 5,
                    color="#22c55e" if hot else "#e2e8f0", markeredgecolor="black",
                    markeredgewidth=0.7, transform=proj, zorder=6)
            ax.text(c["lon"] + 0.06, c["lat"] + 0.04, sid.upper(), fontsize=6, fontweight="bold",
                    color="#f8fafc", transform=proj, zorder=7,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="#0f172a", alpha=0.6, edgecolor="none"))

        ax.set_title(f"HRRR composite reflectivity — valid {row_key}Z (from {cycle}z)",
                     fontsize=8.5, fontweight="bold", color="#0f172a")
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03, ticks=bounds[::2], extend="min")
        cbar.set_label("dBZ", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

        safe_key = row_key.replace("/", "")
        out_name = f"refc_{cycle}z_{safe_key}z.png"
        out_path = os.path.join(out_dir, out_name)
        fig.savefig(out_path, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        return f"maps/refc/{out_name}"
    except Exception as e:
        logging.error(f"REFC map render failed for {row_key}: {e}")
        try:
            plt.close("all")
        except Exception:
            pass
        return None


def _render_refs_echotop_debug_map(row_key, sub_lons, sub_lats, sub_vals, cycle, thr_label):
    """Render one diagnostic PNG of the raw REFS P(echo top > thr) field over the Florida domain,
    styled like DESI (inferno on a dark base, low values transparent so the coastline shows) with
    the launch pads + stations marked. Writes to maps/refs_debug/. Returns the path or None.
    This is deliberately NOT added to history.json / the web page — it's for eyeball comparison."""
    try:
        out_dir = os.path.join(MAPS_DIR, "refs_debug")
        os.makedirs(out_dir, exist_ok=True)
        proj = ccrs.PlateCarree()
        counties = cfeature.NaturalEarthFeature(
            category="cultural", name="admin_2_counties", scale="10m", facecolor="none")
        states = cfeature.NaturalEarthFeature(
            category="cultural", name="admin_1_states_provinces_lines", scale="50m", facecolor="none")

        fig = plt.figure(figsize=(5.5, 5.8), dpi=120)
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        ax.set_extent([FL_DOMAIN["lon_min"], FL_DOMAIN["lon_max"],
                       FL_DOMAIN["lat_min"], FL_DOMAIN["lat_max"]], crs=proj)
        # Dark base so it reads like the DESI satellite backdrop; low prob stays dark.
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#0b1020", zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#141a24", zorder=0)
        ax.add_feature(counties, edgecolor="#334155", linewidth=0.3, zorder=1)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#94a3b8", linewidth=0.8, zorder=3)
        ax.add_feature(states, edgecolor="#94a3b8", linewidth=0.6, zorder=3)

        # Mask <=2% so the geography shows through in quiet areas (matches DESI's dark low end).
        masked = np.ma.masked_less_equal(sub_vals, 2.0)
        mesh = ax.pcolormesh(sub_lons, sub_lats, masked, cmap="inferno", vmin=0, vmax=100,
                             shading="auto", transform=proj, zorder=2)

        for pid, c in LAUNCH_PADS.items():
            ax.plot(c["lon"], c["lat"], marker="o", markersize=4, color="#38bdf8",
                    markeredgecolor="white", markeredgewidth=0.6, transform=proj, zorder=5)
        for sid, c in STN_COORDS.items():
            hot = sid.lower() == "kxmr"
            ax.plot(c["lon"], c["lat"], marker="^", markersize=7 if hot else 5,
                    color="#22c55e" if hot else "#e2e8f0", markeredgecolor="black",
                    markeredgewidth=0.7, transform=proj, zorder=6)
            ax.text(c["lon"] + 0.06, c["lat"] + 0.04, sid.upper(), fontsize=6, fontweight="bold",
                    color="#f8fafc", transform=proj, zorder=7,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="#0f172a", alpha=0.6, edgecolor="none"))

        ax.set_title(f"REFS P(echo top > {thr_label}) — valid {row_key}Z (from {cycle}z)",
                     fontsize=8.5, fontweight="bold", color="#0f172a")
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("Exceedance probability (%)", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

        safe_key = row_key.replace("/", "")
        out_name = f"refs_echotop_{thr_label.replace(' ', '')}_{cycle}z_{safe_key}z.png"
        out_path = os.path.join(out_dir, out_name)
        fig.savefig(out_path, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        return f"maps/refs_debug/{out_name}"
    except Exception as e:
        logging.error(f"REFS echo-top debug map render failed for {row_key}: {e}")
        try:
            plt.close("all")
        except Exception:
            pass
        return None


def _neighborhood_max(grid, nc):
    """Non-wrapping square neighborhood max (radius nc cells). Used ONLY for the diagnostic NMEP
    map; the table columns use the accurate circular ring via _ring_reduce."""
    ny, nx = grid.shape
    out = np.array(grid, dtype=float)
    for dy in range(-nc, nc + 1):
        for dx in range(-nc, nc + 1):
            if dy == 0 and dx == 0:
                continue
            ys0, ys1 = max(0, dy), ny + min(0, dy)
            yd0, yd1 = max(0, -dy), ny + min(0, -dy)
            xs0, xs1 = max(0, dx), nx + min(0, dx)
            xd0, xd1 = max(0, -dx), nx + min(0, -dx)
            if ys1 <= ys0 or xs1 <= xs0:
                continue
            np.maximum(out[yd0:yd1, xd0:xd1], grid[ys0:ys1, xs0:xs1], out=out[yd0:yd1, xd0:xd1])
    return out


def fetch_rrfsens_member_nmep():
    """TRUE 5/10 nm neighborhood-max ensemble probability (NMEP) of the Cumulus echo-top standoff,
    from the individual RRFS ensemble members (rrfs_a/rrfsens.DATE/CC/mNNN/) instead of the 40 km
    enspost 'prob' product. Optionally time-lags the prior cycle's members (+6 h, valid-time
    aligned) for a 10-member ensemble. Per member we take the MAX echo top inside the real circular
    ring; the caller counts the fraction whose in-ring max reaches the -10C / -20C height.
    Returns ({sid: {row_key: {"et5":[m...], "et10":[m...]}}}, {row_key: map_path}); empty on
    failure so the caller falls back to the enspost method."""
    if not REFS_MEMBER_NMEP_ENABLED:
        return {}, {}, {}

    all_coords = {}
    for pid, c in LAUNCH_PADS.items():
        all_coords[pid] = {"lat": c["lat"], "lon": c["lon"]}
    for sid, c in STN_COORDS.items():
        all_coords[sid] = {"lat": c["lat"], "lon": c["lon"]}

    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(pool_connections=30, pool_maxsize=30, max_retries=3))
    session.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"})

    ACCEPT_FH = 24                       # min depth to accept a cycle as meaningfully complete
    WINDOW_FH = REFS_MEMBER_WINDOW_FH     # target cap; actual depth discovered below
    LAG_HOURS = 6
    tmpl = "rrfs_a/rrfsens.{d}/{cc}/{mem}/rrfs.t{cc}z.{mem}.2dfld.3km.f{fh:03d}.conus.grib2"

    def _idx_ok(d, cc, mem, fh):
        u = f"{RRFS_AWS_ROOT}/{tmpl.format(d=d, cc=cc, mem=mem, fh=fh)}.idx"
        try:
            r = session.get(u, timeout=12)
            return r.status_code == 200 and len(r.text) > 50
        except Exception:
            return False

    def _max_fh(d, cc, target):
        """Deepest posted m001 2dfld forecast hour <= target (descending 6 h checkpoints, then
        refine upward by 1 h). Returns 0 if nothing at/under target is posted."""
        base = 0
        for f in range(target - (target % 6), 5, -6):
            if _idx_ok(d, cc, "m001", f):
                base = f
                break
        if base == 0:
            return 0
        f = base
        while f < target and _idx_ok(d, cc, "m001", f + 1):
            f += 1
        return f

    # --- newest cycle with m001 2dfld at least through ACCEPT_FH (freshest usable cycle) ---
    now = datetime.datetime.now(datetime.timezone.utc)
    date_str = cycle = None
    cyc_dt = None
    for back in range(0, 48):
        cand = now - datetime.timedelta(hours=back)
        if cand.hour not in (0, 6, 12, 18):
            continue
        d, cc = cand.strftime("%Y%m%d"), f"{cand.hour:02d}"
        if _idx_ok(d, cc, "m001", ACCEPT_FH):
            date_str, cycle, cyc_dt = d, cc, cand.replace(minute=0, second=0, microsecond=0)
            break
    if not cycle:
        logging.warning(f"REFS member NMEP: no rrfsens cycle with m001 2dfld f{ACCEPT_FH}; falling back to enspost.")
        return {}, {}, {}

    # actual depth of this cycle, up to the target cap
    WINDOW_FH = max(ACCEPT_FH, _max_fh(date_str, cycle, WINDOW_FH))

    # --- member discovery (m001.. until a gap) ---
    members = []
    for n in range(1, 21):
        mem = f"m{n:03d}"
        if _idx_ok(date_str, cycle, mem, 1):
            members.append(mem)
        elif members:
            break
    if not members:
        members = ["m001"]

    # --- time-lag: prior cycle members, bounded by the lag cycle's own posted depth ---
    lag_date = lag_cycle = None
    lag_members = []
    lag_fh_cap = 0   # deepest CURRENT-cycle fh the lag cycle can cover (fh + LAG <= lag depth)
    if REFS_MEMBER_TLE:
        prior = cyc_dt - datetime.timedelta(hours=LAG_HOURS)
        dp, ccp = prior.strftime("%Y%m%d"), f"{prior.hour:02d}"
        lag_depth = _max_fh(dp, ccp, WINDOW_FH + LAG_HOURS)
        if lag_depth >= 1 + LAG_HOURS:
            lag_date, lag_cycle = dp, ccp
            lag_fh_cap = min(WINDOW_FH, lag_depth - LAG_HOURS)
            for mem in members:
                if _idx_ok(dp, ccp, mem, 1 + LAG_HOURS):
                    lag_members.append(mem)
    n_total = len(members) + len(lag_members)
    logging.info("REFS member NMEP: cycle %s %sz (%d members, f1-f%d)%s -> %d-member%s." % (
        date_str, cycle, len(members), WINDOW_FH,
        (f" + lag {lag_date} {lag_cycle}z ({len(lag_members)}, covers f1-f{lag_fh_cap})"
         if lag_members else " (no time-lag)"),
        n_total,
        (f"; f{lag_fh_cap + 1}-f{WINDOW_FH} are {len(members)}-member" if lag_members and lag_fh_cap < WINDOW_FH else "")))

    r5_nm = REFS_CUMULUS_RADII_NM["neg10"]
    r10_nm = REFS_CUMULUS_RADII_NM["neg20"]

    # --- fetch tasks: (valid_row_key, label, d, cc, fh) ---
    tasks = []
    for fh in range(1, WINDOW_FH + 1):
        valid = cyc_dt + datetime.timedelta(hours=fh)
        rk = f"{valid.day:02d}/{valid.hour:02d}"
        for mem in members:
            tasks.append((rk, f"cur-{mem}", date_str, cycle, fh))
        if lag_members and fh <= lag_fh_cap:
            for mem in lag_members:
                tasks.append((rk, f"lag-{mem}", lag_date, lag_cycle, fh + LAG_HOURS))

    sanity = {"done": False}

    def _fetch_crop(d, cc, mem, fh):
        base = f"{RRFS_AWS_ROOT}/{tmpl.format(d=d, cc=cc, mem=mem, fh=fh)}"
        local = os.path.join(CACHE_DIR, f"rrfsens_{cc}z_{mem}_f{fh:03d}_{d}.grib2")
        try:
            r = session.get(base + ".idx", timeout=15)
            if r.status_code != 200:
                return None
            entries = _parse_grib_idx(r.text)
            cand = [e for e in entries
                    if "RETOP" in f"{e['short']} {e['level']}".upper() or "ETOP" in e['short'].upper()]
            if not cand:
                return None
            e = cand[0]
            hdr = {"Range": f"bytes={e['start']}-{e['end'] if e['end'] is not None else ''}"}
            rr = session.get(base, headers=hdr, timeout=30)
            if rr.status_code not in (200, 206) or not rr.content:
                return None
            with open(local, "wb") as fhh:
                fhh.write(rr.content)
            if os.path.getsize(local) == 0:
                return None
            grbs = pygrib.open(local)
            grb = None
            for g in grbs:
                grb = g
                break
            if grb is None:
                grbs.close()
                return None
            vals = _sanitize_grid(grb.values)
            lats, lons = grb.latlons()
            grbs.close()
            lons_n = np.where(lons > 180, lons - 360.0, lons)
            inbox = ((lats >= FL_DOMAIN["lat_min"]) & (lats <= FL_DOMAIN["lat_max"]) &
                     (lons_n >= FL_DOMAIN["lon_min"]) & (lons_n <= FL_DOMAIN["lon_max"]))
            if not inbox.any():
                return None
            yy, xx = np.where(inbox)
            y0, y1, x0, x1 = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
            return (np.array(vals[y0:y1, x0:x1]),
                    np.array(lats[y0:y1, x0:x1]),
                    np.array(lons_n[y0:y1, x0:x1]))
        except Exception as ex:
            logging.debug(f"member RETOP {mem} f{fh:03d} failed: {ex}")
            return None
        finally:
            if os.path.exists(local):
                try:
                    os.remove(local)
                except Exception:
                    pass

    by_rk = {}
    lock = threading.Lock()

    def _worker(rk, label, d, cc, fh):
        mem = label.split("-", 1)[1]
        crop = _fetch_crop(d, cc, mem, fh)
        if crop is None:
            return
        with lock:
            if not sanity["done"]:
                sanity["done"] = True
                try:
                    logging.info(f"[REFS MEMBER NMEP] sample {label} f{fh:03d} echo-top max="
                                 f"{float(np.nanmax(crop[0])):.0f} m (expect ~10000-18000 m if meters).")
                except Exception:
                    pass
            by_rk.setdefault(rk, {})[label] = crop

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(_worker, *t) for t in tasks]
        for _ in concurrent.futures.as_completed(futs):
            pass

    if not by_rk:
        logging.warning("REFS member NMEP: no member RETOP crops fetched; falling back to enspost.")
        return {}, {}, {}

    # --- aggregate: per site per row_key -> per-member in-ring max echo top (m) ---
    matrix = {sid: {} for sid in all_coords}
    debug_field5 = {}    # P(top>20kft) at ~5 nm  -> hovers on the -10C / 5 nm column
    debug_field10 = {}   # P(top>25kft) at ~10 nm -> hovers on the -20C / 10 nm column

    def _nmep_field(stk, ncell, thr_m):
        ex = np.zeros(stk.shape[1:], dtype=float)
        for mi in range(stk.shape[0]):
            ex += (_neighborhood_max(stk[mi], ncell) > thr_m)
        return 100.0 * ex / stk.shape[0]

    for rk, memmap in by_rk.items():
        if not memmap:
            continue
        ref_label = next(iter(memmap))
        _, ref_lats, ref_lons = memmap[ref_label]
        ref_shape = ref_lats.shape
        site_cell = {}
        for sid, c in all_coords.items():
            dist = (ref_lats - c["lat"]) ** 2 + (ref_lons - c["lon"]) ** 2
            iy, ix = np.unravel_index(np.argmin(dist), dist.shape)
            site_cell[sid] = (iy, ix)
        aligned = [(la, lo, vals) for (vals, la, lo) in memmap.values() if la.shape == ref_shape]
        for sid, c in all_coords.items():
            iy, ix = site_cell[sid]
            et5, et10 = [], []
            for (la, lo, vals) in aligned:
                et5.append(_ring_reduce(vals, la, lo, c["lat"], c["lon"], iy, ix, r5_nm, "max"))
                et10.append(_ring_reduce(vals, la, lo, c["lat"], c["lon"], iy, ix, r10_nm, "max"))
            matrix[sid][rk] = {"et5": et5, "et10": et10}
        stack = np.stack([vals for (la, lo, vals) in aligned]) if aligned else None
        if stack is not None and stack.shape[0]:
            # ~5 nm and ~10 nm square neighborhoods on the 3 km grid (9.3 km -> 3 cells,
            # 18.5 km -> 6 cells). Diagnostic only; the table columns use the exact circular ring.
            debug_field5[rk] = (ref_lons, ref_lats, _nmep_field(stack, 3, 6096.0))    # 20 kft
            debug_field10[rk] = (ref_lons, ref_lats, _nmep_field(stack, 6, 7620.0))   # 25 kft

    got = sum(len(v) for v in matrix.values())
    logging.info(f"REFS member NMEP: {got} site-hours from a {n_total}-member ensemble "
                 f"({len(by_rk)} valid hours fetched).")

    # --- diagnostic maps: two thresholds/radii, one per column ---
    #   maps5  = P(top>20 kft) at ~5 nm  -> -10C / 5 nm cell hover
    #   maps10 = P(top>25 kft) at ~10 nm -> -20C / 10 nm cell hover
    maps5, maps10 = {}, {}
    if REFS_CUMULUS_DEBUG_MAPS and (debug_field5 or debug_field10):
        dbg_dir = os.path.join(MAPS_DIR, "refs_debug")
        if os.path.isdir(dbg_dir):
            for old in os.listdir(dbg_dir):
                if old.endswith(".png"):
                    try:
                        os.remove(os.path.join(dbg_dir, old))
                    except Exception:
                        pass
        cyc_tag = f"{cycle}z-m{n_total}"
        for rk in sorted(debug_field5):
            lo, la, fld = debug_field5[rk]
            p = _render_refs_echotop_debug_map(rk, lo, la, fld, cyc_tag, "20kft")
            if p:
                maps5[rk] = p
        for rk in sorted(debug_field10):
            lo, la, fld = debug_field10[rk]
            p = _render_refs_echotop_debug_map(rk, lo, la, fld, cyc_tag, "25kft")
            if p:
                maps10[rk] = p
        logging.info(f"[REFS MEMBER NMEP] wrote {len(maps5)} x 20kft(~5nm) + {len(maps10)} x "
                     f"25kft(~10nm) member-NMEP debug maps to maps/refs_debug/.")

    return matrix, maps5, maps10


def fetch_refs_echotop_probs():
    """Pull REFS pre-computed echo-top exceedance probabilities P(echo top > {6096..15240} m)
    from the ensemble prob file. The RETOP messages ARE the ensemble member-fraction already;
    we collapse them over the LLCC standoff radii (5 nm / 10 nm) with a configurable reducer
    (REFS_CUMULUS_NBR_REDUCER) rather than a spatial MAX, which saturated to ~100% on any
    convective afternoon (the "values far too high" bug).
    Returns {site_id: {row_key: {"c5": {thr_m: pct}, "c10": {thr_m: pct}}}}. Empty on failure."""
    if not REFS_CUMULUS_ENABLED:
        return {}, {}
    all_coords = {}
    for pid, c in LAUNCH_PADS.items():
        all_coords[pid] = {"lat": c["lat"], "lon": c["lon"]}
    for sid, c in STN_COORDS.items():
        all_coords[sid] = {"lat": c["lat"], "lon": c["lon"]}

    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(pool_connections=30, pool_maxsize=30, max_retries=3))
    session.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"})

    # Discover the newest cycle whose PROB files are actually posted. The ensemble prob products
    # lag the ensemble mean, so we must NOT reuse the mean's cycle (that probed a not-yet-posted
    # 12z prob and every .idx 404'd silently). Probe the prob .idx directly, newest-first, and log
    # the outcome so a genuine "no .idx sidecar" case is distinguishable from "not posted yet".
    now = datetime.datetime.now(datetime.timezone.utc)
    date_str = cycle = None
    WINDOW_FH = 24  # end of the launch-planning window; also the tasks-loop depth below

    def _probe(d, cc, fh):
        purl = (f"{RRFS_NOMADS_ROOT}/refs/para/refs.{d}/{cc}/ensprod/"
                f"refs.t{cc}z.prob.f{fh:02d}.conus.grib2.idx")
        try:
            r = session.get(purl, timeout=12)
            return r.status_code == 200 and len(r.text) > 50
        except Exception as e:
            logging.debug(f"REFS cumulus probe {d} {cc}z f{fh:02d} error: {e}")
            return False

    # Pass 1: newest cycle that has FINISHED posting the full window (deep hour present).
    # Committing the moment an early hour (f04) appeared truncated the matrix to ~4 hours
    # when a fresh cycle was only partially posted -- a 2h-old 12z with just f01-f04 up would
    # win over a complete 06z. Requiring f24 trades ~6h of latency for a full 24-hour column.
    for back in range(0, 48):
        cand = now - datetime.timedelta(hours=back)
        if cand.hour not in (0, 6, 12, 18):
            continue
        d, cc = cand.strftime("%Y%m%d"), f"{cand.hour:02d}"
        if _probe(d, cc, WINDOW_FH):
            date_str, cycle = d, cc
            logging.info(f"REFS cumulus: prob files resolved at {d} {cc}z "
                         f"(full window; f{WINDOW_FH:02d} .idx present).")
            break

    # Pass 2 (fallback): nothing has the full window yet -> take the newest cycle with any
    # early hour so we still ship a (short) column rather than an empty one.
    if not cycle:
        for back in range(0, 48):
            cand = now - datetime.timedelta(hours=back)
            if cand.hour not in (0, 6, 12, 18):
                continue
            d, cc = cand.strftime("%Y%m%d"), f"{cand.hour:02d}"
            for probe_fh in (4, 6, 8, 12):
                if _probe(d, cc, probe_fh):
                    date_str, cycle = d, cc
                    logging.warning(f"REFS cumulus: no cycle with a full f{WINDOW_FH:02d} window "
                                    f"posted; falling back to {d} {cc}z (only f{probe_fh:02d} up, "
                                    f"column will be short this run).")
                    break
            if cycle:
                break

    if not cycle:
        logging.warning("REFS cumulus: no REFS PROB cycle with an .idx found on AWS "
                        "(prob files may not be posted yet, or lack .idx sidecars).")
        return {}, {}
    cycle_init = datetime.datetime.strptime(f"{date_str}{cycle}", "%Y%m%d%H").replace(tzinfo=datetime.timezone.utc)
    logging.info(f"REFS cumulus: echo-top probs from {date_str} {cycle}z "
                 f"(neighborhood reducer = '{REFS_CUMULUS_NBR_REDUCER}')")

    if REFS_MEMBER_PROBE:
        if RRFS_MEMBER_NMEP_ENABLED:
            probe_rrfsens_member_retop(session)

    r5_nm = REFS_CUMULUS_RADII_NM['neg10']    # 5 nm standoff ring for the -10C column
    r10_nm = REFS_CUMULUS_RADII_NM['neg20']   # 10 nm standoff ring for the -20C column
    logged_idx = {"done": False}

    # Spread accumulator for the reducer/verification diagnostic. For every site-hour we record
    # the pad point value and the in-ring max at the 20-kft threshold (5 nm ring), then summarize
    # once after the run. This lets the dashboard be cross-checked against the DESI REFS-CONUS
    # echo-top prob field: at a known valid time the in-ring max at the Cape should track what
    # DESI shows within ~5 nm. point vs max also shows how much the "within X nm" ring is adding
    # over the bare pad value.
    nbr_spread = []            # list of (sid, row_key, point, ring_p90, ring_max, ncells)
    debug_crops = []           # list of (row_key, sub_lons, sub_lats, sub_vals) for debug maps
    nbr_lock = threading.Lock()

    def _prob_url(fh):
        return (f"{RRFS_NOMADS_ROOT}/refs/para/refs.{date_str}/{cycle}/ensprod/"
                f"refs.t{cycle}z.prob.f{fh:02d}.conus.grib2")

    def _worker(fh, row_key):
        url = _prob_url(fh)
        local = os.path.join(CACHE_DIR, f"refs_prob_{cycle}z_f{fh:02d}.grib2")
        try:
            r = session.get(url + ".idx", timeout=15)
            if r.status_code != 200:
                return row_key, {}
            entries = _parse_grib_idx(r.text)
            # One-shot diagnostic: on the first hour that returns a populated idx, dump the field
            # inventory so we can see the actual echo-top abbreviation/format instead of guessing.
            if not logged_idx["done"]:
                logged_idx["done"] = True
                uniq = sorted({e["short"] for e in entries})
                etlines = [f"{e['short']}:{e['level']}" for e in entries
                           if "TOP" in e["short"].upper() or "ETOP" in e["short"].upper()
                           or "ECHO" in f"{e['short']} {e['level']}".upper()]
                logging.info(f"[REFS CUMULUS DEBUG] f{fh:02d} idx: {len(entries)} msgs; distinct shorts: {uniq[:40]}")
                logging.info(f"[REFS CUMULUS DEBUG] echo-top-ish idx lines: {etlines[:12] or 'NONE'}")
            # Match echo-top messages: wgrib2 abbrev (RETOP/ETOP/echo) in the short/level fields.
            cand = []
            for e in entries:
                blob = f"{e['short']} {e['level']}".upper()
                if "RETOP" in blob or "ETOP" in blob or "ECHO" in blob:
                    cand.append(e)
            if not cand:
                return row_key, {}
            with open(local, "wb") as fhh:
                for e in cand:
                    hdr = {"Range": f"bytes={e['start']}-{e['end'] if e['end'] is not None else ''}"}
                    rr = session.get(url, headers=hdr, timeout=25)
                    if rr.status_code in (200, 206):
                        fhh.write(rr.content)
            if os.path.getsize(local) == 0:
                return row_key, {}
            grbs = pygrib.open(local)
            raw_grids = {}
            lats = lons = None
            dbg_msgs = []
            for grb in grbs:
                # One-shot: dump the probability-message metadata. If these RETOP probs are a
                # neighborhood-max product (NMEP) with a baked-in radius, our point sample is
                # already spatially smeared and can read high where DESI's point field is nil.
                # PDT 4.9 is a plain probability template; a spatial/neighborhood product would
                # show up as PDT 4.15 or via a spatialProcessing / typeOfSpatialProcessing key.
                if not logged_idx.get("meta"):
                    logged_idx["meta"] = True
                    meta = {}
                    for k in ("productDefinitionTemplateNumber", "probabilityType",
                              "typeOfProcessedData", "numberOfForecastsInEnsemble",
                              "typeOfSpatialProcessing", "numberOfPointsUsed",
                              "scaledValueOfLowerLimit", "scaledValueOfUpperLimit",
                              "typeOfFirstFixedSurface", "localDefinitionNumber"):
                        try:
                            meta[k] = getattr(grb, k)
                        except Exception:
                            pass
                    logging.info(f"[REFS CUMULUS DEBUG] RETOP msg meta: {meta} "
                                 f"(PDT 4.15 or a typeOfSpatialProcessing/numberOfPointsUsed key "
                                 f"=> baked-in neighborhood; PDT 4.9 with none => point field)")
                # We byte-ranged ONLY the RETOP echo-top messages, so every message here is echo
                # top — pygrib reports RETOP's name/short as 'unknown' for this NCEP-local encoding,
                # so we key purely off the PDT-4.9 upperLimit (which decodes correctly).
                thr = None
                for attr in ("upperLimit", "scaledValueOfUpperLimit"):
                    try:
                        thr = float(getattr(grb, attr))
                        break
                    except (TypeError, ValueError, AttributeError):
                        continue
                if not logged_idx.get("parsed"):
                    dbg_msgs.append(f"upperLimit={thr}")
                if thr is None:
                    continue
                thr_snap = min(REFS_ECHOTOP_THRESHOLDS_M, key=lambda t: abs(t - thr))
                if abs(thr_snap - thr) > 250:  # not one of our thresholds
                    continue
                raw_grids[thr_snap] = _sanitize_grid(grb.values)
                if lats is None:
                    lats, lons = grb.latlons()
            grbs.close()
            # Decide fraction-vs-percent once across all thresholds (a genuinely low field would
            # trip a per-grid max<=1 test), then clamp to 0-100.
            gmax = max((g.max() for g in raw_grids.values() if g.size), default=0.0)
            scale = 100.0 if gmax <= 1.0 else 1.0
            grids = {th: np.clip(g * scale, 0.0, 100.0) for th, g in raw_grids.items()}
            if not logged_idx.get("parsed"):
                logged_idx["parsed"] = True
                logging.info(f"[REFS CUMULUS DEBUG] f{fh:02d} downloaded {len(cand)} cand msgs; "
                             f"parsed: {dbg_msgs[:8]}; matched thresholds: {sorted(grids.keys())}; "
                             f"raw max={gmax:.4g} -> scale x{scale:g}")
            if not grids or lats is None:
                return row_key, {}
            lons_n = np.where(lons > 180, lons - 360.0, lons)

            # Diagnostic: stash the Florida-domain crop of the 20-kft field for this forecast hour
            # so it can be rendered to maps/refs_debug/ after the pool (matplotlib isn't thread
            # safe, so we only COLLECT here and draw single-threaded later).
            if REFS_CUMULUS_DEBUG_MAPS and 6096 in grids:
                inbox = ((lats >= FL_DOMAIN["lat_min"]) & (lats <= FL_DOMAIN["lat_max"]) &
                         (lons_n >= FL_DOMAIN["lon_min"]) & (lons_n <= FL_DOMAIN["lon_max"]))
                if inbox.any():
                    yy, xx = np.where(inbox)
                    yy0, yy1, xx0, xx1 = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
                    with nbr_lock:
                        debug_crops.append((
                            row_key,
                            np.array(lons_n[yy0:yy1, xx0:xx1]),
                            np.array(lats[yy0:yy1, xx0:xx1]),
                            np.array(grids[6096][yy0:yy1, xx0:xx1]),
                        ))

            out = {}
            for sid, c in all_coords.items():
                dist = (lats - c["lat"]) ** 2 + (lons_n - c["lon"]) ** 2
                iy, ix = np.unravel_index(np.argmin(dist), dist.shape)

                # One-shot: confirm the nearest REFS cell for the Cape is actually on the Cape and
                # not snapped inland into the convective wash. If this is >3-4 km inland it alone
                # could explain elevated values vs DESI at the coastline.
                if sid.lower() == "kxmr" and not logged_idx.get("kxmr_cell"):
                    logged_idx["kxmr_cell"] = True
                    clat_cell, clon_cell = float(lats[iy, ix]), float(lons_n[iy, ix])
                    ddx = (clon_cell - c["lon"]) * 111.320 * math.cos(math.radians(c["lat"]))
                    ddy = (clat_cell - c["lat"]) * 110.574
                    dkm = (ddx * ddx + ddy * ddy) ** 0.5
                    logging.info(f"[REFS CUMULUS DEBUG] KXMR nearest cell grid[{iy},{ix}] = "
                                 f"({clat_cell:.3f},{clon_cell:.3f}) vs true "
                                 f"({c['lat']:.3f},{c['lon']:.3f}) -> {dkm:.1f} km off")

                # Record pad-point vs in-ring stats at the 20-kft threshold (5 nm ring) for this
                # site-hour, for the post-run summary and DESI cross-check. Uses the same true
                # circular ring as the actual columns, so the diagnostic and the shipped value
                # agree by construction.
                if 6096 in grids:
                    a = grids[6096]
                    pt = _ring_reduce(a, lats, lons_n, c["lat"], c["lon"], iy, ix, r5_nm, "point")
                    rp90 = _ring_reduce(a, lats, lons_n, c["lat"], c["lon"], iy, ix, r5_nm, "p90")
                    rmax = _ring_reduce(a, lats, lons_n, c["lat"], c["lon"], iy, ix, r5_nm, "max")
                    with nbr_lock:
                        nbr_spread.append((sid, row_key, pt, rp90, rmax, 0))

                c5, c10 = {}, {}
                for th, arr in grids.items():
                    c5[th] = _ring_reduce(arr, lats, lons_n, c["lat"], c["lon"], iy, ix,
                                          r5_nm, REFS_CUMULUS_NBR_REDUCER)
                    c10[th] = _ring_reduce(arr, lats, lons_n, c["lat"], c["lon"], iy, ix,
                                           r10_nm, REFS_CUMULUS_NBR_REDUCER)
                out[sid] = {"c5": c5, "c10": c10}
            return row_key, out
        except Exception as e:
            logging.debug(f"REFS prob f{fh:02d} failed: {e}")
            return row_key, {}
        finally:
            if os.path.exists(local):
                try: os.remove(local)
                except Exception: pass

    matrix = {sid: {} for sid in all_coords}
    tasks = []
    for fh in range(1, WINDOW_FH + 1):  # REFS runs to 60 h; the launch window here is ~24 h
        valid = cycle_init + datetime.timedelta(hours=fh)
        tasks.append((fh, f"{valid.day:02d}/{valid.hour:02d}"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(_worker, fh, rk) for fh, rk in tasks]
        for fut in concurrent.futures.as_completed(futs):
            try:
                rk, sv = fut.result()
                for sid, v in sv.items():
                    matrix[sid][rk] = v
            except Exception:
                pass
    got = sum(len(v) for v in matrix.values())
    logging.info(f"REFS cumulus: echo-top prob curves for {got} site-hours.")

    # Verification diagnostic. Summarize pad-point vs in-ring stats at 20 kft over the site-hours
    # that carry signal (ring max > 0), so the shipped 'max' column can be cross-checked against
    # DESI at a known valid time. Also surface the KXMR (Cape) ring max explicitly, since that's
    # the site being compared against the DESI echo-top prob field.
    active = [row for row in nbr_spread if row[4] > 0]
    if active:
        n = len(active)
        mp = sum(r[2] for r in active) / n
        m9 = sum(r[3] for r in active) / n
        mx = sum(r[4] for r in active) / n
        hottest = max(active, key=lambda r: r[4])   # single hottest in-ring max, with its hour
        logging.info(f"[REFS CUMULUS DEBUG] 20kft over {n}/{len(nbr_spread)} active site-hours "
                     f"(ring max>0): mean point={mp:.0f} p90={m9:.0f} ring-max={mx:.0f}; "
                     f"hottest {hottest[0]} {hottest[1]} point={hottest[2]:.0f} "
                     f"ring-max={hottest[4]:.0f} -> shipping reducer='{REFS_CUMULUS_NBR_REDUCER}'")
        cape = sorted([r for r in active if r[0].lower() == "kxmr"], key=lambda r: r[1])
        if cape:
            ringser = " ".join(f"{r[1]}={r[4]:.0f}" for r in cape)
            ptser = " ".join(f"{r[1]}={r[2]:.0f}" for r in cape)
            logging.info(f"[REFS CUMULUS DEBUG] KXMR 20kft ring-max by valid time: {ringser}")
            logging.info(f"[REFS CUMULUS DEBUG] KXMR 20kft POINT by valid time:   {ptser}")
            logging.info("[REFS CUMULUS DEBUG] ^ compare KXMR 11/21 above to the DESI F15 "
                         "(21Z) echo-top-prob pixel at the Cape; if ours >> DESI-at-Cape, the "
                         "grib is pre-neighborhooded (see RETOP msg meta) and needs a point source.")
    else:
        logging.info(f"[REFS CUMULUS DEBUG] no active (ring max>0) 20kft site-hours across "
                     f"{len(nbr_spread)} sampled; nothing convective to compare against DESI.")

    # Diagnostic render pass (single-threaded; matplotlib isn't thread safe). One PNG per
    # forecast hour of the raw P(top>20kft) field to maps/refs_debug/, now also surfaced as a
    # hover popup on the REFS cumulus cells (paths returned to the caller -> history.json).
    debug_map_paths = {}   # {row_key: "maps/refs_debug/....png"}
    if REFS_CUMULUS_DEBUG_MAPS and debug_crops:
        # Wipe last run's PNGs so this subfolder doesn't accumulate stale valid-times across
        # cycles (prune_stale_maps only sweeps the top-level maps/ dir, not this subfolder).
        dbg_dir = os.path.join(MAPS_DIR, "refs_debug")
        if os.path.isdir(dbg_dir):
            for old in os.listdir(dbg_dir):
                if old.endswith(".png"):
                    try:
                        os.remove(os.path.join(dbg_dir, old))
                    except Exception:
                        pass
        for row_key, slons, slats, svals in sorted(debug_crops, key=lambda r: r[0]):
            p = _render_refs_echotop_debug_map(row_key, slons, slats, svals, cycle, "20kft")
            if p:
                debug_map_paths[row_key] = p
        logging.info(f"[REFS CUMULUS DEBUG] wrote {len(debug_map_paths)}/{len(debug_crops)} echo-top "
                     f"debug maps to {os.path.join(MAPS_DIR, 'refs_debug')}/ (P(top>20kft), one per hour).")

    return matrix, debug_map_paths


# ---------------------------------------------------------------------------------------------
# Launch-thermo climatology (KXMR), assessed at 10Z. Both PWAT and Thompson are now full 15-point
# monthly distributions from the XMR period of record; _climo_percentile interpolates the value's
# rank within them. Percentile ranks (fractions -> 0-100):
CLIMO_PCTL_POINTS_15 = [0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100]

# PWAT: monthly percentile distribution (inches) for XMR / Cape Kennedy.
PWAT_PCTL_POINTS = CLIMO_PCTL_POINTS_15
PWAT_CLIMO_XMR = {
     1: [0.093, 0.171, 0.27, 0.338, 0.466, 0.577, 0.699, 0.808, 0.948, 1.083, 1.229, 1.432, 1.582, 1.757, 2.834],  # Jan
     2: [0.091, 0.19, 0.297, 0.375, 0.53, 0.676, 0.782, 0.907, 1.032, 1.154, 1.275, 1.401, 1.539, 1.726, 2.026],  # Feb
     3: [0.096, 0.217, 0.324, 0.421, 0.566, 0.676, 0.791, 0.889, 0.992, 1.118, 1.236, 1.412, 1.551, 1.779, 1.962],  # Mar
     4: [0.131, 0.252, 0.431, 0.518, 0.679, 0.779, 0.892, 1.003, 1.098, 1.238, 1.36, 1.552, 1.681, 1.852, 2.236],  # Apr
     5: [0.179, 0.422, 0.671, 0.809, 0.961, 1.097, 1.192, 1.293, 1.39, 1.502, 1.603, 1.774, 1.904, 2.143, 2.715],  # May
     6: [0.0, 0.768, 1.096, 1.287, 1.471, 1.585, 1.66, 1.748, 1.836, 1.922, 2.004, 2.13, 2.207, 2.435, 2.669],  # Jun
     7: [0.142, 1.016, 1.307, 1.444, 1.637, 1.716, 1.806, 1.895, 1.957, 2.025, 2.093, 2.191, 2.264, 2.447, 2.987],  # Jul
     8: [0.011, 0.751, 1.225, 1.456, 1.677, 1.787, 1.875, 1.943, 2.0, 2.067, 2.158, 2.249, 2.317, 2.491, 2.762],  # Aug
     9: [0.038, 0.743, 1.137, 1.257, 1.472, 1.635, 1.761, 1.844, 1.926, 2.022, 2.113, 2.218, 2.314, 2.478, 2.901],  # Sep
    10: [0.277, 0.399, 0.638, 0.773, 0.975, 1.11, 1.262, 1.403, 1.575, 1.729, 1.906, 2.075, 2.264, 2.475, 2.787],  # Oct
    11: [0.065, 0.325, 0.471, 0.577, 0.732, 0.861, 0.965, 1.093, 1.204, 1.337, 1.485, 1.68, 1.827, 2.075, 2.352],  # Nov
    12: [0.132, 0.217, 0.307, 0.394, 0.577, 0.739, 0.847, 0.967, 1.094, 1.211, 1.362, 1.523, 1.689, 1.917, 2.873],  # Dec
}

# Thompson Index (K − LI): monthly percentile distribution for XMR (coworker-supplied climatology).
THOMPSON_PCTL_POINTS = CLIMO_PCTL_POINTS_15
THOMPSON_CLIMO_XMR = {
     1: [-104.6, -74.3, -55.2, -48.2, -34.7, -25.6, -17.9, -9.0, -1.9, 5.0, 16.6, 25.3, 30.5, 36.8, 40.9],  # Jan
     2: [-88.3, -69.4, -53.5, -41.8, -27.6, -19.0, -11.4, -3.1, 3.9, 11.1, 19.9, 27.2, 32.0, 37.4, 42.2],  # Feb
     3: [-73.6, -61.5, -47.6, -37.1, -22.9, -15.8, -8.4, -2.0, 4.3, 10.8, 19.8, 29.1, 34.8, 39.2, 45.9],  # Mar
     4: [-60.8, -51.1, -34.9, -26.5, -15.9, -8.0, -2.0, 3.4, 10.7, 17.9, 25.6, 32.1, 36.2, 41.3, 45.1],  # Apr
     5: [-50.3, -31.4, -17.1, -8.8, 0.6, 7.1, 12.8, 18.7, 23.8, 28.4, 32.3, 35.9, 38.7, 42.4, 47.3],  # May
     6: [-27.3, -6.3, 7.6, 14.9, 23.0, 27.2, 30.3, 32.2, 33.8, 35.8, 37.5, 39.5, 41.4, 44.3, 48.6],  # Jun
     7: [-15.4, 6.2, 16.3, 20.5, 26.7, 29.6, 31.7, 33.2, 34.8, 36.1, 37.6, 39.7, 41.6, 44.6, 51.6],  # Jul
     8: [-15.5, 0.2, 14.0, 21.6, 27.6, 30.4, 32.2, 33.9, 35.0, 36.6, 38.4, 40.4, 41.7, 44.9, 51.2],  # Aug
     9: [-32.3, -13.8, 2.4, 9.2, 21.8, 27.2, 30.4, 32.4, 34.6, 36.2, 37.8, 39.6, 41.2, 44.1, 46.7],  # Sep
    10: [-68.6, -50.5, -29.3, -18.7, -7.9, 1.0, 9.4, 17.0, 23.4, 30.2, 34.3, 38.4, 40.0, 44.9, 54.2],  # Oct
    11: [-73.9, -55.5, -40.4, -31.5, -19.5, -11.3, -4.3, 0.9, 8.2, 16.4, 24.2, 31.3, 35.1, 41.3, 45.4],  # Nov
    12: [-87.2, -67.5, -52.9, -42.0, -28.4, -17.5, -10.3, -3.4, 3.2, 11.2, 18.8, 27.7, 31.4, 37.4, 45.1],  # Dec
}

# When a model has no sounding valid exactly at the assessment hour (10Z) on a given day — common
# for short-range RAP/HRRR depending on cycle timing — accept the nearest hour within this many
# hours instead of dropping the day. Exact 10Z always wins when present.
ASSESS_HOUR_TOL = 2

# How many panel snapshots to retain for its DPROG/DT. The cron is hourly but the models are not:
# GFS/ECMWF/RRFS/REFS cycle every 6 h, so consecutive hourly snapshots of those columns are
# IDENTICAL and stepping run-by-run tells you nothing. The frontend therefore steps by TIME
# (-6 h / -12 h for 6-hourly models, -1 h for hourly RAP/HRRR) and picks the nearest stored
# snapshot, so this needs ~19 h of depth (3 x 6 h back) plus margin for missed runs.
LAUNCH_THERMO_HISTORY_RUNS = 21

# 700-500 mb mean RH: monthly percentile distribution (%) for XMR.
RH75_PCTL_POINTS = CLIMO_PCTL_POINTS_15
RH75_CLIMO_XMR = {
     1: [0.9, 2.1, 5.5, 7.4, 8.3, 11.2, 15.8, 20.9, 25.7, 33.5, 46.1, 69.7, 80.4, 94.3, 99.6],  # Jan
     2: [1.1, 2.4, 6.1, 7.5, 9.2, 13.3, 18.3, 23.5, 29.4, 37.4, 49.9, 66.5, 81.4, 95.3, 98.9],  # Feb
     3: [1.6, 2.7, 6.0, 7.5, 9.1, 13.5, 17.3, 22.0, 27.5, 33.7, 43.7, 62.6, 76.0, 92.2, 98.8],  # Mar
     4: [1.8, 3.0, 6.0, 7.7, 10.1, 13.4, 17.3, 23.7, 28.8, 35.5, 44.5, 60.7, 73.3, 93.8, 99.7],  # Apr
     5: [1.8, 3.5, 7.8, 8.5, 13.2, 17.6, 23.7, 30.5, 35.7, 43.2, 52.9, 68.6, 79.8, 92.6, 97.0],  # May
     6: [4.1, 8.3, 13.8, 21.0, 29.8, 38.6, 45.9, 53.7, 60.5, 68.0, 75.3, 83.9, 89.7, 94.8, 97.4],  # Jun
     7: [8.8, 11.4, 19.1, 26.8, 35.9, 43.3, 50.9, 57.4, 63.3, 69.2, 76.3, 82.2, 87.4, 94.1, 97.6],  # Jul
     8: [3.7, 9.0, 19.3, 26.2, 37.5, 45.5, 51.6, 57.4, 63.4, 68.1, 73.6, 81.3, 86.3, 94.2, 100.0],  # Aug
     9: [3.9, 6.2, 10.8, 15.3, 24.5, 34.5, 43.3, 52.0, 59.9, 66.6, 73.8, 82.9, 88.1, 95.1, 99.6],  # Sep
    10: [1.7, 3.0, 6.3, 8.3, 10.8, 14.4, 19.6, 26.4, 33.3, 44.0, 57.2, 75.3, 87.7, 94.5, 99.1],  # Oct
    11: [1.4, 2.5, 5.7, 7.8, 8.9, 11.3, 15.0, 19.9, 24.9, 31.1, 41.4, 64.6, 76.5, 92.9, 99.1],  # Nov
    12: [1.2, 3.2, 5.4, 7.5, 8.7, 11.7, 15.2, 19.3, 25.7, 32.0, 43.0, 60.5, 77.6, 94.3, 98.6],  # Dec
}


def _climo_percentile(value, breaks, points):
    """Interpolated percentile rank of `value` within monthly breakpoint values at `points`."""
    if value is None or not breaks or not points or len(breaks) != len(points):
        return None
    if value <= breaks[0]:
        return points[0]
    if value >= breaks[-1]:
        return points[-1]
    for i in range(len(breaks) - 1):
        if breaks[i] <= value <= breaks[i + 1]:
            if breaks[i + 1] == breaks[i]:
                return points[i]
            f = (value - breaks[i]) / (breaks[i + 1] - breaks[i])
            return round(points[i] + f * (points[i + 1] - points[i]))
    return None


def _valid_day_fields(dd, now):
    """From a forecast day-of-month and 'now', reconstruct (weekday, 'Mon DD', sort key, month, year),
    wrapping into next month when the day has already passed this month."""
    month, year = now.month, now.year
    if dd < now.day - 5:
        month += 1
        if month > 12:
            month, year = 1, year + 1
    try:
        vdate = datetime.date(year, month, dd)
        return (vdate.strftime("%A"), vdate.strftime("%b %d"),
                f"{year:04d}{month:02d}{dd:02d}", month, year)
    except Exception:
        return (f"{dd:02d}", f"{dd:02d}", f"{dd:02d}", now.month, now.year)


def fetch_gefs_member_thermo(site="kxmr", assess_hour=10, cache=None):
    """GEFS ensemble launch-thermo for the 10Z panel: pull each member's isobaric sounding at the
    site for the forecast hours nearest the assessment hour, compute the indices PER MEMBER, and
    average the results via _ensemble_thermo_row.

    Returns (rows, cycle_key). `cache` may be a previous {"cycle":..., "rows":...}; if the newest
    posted cycle matches it the cached rows are returned untouched, which skips the whole fetch on
    the ~5 of every 6 hourly runs where GEFS has not advanced."""
    if not GEFS_ENABLED:
        return {}, None
    sc = STN_COORDS.get(site)
    if not sc:
        return {}, None
    coords = {site: {"lat": sc["lat"], "lon": sc["lon"]}}

    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=3))
    session.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"})

    members = ["c00"] + [f"p{i:02d}" for i in range(1, max(1, GEFS_MEMBERS))]

    # SOURCE: AWS S3, not the NOMADS grib_filter CGI. The filter would be far cheaper in bytes
    # (server-side cropping), but by the time this runs the pipeline has already made hundreds of
    # NOMADS requests for the GFS/RAP pad columns, HREF and HREFCT - so the filter answers every
    # GEFS probe with "302 Over Rate Limit" and the column silently vanishes. S3 has no such limit.
    #
    # The tradeoff is bandwidth: a 0.5-deg message is global (~200 KB), so byte-ranging pulls whole
    # fields to read one point. Trimmed to 4 vars x 11 levels and cached against the 6-hourly cycle
    # that is ~0.9 GB per cycle-change, i.e. ~150 MB/run amortized - roughly a tenth of what the
    # hourly ECMWF fetch already costs.
    def _url(d, cc, mem, fh, ab):
        sub = "pgrb2ap5" if ab == "a" else "pgrb2bp5"
        return (f"{GEFS_AWS_ROOT}/gefs.{d}/{cc}/atmos/{sub}/"
                f"ge{mem}.t{cc}z.pgrb2{ab}.0p50.f{fh:03d}")

    _lvl_re = re.compile(r"^(\d+)\s*mb$")

    def _wanted(entries):
        """Pick the TMP/RH/UGRD/VGRD messages at GEFS_LEVELS_HPA out of a parsed .idx."""
        want = []
        for e in entries:
            if e["short"] not in GEFS_VARS:
                continue
            m = _lvl_re.match((e.get("level") or "").strip())
            if m and int(m.group(1)) in GEFS_LEVELS_HPA:
                want.append(e)
        return want

    def _merge(entries, gap=4096):
        """Merge byte ranges that are adjacent or nearly so. GRIB messages for one variable sit
        contiguously in the file, so this collapses ~44 requests into a handful without pulling
        materially more data."""
        rngs = sorted(((e["start"], e["end"]) for e in entries), key=lambda x: x[0])
        out = []
        for s, e in rngs:
            if out and out[-1][1] is not None and s - out[-1][1] <= gap:
                out[-1] = (out[-1][0], e if e is not None else None)
            else:
                out.append((s, e))
        return out

    def _idx_ok(d, cc, mem, fh, ab="a"):
        """A cycle exists if its .idx is served (a few KB of text, no data transfer)."""
        try:
            r = session.get(_url(d, cc, mem, fh, ab) + ".idx", timeout=15)
            return r.status_code == 200 and "TMP" in r.text
        except Exception:
            return False

    def _deepest_fh_for(cyc):
        """The largest forecast hour this cycle needs to cover every 10Z day in range."""
        best = 0
        for fh in range(3, GEFS_MAX_FH + 1, 3):
            v = cyc + datetime.timedelta(hours=fh)
            if abs(v.hour - assess_hour) <= ASSESS_HOUR_TOL:
                best = fh
        return best

    # Newest COMPLETE GEFS cycle, probing back up to 24 h.
    #
    # The previous version probed f003 only. GEFS posts progressively over roughly an hour,
    # so f003 appears long before f168 — an hourly cron firing mid-post would accept the new
    # cycle, then find the deep forecast hours missing and silently produce a 2-3 day panel
    # instead of 7. Probing the DEEPEST hour the panel actually needs (plus a mid-range
    # member, since members also trickle) means an in-progress cycle is skipped and the
    # previous complete one is used until the new one has fully landed.
    now = datetime.datetime.now(datetime.timezone.utc)
    date_str = cycle = None
    cyc_dt = None
    skipped_partial = None
    for back in range(0, 25):
        cand = now - datetime.timedelta(hours=back)
        if cand.hour not in (0, 6, 12, 18):
            continue
        d, cc = cand.strftime("%Y%m%d"), f"{cand.hour:02d}"
        cand0 = cand.replace(minute=0, second=0, microsecond=0)
        if not _idx_ok(d, cc, "c00", 3):
            continue                      # cycle hasn't started posting at all
        deep = _deepest_fh_for(cand0)
        last_mem = members[-1]
        if deep and not (_idx_ok(d, cc, "c00", deep) and _idx_ok(d, cc, last_mem, deep)):
            # Started but not finished. Remember it so the log explains the fallback.
            if skipped_partial is None:
                skipped_partial = f"{d} {cc}z"
            continue
        date_str, cycle = d, cc
        cyc_dt = cand0
        break
    if not cycle:
        logging.warning("GEFS thermo: no COMPLETE cycle found on AWS after probing 24 h back; "
                        "GEFS omitted from panel.")
        return {}, None
    if skipped_partial:
        logging.info(f"GEFS thermo: cycle {skipped_partial} is still posting "
                     f"(deep forecast hours absent) — using {date_str} {cycle}z instead.")

    cycle_key = f"{date_str}{cycle}"
    if GEFS_CACHE_ENABLED and isinstance(cache, dict) and cache.get("cycle") == cycle_key and cache.get("rows"):
        logging.info(f"GEFS thermo: cycle {date_str} {cycle}z unchanged - reusing {len(cache['rows'])} cached rows.")
        return cache["rows"], cycle_key

    # GEFS is 3-hourly, so an exact 10Z valid time never exists off a 00/06/12/18Z cycle. Take the
    # step nearest the assessment hour on each forecast day, within ASSESS_HOUR_TOL.
    best_by_day = {}
    for fh in range(3, GEFS_MAX_FH + 1, 3):
        v = cyc_dt + datetime.timedelta(hours=fh)
        diff = abs(v.hour - assess_hour)
        if diff > ASSESS_HOUR_TOL:
            continue
        key = v.strftime("%Y%m%d")
        if key not in best_by_day or diff < best_by_day[key][0]:
            best_by_day[key] = (diff, fh, v)
    picks = sorted(best_by_day.values(), key=lambda x: x[1])
    if not picks:
        return {}, cycle_key

    def _grab(d, cc, mem, fh, ab, out_fh):
        """Byte-range the wanted messages out of one pgrb2a/b file into `out_fh`. Returns bytes
        written. GRIB2 files are concatenated messages, so a and b can share one local file."""
        try:
            ir = session.get(_url(d, cc, mem, fh, ab) + ".idx", timeout=20)
            if ir.status_code != 200:
                return 0
            want = _wanted(_parse_grib_idx(ir.text))
            if not want:
                return 0
            n = 0
            for s, e in _merge(want):
                rng = f"bytes={s}-{'' if e is None else e}"
                rr = session.get(_url(d, cc, mem, fh, ab), headers={"Range": rng}, timeout=60)
                if rr.status_code in (200, 206) and rr.content:
                    out_fh.write(rr.content)
                    n += len(rr.content)
            return n
        except Exception as exc:
            logging.debug(f"GEFS {mem} f{fh:03d} pgrb2{ab}: {exc}")
            return 0

    out = {}
    dropped_rows = []
    probed = False
    for (_diff, fh, valid) in picks:
        rk = f"{valid.day:02d}/{valid.hour:02d}"
        per = []
        for mi, mem in enumerate(members):
            if mi:
                time.sleep(GEFS_REQUEST_PAUSE_S)   # brief courtesy pause; S3 has no burst limit
            local = os.path.join(CACHE_DIR, f"gefs_{mem}_{cycle}z_f{fh:03d}.grib2")
            try:
                with open(local, "wb") as fhandle:
                    n_a = _grab(date_str, cycle, mem, fh, "a", fhandle)
                    n_b = _grab(date_str, cycle, mem, fh, "b", fhandle)
                if not probed:
                    # One-time check: pgrb2a carries TMP/RH only at 1000/925/850, so a zero-byte
                    # b file would silently blank Thompson and 700-500 RH for the whole GEFS
                    # column. Log both sizes so that failure mode is visible immediately.
                    logging.info(f"[GEFS PROBE] {mem} f{fh:03d}: pgrb2a {n_a/1024:.0f} KB + "
                                 f"pgrb2b {n_b/1024:.0f} KB via S3 byte-range "
                                 f"(mid/upper TMP+RH come from the b file).")
                    probed = True
                if (n_a + n_b) == 0:
                    continue
                prof = build_pad_profiles_from_grib(local, coords, debug=False).get(site)
                if not prof:
                    continue
                th = compute_launch_thermo(prof)
                if th:
                    per.append(th)
            except Exception as e:
                logging.debug(f"GEFS thermo {mem} f{fh:03d}: {e}")
            finally:
                if os.path.exists(local):
                    try:
                        os.remove(local)
                    except Exception:
                        pass
        # A row built from one or two members is not an ensemble — it is a single run wearing
        # an ensemble's label, and it was reaching the panel as a legitimate-looking final day
        # (its spread collapses to zero, which reads as high confidence rather than no data).
        # Rows below the floor are dropped outright.
        if per and len(per) >= GEFS_MIN_MEMBERS_PER_ROW:
            out[rk] = _ensemble_thermo_row(per)
        elif per:
            dropped_rows.append((rk, len(per)))

    if dropped_rows:
        logging.warning("GEFS thermo: dropped %s under-populated row(s) (need >=%d members): %s"
                        % (len(dropped_rows), GEFS_MIN_MEMBERS_PER_ROW,
                           ", ".join(f"{rk} had {n}" for rk, n in dropped_rows)))

    # Report the WEAKEST row, not the strongest. The old code took max() across rows, so a
    # single full row masked a final day built from one member — exactly the case that made
    # the panel look complete when it wasn't.
    counts = [r.get("n", 0) for r in out.values()]
    worst = min(counts) if counts else 0
    best = max(counts) if counts else 0
    logging.info(f"GEFS thermo: cycle {date_str} {cycle}z, members per row {worst}-{best} of "
                 f"{len(members)}, {len(out)} rows near {assess_hour}Z at {site.upper()} "
                 f"(index-of-member mean).")

    # Only cache a result that is actually worth freezing for the next six hours: every row
    # adequately populated AND the expected number of days present. Anything less returns
    # cycle_key=None, which forces a refetch on the next run instead.
    expected_days = len(picks)
    complete = (out and worst >= max(2, len(members) // 2) and len(out) >= expected_days)
    if not complete:
        logging.warning(f"GEFS thermo: incomplete (rows {len(out)}/{expected_days}, weakest row "
                        f"{worst}/{len(members)} members) — not caching, will refetch next run.")
        return out, None
    return out, cycle_key


def _ensemble_thermo_row(per):
    """Collapse a list of per-member compute_launch_thermo() dicts into one ensemble row.

    Scalars (Thompson, PWAT, RH) are averaged directly; winds are averaged in u/v COMPONENT space
    and re-derived to direction/speed so opposing directions don't average to nonsense. The
    lightning RF is run on EACH member's own environment and the probabilities averaged (with the
    member spread kept) — never on a mean sounding, whose moisture structure is smeared.
    Shared by the REFS and GEFS ensemble columns."""
    def _avg(key):
        v = [t[key] for t in per if t.get(key) is not None]
        return sum(v) / len(v) if v else None

    COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    row = {"n": len(per)}
    for u_key, v_key, pfx in (("mf_u", "mf_v", "mf"), ("av_u", "av_v", "av")):
        uu, vv = _avg(u_key), _avg(v_key)
        if uu is None or vv is None:
            continue
        frm = math.degrees(math.atan2(-uu, -vv)) % 360.0
        row[f"{pfx}_dir"] = round(frm)
        row[f"{pfx}_spd"] = round(math.hypot(uu, vv), 1)
        row[f"{pfx}_regime"] = COMPASS[int((frm + 22.5) // 45) % 8]
    ti, pw, rh = _avg("thompson"), _avg("pwat_in"), _avg("rh_700_500")
    if ti is not None:
        row["thompson"] = round(ti, 1)
    if pw is not None:
        row["pwat_in"] = round(pw, 2)
    if rh is not None:
        row["rh_700_500"] = round(rh, 1)
    # IVT magnitude is averaged as a SCALAR, matching how Thompson and PWAT are handled
    # here. That is the mean of the member magnitudes, not the magnitude of the mean
    # transport vector -- the two differ whenever members disagree on direction, and the
    # scalar mean is the one that stays comparable to the deterministic columns.
    iv = _avg("ivt")
    if iv is not None:
        row["ivt"] = round(iv, 1)
    member_p = []
    for t in per:
        p = rf_lightning_prob(t.get("thompson"),
                              rf_lightning_u_wind(t.get("mf_dir"), t.get("mf_spd")),
                              t.get("rh_700_500"))
        if p is not None:
            member_p.append(p)
    if member_p:
        row["ltg"] = round(sum(member_p) / len(member_p), 1)
        row["ltg_min"] = round(min(member_p), 1)
        row["ltg_max"] = round(max(member_p), 1)
        row["ltg_n"] = len(member_p)
    return row


def fetch_refs_member_thermo(site="kxmr", assess_hour=10):
    """Meteorologically valid REFS launch-thermo: pull EACH RRFS ensemble member's isobaric sounding
    at the site for the forecast hours that land on the assessment hour (10Z), compute the indices
    per member with compute_launch_thermo, and average the RESULTS (TI/PWAT as scalar means; mean
    flow in u/v component space). This is the correct ensemble number — never the index of the mean
    sounding. Returns {row_key: {mf_dir, mf_spd, mf_regime, thompson, pwat_in, n}} (empty on failure,
    so the panel simply omits REFS)."""
    if not REFS_MEMBER_THERMO_ENABLED:
        return {}
    sc = STN_COORDS.get(site)
    if not sc:
        return {}
    coords = {site: {"lat": sc["lat"], "lon": sc["lon"]}}

    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=3))
    session.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"})

    tmpl = "rrfs_a/rrfsens.{d}/{cc}/{mem}/rrfs.t{cc}z.{mem}.prslev.3km.f{fh:03d}.conus.grib2"

    def _idx_ok(d, cc, mem, fh):
        try:
            r = session.get(f"{RRFS_AWS_ROOT}/{tmpl.format(d=d, cc=cc, mem=mem, fh=fh)}.idx", timeout=12)
            return r.status_code == 200 and len(r.text) > 50
        except Exception:
            return False

    # newest rrfsens cycle whose m001 prslev reaches at least the first assess-hour valid time
    now = datetime.datetime.now(datetime.timezone.utc)
    date_str = cycle = None
    cyc_dt = None
    for back in range(0, 48):
        cand = now - datetime.timedelta(hours=back)
        if cand.hour not in (0, 6, 12, 18):
            continue
        d, cc = cand.strftime("%Y%m%d"), f"{cand.hour:02d}"
        first_fh = ((assess_hour - cand.hour) % 24) or 24
        if _idx_ok(d, cc, "m001", first_fh):
            date_str, cycle, cyc_dt = d, cc, cand.replace(minute=0, second=0, microsecond=0)
            break
    if not cycle:
        logging.warning("REFS member thermo: no rrfsens prslev cycle found; REFS omitted from panel.")
        return {}

    members = []
    for n in range(1, 21):
        mem = f"m{n:03d}"
        first_fh = ((assess_hour - cyc_dt.hour) % 24) or 24
        if _idx_ok(date_str, cycle, mem, first_fh):
            members.append(mem)
        elif members:
            break
    if not members:
        return {}

    fhs = [fh for fh in range(1, REFS_MEMBER_WINDOW_FH + 1)
           if (cyc_dt + datetime.timedelta(hours=fh)).hour == assess_hour]
    if not fhs:
        return {}

    # --- time-lagged ensemble: fold in the prior cycle's members, valid-time aligned ---------
    # Same principle as the Cumulus echo-top NMEP: pairing this cycle's members with the -6 h
    # cycle's members at the SAME valid time doubles the sample (5 -> 10). For daily airmass
    # indices a 6 h older forecast is a legitimate additional draw on the same airmass, so this is
    # more defensible here than it would be for convective placement.
    sources = [(date_str, cycle, cyc_dt, members)]
    if REFS_MEMBER_TLE:
        lag_dt = cyc_dt - datetime.timedelta(hours=6 * REFS_MEMBER_LAG_CYCLES)
        ld, lcc = lag_dt.strftime("%Y%m%d"), f"{lag_dt.hour:02d}"
        lag_first_fh = int((cyc_dt + datetime.timedelta(hours=fhs[0]) - lag_dt).total_seconds() // 3600)
        if lag_first_fh <= REFS_MEMBER_WINDOW_FH and _idx_ok(ld, lcc, "m001", lag_first_fh):
            lag_members = []
            for n in range(1, 21):
                mem = f"m{n:03d}"
                if _idx_ok(ld, lcc, mem, lag_first_fh):
                    lag_members.append(mem)
                elif lag_members:
                    break
            if lag_members:
                sources.append((ld, lcc, lag_dt, lag_members))

    out = {}
    for fh in fhs:
        valid = cyc_dt + datetime.timedelta(hours=fh)
        rk = f"{valid.day:02d}/{valid.hour:02d}"
        per = []
        for (s_date, s_cycle, s_dt, s_members) in sources:
            s_fh = int((valid - s_dt).total_seconds() // 3600)
            if s_fh < 1 or s_fh > REFS_MEMBER_WINDOW_FH:
                continue
            for mem in s_members:
                grib_url = f"{RRFS_AWS_ROOT}/{tmpl.format(d=s_date, cc=s_cycle, mem=mem, fh=s_fh)}"
                local = None
                try:
                    ir = session.get(grib_url + ".idx", timeout=15)
                    if ir.status_code != 200:
                        continue
                    local = _range_download_grib(session, grib_url, _parse_grib_idx(ir.text),
                                                 PAD_LEVELS_HPA, debug=False)
                    if not local:
                        continue
                    prof = build_pad_profiles_from_grib(local, coords, debug=False).get(site)
                    if not prof:
                        continue
                    th = compute_launch_thermo(prof)
                    if th:
                        per.append(th)
                except Exception as e:
                    logging.debug(f"REFS member thermo {s_cycle}z {mem} f{s_fh:03d}: {e}")
                finally:
                    if local and os.path.exists(local):
                        try:
                            os.remove(local)
                        except Exception:
                            pass
        if not per:
            continue
        out[rk] = _ensemble_thermo_row(per)

    src_txt = " + ".join(f"{s[1]}z({len(s[3])})" for s in sources)
    n_total = sum(len(s[3]) for s in sources)
    logging.info(f"REFS member thermo: cycle {date_str} {cycle}z, {n_total}-member TLE [{src_txt}], "
                 f"{len(out)} x {assess_hour}Z rows at {site.upper()} (index-of-member mean).")
    return out



def _ecmwf_ens_profiles_by_member(filepath, lat, lon, levels=None):
    """Split one multi-member ENS GRIB file into per-member point profiles.

    A single ENS file holds every requested member for a step. build_pad_profiles_from_grib()
    has no notion of members, so it would let each member overwrite the last and return only
    the final one's column. This walks the file grouping by perturbationNumber, then hands
    each member's level dict to the same _grib_levels_to_layers() everything else uses, so
    the profile schema stays identical.

    Returns {member_number: profile_layers}.
    """
    allowed = set(levels if levels is not None else ECMWF_ENS_LEVELS_HPA)
    per_member = {}
    iy = ix = None
    try:
        grbs = pygrib.open(filepath)
        for grb in grbs:
            try:
                if getattr(grb, "typeOfLevel", "") != "isobaricInhPa":
                    continue
                level = grb.level
                # Filter to the levels we actually REQUESTED, not the wider pad set. If the
                # server ever returns a stray upper level, letting it through would hand
                # compute_launch_thermo a single-level "anvil flow" — a plausible-looking
                # number derived from one level, which is worse than the honest em dash.
                if level not in allowed:
                    continue
                short = getattr(grb, "shortName", "")
                mem = getattr(grb, "perturbationNumber", None)
            except Exception:
                continue
            if short in ("t", "TMP"):
                field = "t"
            elif short in ("r", "RH"):
                field = "rh"
            elif short in ("gh", "HGT"):
                field = "hgt"
            elif short in ("u", "UGRD"):
                field = "u"
            elif short in ("v", "VGRD"):
                field = "v"
            else:
                continue
            if mem is None:
                continue
            if iy is None:
                # Nearest grid cell, resolved once from the first usable message.
                glats, glons = grb.latlons()
                gl = np.where(glons > 180, glons - 360.0, glons)
                d = (glats - lat) ** 2 + (gl - lon) ** 2
                iy, ix = np.unravel_index(np.argmin(d), d.shape)
            per_member.setdefault(mem, {}).setdefault(level, {})[field] = float(grb.values[iy, ix])
        grbs.close()
    except Exception as e:
        logging.error(f"ECMWF ENS GRIB parse failed for {os.path.basename(filepath)}: {e}")
        return {}

    out = {}
    for mem, levels in per_member.items():
        layers = _grib_levels_to_layers(levels)
        if layers:
            out[mem] = layers
    return out


def fetch_ecmwf_ens_member_thermo(site="kxmr", assess_hour=10, cache=None):
    """ECMWF ENS column for the 10Z panel: pull each perturbed member's sounding at the site,
    compute the indices PER MEMBER, and average the RESULTS via _ensemble_thermo_row — never
    from an ensemble-mean sounding, whose moisture structure is smeared.

    One retrieve per forecast step covering all members at once (10 requests would pay the
    index lookup ten times over for the same bytes). Returns (rows, cycle_key); `cache` may be
    a previous {"cycle":..., "rows":...} and is returned untouched when the cycle hasn't moved.
    """
    if not ECMWF_ENS_ENABLED:
        return {}, None
    sc = STN_COORDS.get(site)
    if not sc:
        return {}, None
    try:
        from ecmwf.opendata import Client
    except Exception as e:
        logging.warning(f"ecmwf-opendata not installed; skipping ECMWF ENS column ({e}).")
        return {}, None

    client = Client(source=ECMWF_SOURCE)
    members = list(range(1, max(1, ECMWF_ENS_MEMBERS) + 1))

    # Which cycle is newest? latest() probes the index rather than guessing at latency.
    try:
        init_dt = client.latest(stream="enfo", type="pf", levtype="pl",
                                param="t", number=1)
    except Exception as e:
        logging.error(f"ECMWF ENS: could not resolve latest cycle ({e}); skipping column.")
        return {}, None
    if init_dt is None:
        return {}, None
    cycle_key = init_dt.strftime("%Y%m%d%H")

    if ECMWF_ENS_CACHE_ENABLED and isinstance(cache, dict) \
            and cache.get("cycle") == cycle_key and cache.get("rows"):
        logging.info(f"ECMWF ENS: cycle {cycle_key} unchanged, reusing {len(cache['rows'])} cached rows.")
        return cache["rows"], cycle_key

    # ENS is 3-hourly to 144 h. Take the step nearest the assessment hour on each forecast
    # day, within ASSESS_HOUR_TOL — same rule the GEFS column uses.
    best_by_day = {}
    for fh in range(3, ECMWF_ENS_MAX_FH + 1, 3):
        v = init_dt + datetime.timedelta(hours=fh)
        diff = abs(v.hour - assess_hour)
        if diff > ASSESS_HOUR_TOL:
            continue
        key = v.strftime("%Y%m%d")
        if key not in best_by_day or diff < best_by_day[key][0]:
            best_by_day[key] = (diff, fh, v)
    picks = sorted(best_by_day.values(), key=lambda x: x[1])
    if not picks:
        logging.warning("ECMWF ENS: no forecast step landed near the assessment hour.")
        return {}, cycle_key

    out = {}
    t_start = time.time()
    total_mb = 0.0
    for (_diff, fh, valid) in picks:
        rk = f"{valid.day:02d}/{valid.hour:02d}"
        local = os.path.join(CACHE_DIR, f"ecens_{cycle_key}_f{fh:03d}.grib2")
        try:
            client.retrieve(
                stream="enfo", type="pf", number=members, step=fh,
                levtype="pl", levelist=ECMWF_ENS_LEVELS_HPA,
                param=ECMWF_ENS_PARAMS, target=local,
            )
            total_mb += os.path.getsize(local) / 1e6
            profiles = _ecmwf_ens_profiles_by_member(local, sc["lat"], sc["lon"])
            per = []
            for _mem, layers in sorted(profiles.items()):
                th = compute_launch_thermo(layers)
                if th:
                    per.append(th)
            if per:
                out[rk] = _ensemble_thermo_row(per)
        except Exception as e:
            logging.warning(f"ECMWF ENS f{fh:03d} ({rk}): {type(e).__name__}: {e}")
        finally:
            if os.path.exists(local):
                try:
                    os.remove(local)
                except Exception:
                    pass

    got = max((r.get("n", 0) for r in out.values()), default=0)
    logging.info(f"ECMWF ENS thermo: cycle {cycle_key}, {got}/{len(members)} members, "
                 f"{len(out)} rows near {assess_hour}Z at {site.upper()} "
                 f"({total_mb:.0f} MB in {time.time() - t_start:.0f}s, index-of-member mean).")

    # Never cache a badly degraded fetch — the cache is keyed to a 6-hourly cycle, so a
    # partial result would be frozen in for hours. cycle_key=None forces a retry next run.
    if out and got < max(2, len(members) // 2):
        logging.warning(f"ECMWF ENS: only {got}/{len(members)} members returned — not caching.")
        return out, None
    return out, cycle_key

def _convective_params(layers):
    """CAPE/CIN/LCL/LFC/EL for one profile, plus the existing KI/LI/Thompson/PWAT.

    Surface-based AND mixed-layer parcels are both returned: at the Cape the two diverge
    sharply on a sea-breeze day, and which one matters depends on whether the forcing is
    surface convergence or elevated. Returns {} when the profile is too thin to trust.

    MetPy is the reference path. The numpy fallback below it computes CAPE/CIN by direct
    integration; it agrees with MetPy to within a few percent on well-resolved soundings but
    degrades on coarse mandatory-level columns, so the engine used is always reported.
    """
    try:
        good = sorted([L for L in layers
                       if L.get("pres") and L.get("tmpc") is not None and L.get("dwpt") is not None],
                      key=lambda x: -x["pres"])
        if len(good) < 5:
            return {}
        out = {}
        if _HAVE_METPY:
            try:
                import metpy.calc as mc
                from metpy.units import units as u
                p = np.array([L["pres"] for L in good]) * u.hPa
                T = np.array([L["tmpc"] for L in good]) * u.degC
                Td = np.array([min(L["dwpt"], L["tmpc"]) for L in good]) * u.degC

                def _f(q, nd=1):
                    try:
                        v = float(np.atleast_1d(q.magnitude)[0])
                        return None if (v != v) else round(v, nd)
                    except Exception:
                        return None

                sbcape, sbcin = mc.surface_based_cape_cin(p, T, Td)
                out["sbcape"] = _f(sbcape, 0)
                out["sbcin"] = _f(sbcin, 0)
                try:
                    mlcape, mlcin = mc.mixed_layer_cape_cin(p, T, Td,
                                                            depth=_ML_DEPTH_HPA * u.hPa)
                    out["mlcape"] = _f(mlcape, 0)
                    out["mlcin"] = _f(mlcin, 0)
                except Exception:
                    pass
                lcl_p, lcl_t = mc.lcl(p[0], T[0], Td[0])
                out["lcl_p"] = _f(lcl_p, 1)
                try:
                    prof = mc.parcel_profile(p, T[0], Td[0])
                    lfc_p, _ = mc.lfc(p, T, Td, parcel_temperature_profile=prof)
                    el_p, _ = mc.el(p, T, Td, parcel_temperature_profile=prof)
                    out["lfc_p"] = _f(lfc_p, 1)
                    out["el_p"] = _f(el_p, 1)
                except Exception:
                    pass
                out["cape_engine"] = "metpy"
            except Exception as e:
                logging.debug(f"MetPy convective params failed, falling back: {e}")

        if "sbcape" not in out:
            out.update(_cape_numpy(good))
            out["cape_engine"] = "numpy"

        # Reuse the existing index calculation so the skew-T panel and the 10Z panel can
        # never disagree about Thompson or PWAT for the same profile.
        base = compute_launch_thermo(layers) or {}
        # Includes the 1000-700 mb mean flow and the 300-150 mb anvil flow. These are carried
        # through from compute_launch_thermo rather than recomputed for the skew-T, so the
        # sounding panel and the 10Z Synoptic table are reading literally the same numbers —
        # two independent layer-mean implementations would eventually disagree by a degree or
        # a knot and there would be no way to tell which was right.
        for k in ("k_index", "lifted_index", "thompson", "pwat_in", "rh_700_500",
                  "mf_dir", "mf_spd", "mf_regime", "av_dir", "av_spd", "av_regime"):
            if base.get(k) is not None:
                out[k] = base[k]

        # Cizek lightning probability for THIS profile.
        #
        # All three RF features are already sitting in `base` — Thompson, the 1000-700 mb mean
        # flow (from which the westerly-positive u component is rebuilt with the upstream
        # formula) and the 700-500 mb mean RH — so this costs one forest walk per profile and
        # is guaranteed to agree with the 10Z panel wherever they overlap, because it is
        # literally the same three numbers through the same function.
        #
        # OFF-LABEL WARNING, and the frontend repeats it: the forest was trained on the 10Z
        # KSC/CCSFS sounding only. Applied at 19Z it is being handed a daytime-heated Thompson
        # from a feature distribution it never saw with that label, so treat anything outside
        # roughly 08-12Z as a relative trend between models and hours, not a calibrated
        # probability. `ltg_hour_ok` carries that judgement to the UI rather than leaving the
        # panel to re-derive it.
        if SKEWT_LIGHTNING:
            u_ltg = rf_lightning_u_wind(out.get("mf_dir"), out.get("mf_spd"))
            ltg = rf_lightning_prob(out.get("thompson"), u_ltg, out.get("rh_700_500"))
            if ltg is not None:
                out["ltg"] = ltg
                out["ltg_u"] = round(u_ltg, 2)
        return out
    except Exception:
        return {}


def _cape_numpy(layers):
    """Surface-parcel CAPE/CIN/LCL/LFC/EL by direct integration, for when MetPy is absent.

    Lifts the surface parcel dry-adiabatically to its LCL, then along a pseudoadiabat, and
    integrates g*(Tv_parcel - Tv_env)/Tv_env through the column. Virtual temperature is used
    rather than plain temperature — ignoring it understates CAPE by roughly 10% in the moist
    Florida boundary layer.

    Two things this gets right that a naive integration does not:
      * Below the LCL the parcel's MIXING RATIO is conserved, not its dewpoint. Recomputing
        saturation mixing ratio at each level instead inflates the parcel's virtual
        temperature and badly overstates CAPE.
      * CIN is only meaningful up to the LFC. If the parcel never becomes buoyant, there is
        no LFC and no CAPE, and integrating negative area to the tropopause produces a
        meaningless five-figure CIN — so the search stops at CIN_SEARCH_TOP_HPA and reports
        no LFC instead.
    """
    CIN_SEARCH_TOP_HPA = 450.0   # above this with no LFC, the parcel is not going anywhere
    out = {}
    try:
        p0 = layers[0]["pres"]; t0 = layers[0]["tmpc"]; td0 = min(layers[0]["dwpt"], t0)
        tk = t0 + 273.15; tdk = td0 + 273.15
        tlcl = 1.0 / (1.0 / (tdk - 56.0) + math.log(tk / tdk) / 800.0) + 56.0  # Bolton 1980
        theta = tk * (1000.0 / p0) ** 0.2854
        p_lcl = 1000.0 * (tlcl / theta) ** (1.0 / 0.2854)
        out["lcl_p"] = round(p_lcl, 1)

        def _es(tc):
            return 6.112 * math.exp(17.67 * tc / (tc + 243.5))

        def _mixr(tc, pr):
            e = _es(tc)
            return 0.622 * e / max(1e-6, pr - e)

        def _tv(tc, w):
            return (tc + 273.15) * (1.0 + 0.61 * w)

        def _moist_lapse(tc, pr):
            tkl = tc + 273.15
            w = _mixr(tc, pr)
            num = 1.0 + (2.501e6 * w) / (287.0 * tkl)
            den = 1.0 + (2.501e6 ** 2 * w * 0.622) / (1004.0 * 287.0 * tkl * tkl)
            return (287.0 * tkl / (1004.0 * pr)) * (num / den)

        w0 = _mixr(td0, p0)          # conserved below the LCL
        cape = cin = 0.0
        lfc_p = el_p = None
        prev = None
        t_par = t0
        for L in layers:
            pr = L["pres"]
            if pr > p0:
                continue
            if prev is not None:
                dp = prev[0] - pr
                if dp <= 0:
                    continue
                if pr >= p_lcl:
                    t_par = (t_par + 273.15) * (pr / prev[0]) ** 0.2854 - 273.15
                else:
                    steps = max(1, int(dp / 5))
                    sub = dp / steps
                    pp = prev[0]
                    for _ in range(steps):
                        t_par += _moist_lapse(t_par, pp) * (-sub)
                        pp -= sub
            w_par = w0 if pr >= p_lcl else _mixr(t_par, pr)
            w_env = _mixr(min(L["dwpt"], L["tmpc"]), pr)
            tv_env = _tv(L["tmpc"], w_env)
            buoy = (_tv(t_par, w_par) - tv_env) / tv_env
            if prev is not None:
                dz = 29.27 * (L["tmpc"] + 273.15) * math.log(prev[0] / pr)
                contrib = 9.81 * buoy * dz
                if buoy > 0:
                    if lfc_p is None:
                        lfc_p = pr
                    cape += contrib
                elif lfc_p is None:
                    if pr >= CIN_SEARCH_TOP_HPA:
                        cin += contrib
                elif el_p is None:
                    el_p = pr
            prev = (pr, L["tmpc"])

        if lfc_p is None:
            # No level of free convection within the search layer: fully capped.
            out["sbcape"] = 0.0
            out["sbcin"] = round(min(0.0, cin), 0)
            return out
        out["sbcape"] = round(max(0.0, cape), 0)
        out["sbcin"] = round(min(0.0, cin), 0)
        out["lfc_p"] = round(lfc_p, 1)
        if el_p:
            out["el_p"] = round(el_p, 1)
    except Exception:
        pass
    return out


def build_soundings_export(combined_data, site="kxmr", max_hours=SKEWT_MAX_PROFILES):
    """Compact per-model, per-hour soundings at one site, for the frontend skew-T.

    Must run BEFORE generate_aviation_dashboard strips `_layers`, since it reuses exactly the
    profiles the matrix was built from — no refetching, and no risk of the skew-T disagreeing
    with the table beside it.
    """
    if not SKEWT_ENABLED:
        return None
    site = site.lower()
    models = (combined_data.get(site) or {})
    if not models:
        logging.warning(f"Skew-T export: no data for {site.upper()}.")
        return None

    t0 = time.time()
    out = {"site": site.upper(), "models": {}, "n_profiles": 0}
    skipped_thin = {}     # model -> count of profiles rejected for too few levels
    for mdl, rows in models.items():
        if not isinstance(rows, dict):
            continue
        # Hard exclusion by name, for columns that are useful in the matrix but are not
        # soundings. See SKEWT_EXCLUDE_MODELS — REFS publishes five pressure levels, which
        # draws as four straight segments and produces parcel numbers that are wrong in a
        # way that looks plausible.
        if mdl in SKEWT_EXCLUDE_MODELS:
            logging.info(f"Skew-T export: skipping '{mdl}' (SKEWT_EXCLUDE_MODELS) — "
                         f"{len(rows)} hour(s) not exported.")
            continue
        # Cap on profile COUNT, so an hourly model and a 3-hourly one reach different
        # horizons from the same number. That is intentional: the point is to export what
        # each model actually has, not to truncate them all to the shortest.
        keys = sorted(rows.keys(), key=lambda k: _row_sort_key(k))[:max_hours]
        prof_out = {}
        for rk in keys:
            prof = rows.get(rk)
            if not isinstance(prof, dict):
                continue
            layers = prof.get("_layers") or []
            if len(layers) < SKEWT_MIN_LEVELS:
                skipped_thin[mdl] = skipped_thin.get(mdl, 0) + 1
                continue
            lv = []
            for L in layers:
                if L.get("pres") is None or L.get("tmpc") is None:
                    continue
                if L["pres"] < SKEWT_TOP_HPA:
                    continue
                lv.append([
                    round(L["pres"], 1),
                    round(L["hght"]) if L.get("hght") is not None else None,
                    round(L["tmpc"], 1),
                    round(L["dwpt"], 1) if L.get("dwpt") is not None else None,
                    round(L["drct"]) if L.get("drct") is not None else None,
                    round(L["sknt"], 1) if L.get("sknt") is not None else None,
                ])
            if len(lv) < SKEWT_MIN_LEVELS:
                skipped_thin[mdl] = skipped_thin.get(mdl, 0) + 1
                continue
            entry = {"lv": lv}
            params = _convective_params(layers)
            if params:
                entry["p"] = params
            # Carry the provenance flags so the skew-T panel can repeat the same warnings
            # the matrix cell shows rather than presenting a coarse column as authoritative.
            if prof.get("coarse"):
                entry["coarse"] = prof["coarse"]
            if prof.get("stale"):
                entry["stale"] = prof["stale"]
            prof_out[rk] = entry
            out["n_profiles"] += 1
        if prof_out:
            out["models"][mdl] = prof_out

    out["elapsed_s"] = round(time.time() - t0, 1)
    if skipped_thin:
        # Say so rather than letting a column quietly vanish. A model dropped for thin
        # profiles and a model that failed to fetch look identical in the output file.
        logging.info("Skew-T export: dropped thin profiles (< "
                     f"{SKEWT_MIN_LEVELS} usable levels): "
                     + ", ".join(f"{m}={n}" for m, n in sorted(skipped_thin.items())))
    logging.info(f"Skew-T export: {out['n_profiles']} profiles across "
                 f"{len(out['models'])} models at {site.upper()} in {out['elapsed_s']}s "
                 f"[{', '.join(sorted(out['models'])) or 'none'}].")
    return out


def _row_sort_key(rk):
    """Sort 'DD/HH' row keys across a month boundary."""
    try:
        d, h = map(int, rk.split("/"))
        return (d if d > 15 else d + 32, h)
    except Exception:
        return (999, 999)

def build_launch_thermo(combined_data, site="kxmr", assess_hour=10, refs_member_rows=None,
                        gefs_member_rows=None, ecens_member_rows=None):
    """Assemble the launch-thermo panel: for each model that has a KXMR sounding, one row per
    forecast day at the assessment hour (10Z), with mean flow, regime, Thompson Index (+percentile),
    and PWAT (+percentile). Returns {"site","hour","models":[...],"by_model":{model:[rows]}}."""
    now = datetime.datetime.now(datetime.timezone.utc)
    site_models = combined_data.get(site, {}) or {}
    by_model = {}
    for model, rows in site_models.items():
        if not isinstance(rows, dict):
            continue
        # The REFS ensemble-MEAN sounding. Its thermodynamic indices are biased (see
        # REFS_MEAN_IN_PANEL) because averaging RH across members destroys moisture structure.
        # Included anyway when the flag is on — the flow fields are unaffected and a REFS row
        # is better than no REFS row — but every row is tagged so the panel can say so.
        if model == "refs" and not REFS_MEAN_IN_PANEL:
            continue
        day_rows = []
        # Gather candidate profiles per forecast day within ASSESS_HOUR_TOL of the assessment hour,
        # then keep the one nearest to it (exact 10Z wins). Short-range models (RAP/HRRR) frequently
        # skip exactly 10Z depending on cycle timing, so an exact-only match dropped them entirely.
        day_cands = {}
        for row_key, prof in rows.items():
            if not isinstance(prof, dict):
                continue
            try:
                dd, hh = map(int, row_key.split("/"))
            except Exception:
                continue
            diff = abs(hh - assess_hour)
            if diff > ASSESS_HOUR_TOL or not prof.get("_layers"):
                continue
            day_cands.setdefault(dd, []).append((diff, hh, prof))
        for dd, cands in day_cands.items():
            cands.sort(key=lambda c: (c[0], c[1]))
            for diff, hh, prof in cands:
                th = compute_launch_thermo(prof["_layers"])  # MetPy (mixed-layer LI), on demand
                # Count levels that actually carry humidity. A column too thin here keeps its
                # wind fields and drops the moisture-driven indices rather than publishing
                # numbers derived from one RH level.
                _rh_n = sum(1 for _L in (prof.get("_layers") or [])
                            if _L.get("rh") is not None or _L.get("dwpt") is not None)
                if th and _rh_n < PANEL_MIN_RH_LEVELS:
                    for _k in ("k_index", "lifted_index", "thompson", "pwat_in", "pwat_mm",
                               "rh_700_500", "ivt", "ivt_dir"):
                        th[_k] = None
                    th["thin_moisture"] = _rh_n
                if not th:
                    continue
                day_label, date_str, sort_key, month, _yr = _valid_day_fields(dd, now)
                ti = th.get("thompson")
                pwat = th.get("pwat_in")
                u_ltg = rf_lightning_u_wind(th.get("mf_dir"), th.get("mf_spd"))
                rh_ltg = th.get("rh_700_500")
                ltg = rf_lightning_prob(ti, u_ltg, rh_ltg)
                day_rows.append({
                    "day": day_label,
                    "date": date_str,
                    "sort": sort_key,
                    "vhh": hh,
                    "month": month,
                    "ltg": ltg,
                    "ltg_u": None if u_ltg is None else round(u_ltg, 2),
                    "ltg_rh": rh_ltg,
                    "rh_pct": _climo_percentile(rh_ltg, RH75_CLIMO_XMR.get(month), RH75_PCTL_POINTS),
                    "mf_dir": th.get("mf_dir"),
                    "mf_spd": th.get("mf_spd"),
                    "regime": th.get("mf_regime"),
                    "av_dir": th.get("av_dir"),
                    "av_spd": th.get("av_spd"),
                    "ti": ti,
                    "ti_pct": _climo_percentile(ti, THOMPSON_CLIMO_XMR.get(month), THOMPSON_PCTL_POINTS),
                    "pwat": pwat,
                    "pwat_pct": _climo_percentile(pwat, PWAT_CLIMO_XMR.get(month), PWAT_PCTL_POINTS),
                    # No climatology table for IVT yet, so no percentile badge -- the raw
                    # value only. Add PWAT-style monthly breaks later if it earns one.
                    "ivt": th.get("ivt"),
                    "ivt_dir": th.get("ivt_dir"),
                    # Tag REFS rows so the frontend can flag the mean-sounding caveat. The
                    # ensemble columns use "...N-mem" here; this is deliberately distinct.
                    # Tag the row so the panel can say which fields to trust. "thin-moisture"
                    # means the moisture indices were suppressed for lack of humidity levels.
                    "engine": ("thin-moisture" if th.get("thin_moisture") is not None
                               else ("mean-sounding" if model == "refs" else th.get("engine"))),
                    "rh_levels": th.get("thin_moisture"),
                })
                break
        if day_rows:
            day_rows.sort(key=lambda r: r["sort"])
            by_model[model] = day_rows

    # Ensemble columns (REFS, GEFS) come in pre-averaged from their member fetches: per-member
    # indices averaged, never the index of a mean sounding. Same row shape as the deterministic
    # models, plus the member count and the lightning spread.
    def _add_ensemble(model_name, member_rows):
        if not member_rows:
            return
        rows_out = []
        for row_key, r in member_rows.items():
            try:
                dd, hh = map(int, row_key.split("/"))
            except Exception:
                continue
            day_label, date_str, sort_key, month, _yr = _valid_day_fields(dd, now)
            ti, pwat = r.get("thompson"), r.get("pwat_in")
            rows_out.append({
                "day": day_label,
                "date": date_str,
                "sort": sort_key,
                "vhh": hh,
                "month": month,
                "ltg": r.get("ltg"),
                "ltg_min": r.get("ltg_min"),
                "ltg_max": r.get("ltg_max"),
                "ltg_n": r.get("ltg_n"),
                "ltg_u": (None if r.get("mf_dir") is None else
                          round(rf_lightning_u_wind(r.get("mf_dir"), r.get("mf_spd")), 2)),
                "ltg_rh": r.get("rh_700_500"),
                "rh_pct": _climo_percentile(r.get("rh_700_500"), RH75_CLIMO_XMR.get(month), RH75_PCTL_POINTS),
                "ltg_members": r.get("n"),
                "mf_dir": r.get("mf_dir"),
                "mf_spd": r.get("mf_spd"),
                "regime": r.get("mf_regime"),
                "av_dir": r.get("av_dir"),
                "av_spd": r.get("av_spd"),
                "ti": ti,
                "ti_pct": _climo_percentile(ti, THOMPSON_CLIMO_XMR.get(month), THOMPSON_PCTL_POINTS),
                "pwat": pwat,
                "pwat_pct": _climo_percentile(pwat, PWAT_CLIMO_XMR.get(month), PWAT_PCTL_POINTS),
                "ivt": r.get("ivt"),
                "ivt_dir": r.get("ivt_dir"),
                "engine": f"metpy\u00b7{r.get('n', 0)}-mem",
            })
        if rows_out:
            rows_out.sort(key=lambda x: x["sort"])
            by_model[model_name] = rows_out

    _add_ensemble("refs", refs_member_rows)
    _add_ensemble("gefs", gefs_member_rows)
    _add_ensemble("ecens", ecens_member_rows)

    # order models: put the ones with the most rows first, stable-ish preferred order
    pref = ["gfs", "ecmwf", "gefs", "ecens", "rrfs", "refs", "rap", "hrrr"]
    models = sorted(by_model.keys(), key=lambda m: (pref.index(m) if m in pref else 99, m))

    # Dump the EXACT feature values fed to the Cizek RF so they can be typed straight into the
    # upstream Streamlit tool and compared. Feature order matches model.feature_names_in_.
    try:
        logging.info("=" * 78)
        logging.info(f"CIZEK LIGHTNING RF INPUTS — {site.upper()} {assess_hour:02d}Z "
                     f"(Thompson_Index, 1000-700mb_Average_U-Wind_Component[kt], 700-500mb_Average_RH[%])")
        logging.info(f"  {'model':6s} {'valid':11s} {'hr':>4s} {'Thompson':>9s} {'U-wind':>8s} {'RH':>7s} {'P(ltg)':>8s}")
        for m in models:
            for r in by_model[m]:
                hr = f"{r.get('vhh'):02d}Z" if r.get("vhh") is not None else "--"
                ti = "n/a" if r.get("ti") is None else f"{r['ti']:9.1f}"
                uu = "     n/a" if r.get("ltg_u") is None else f"{r['ltg_u']:8.2f}"
                rh = "    n/a" if r.get("ltg_rh") is None else f"{r['ltg_rh']:7.1f}"
                pp = "     n/a" if r.get("ltg") is None else f"{r['ltg']:7.1f}%"
                note = ""
                if m == "refs" and r.get("ltg_members"):
                    note = (f"   [P is the mean of {r['ltg_members']} per-member probabilities; "
                            f"features shown are member means and will NOT reproduce it exactly]")
                logging.info(f"  {m:6s} {r['date']:11s} {hr:>4s} {ti} {uu} {rh} {pp}{note}")
        logging.info("=" * 78)
    except Exception as e:
        logging.debug(f"Cizek RF input dump failed: {e}")

    return {"site": site.upper(), "hour": assess_hour, "models": models, "by_model": by_model}


def generate_aviation_dashboard(stations, models, current_sounding_matrix, time_rows, pad_matrix=None):
    href_lightning, href_maps = fetch_href_lightning(time_rows)

    # HREF Calibrated Thunder (HREFCT): ML-calibrated probability of >=1 CG flash within
    # 20 km. Fetch both the 1-hour and 4-hour windows. The 4-hour field also drives the
    # calibrated-thunder spatial slider; 1-hour supplies its own table column + maps.
    try:
        ct1_points, ct1_maps = fetch_calibrated_thunder(window="1hr")
    except Exception as e:
        logging.error(f"HREFCT 1hr fetch failed: {e}")
        ct1_points, ct1_maps = {stn: {} for stn in STATIONS}, {}
    try:
        ct4_points, ct4_maps = fetch_calibrated_thunder(window="4hr")
    except Exception as e:
        logging.error(f"HREFCT 4hr fetch failed: {e}")
        ct4_points, ct4_maps = {stn: {} for stn in STATIONS}, {}

    history_runs = []
    prior_thermo_runs = []
    prior_gefs_cache = None
    prior_ecens_cache = None
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                existing = json.load(f)
            # Tolerate the legacy flat-array history.json format from before href_maps_latest existed.
            history_runs = existing.get("runs", []) if isinstance(existing, dict) else existing
            if isinstance(existing, dict):
                prior_thermo_runs = existing.get("launch_thermo_runs", []) or []
                prior_gefs_cache = existing.get("gefs_cache") or None
                prior_ecens_cache = existing.get("ecens_cache") or None
        except Exception:
            history_runs = []

    # Merge launch-pad (raw-GRIB) soundings into the same station-keyed data block so the
    # frontend treats them identically to the BUFKIT stations (just extra dropdown entries).
    combined_data = dict(current_sounding_matrix)
    if pad_matrix:
        for pid, model_data in pad_matrix.items():
            combined_data[pid] = model_data

    # Convective (cumulus/anvil) mask: tag site-hours where HRRR composite reflectivity shows a
    # convective core within ~10 nm, so the Thick Cloud Layer / Max Layer Thickness fields read as
    # cu/anvil-governed (their own LLCC rules) rather than a stratiform thick-cloud bust. HRRR is
    # the convection truth source and the tag is applied across every model column for that
    # site-hour; the underlying thickness values are preserved so nothing is lost.
    # Spatial IVT panels. Independent of everything below: a failure here costs the IVT
    # maps and nothing else, which is why it gets its own try and returns {} rather than
    # raising. GFS is the only enabled model by default -- see ivt_maps.py for why RAP/NAM
    # are optional and HRRR/GEFS are excluded outright.
    ivt_map_set = {}
    if ivt_maps.IVT_MAPS_ENABLED:
        try:
            ivt_map_set = ivt_maps.fetch_ivt_maps()
        except Exception as e:
            logging.error(f"[IVT] map set failed, continuing without it: {e}")

    refc_maps = {}   # {row_key: "maps/refc/....png"} — backs the ANVIL hover popup
    if CONVECTIVE_MASK_ENABLED:
        try:
            refc, refc_maps_out = fetch_convective_reflectivity(time_rows)
            refc_maps.update(refc_maps_out or {})
            tagged = 0
            anvil_tagged = 0
            anvil_diag = 0
            for sid, hours in (refc or {}).items():
                site_models = combined_data.get(sid)
                if not site_models:
                    continue
                for row_key, cell in hours.items():
                    if not isinstance(cell, dict):
                        continue
                    dbz = cell.get("nbr")
                    point = cell.get("point")
                    sectors = cell.get("sectors") or {}
                    conv = (dbz is not None and dbz >= CONVECTIVE_DBZ)
                    for _model, mrows in site_models.items():
                        p = mrows.get(row_key) if isinstance(mrows, dict) else None
                        if not isinstance(p, dict):
                            continue
                        p["refc_nbr"] = dbz
                        if conv:
                            p["convective"] = True
                            tagged += 1
                        elif ANVIL_MASK_ENABLED:
                            hit = _anvil_eval(p, point, sectors)
                            if hit and p.get("thick_layer") == 1:
                                # Real mask: this would have read VIOLATED, but it's anvil debris,
                                # governed by the Anvil rule instead.
                                p["anvil"] = True
                                p["anvil_src_dbz"] = hit["src"]
                                p["anvil_dir"] = hit["dir"]
                                anvil_tagged += 1
                            elif hit and ANVIL_DIAG_MODE:
                                # Diagnostic only: anvil conditions are met but there's no thick-
                                # layer violation to suppress. Recorded for validation; the cell
                                # keeps its normal OK/value rendering.
                                p["anvil_diag"] = True
                                p["anvil_src_dbz"] = hit["src"]
                                p["anvil_dir"] = hit["dir"]
                                anvil_diag += 1
            logging.info(f"Convective mask applied: {tagged} site-hour-model cells tagged convective; "
                         f"{anvil_tagged} tagged anvil"
                         + (f"; {anvil_diag} anvil-overhead (diagnostic only)." if ANVIL_DIAG_MODE else "."))
        except Exception as e:
            logging.error(f"Convective/anvil mask failed, continuing without it: {e}")

    # REFS Cumulus Cloud standoff probabilities: interpolate the REFS echo-top exceedance curves
    # at each site's REFS-mean isotherm heights and store the go/no-go probabilities on the REFS
    # profile. Rule a = P(top>=-20C within 10 nm); Rule b = P(top>=-10C within 5 nm). Both isotherms
    # sit above the 20 kft echo-top floor in this environment, so they interpolate cleanly.
    refs_cumulus_maps = {}
    refs_cumulus_maps10 = {}
    if REFS_CUMULUS_ENABLED:
        try:
            KFT_TO_M = 304.8
            filled = 0
            member_matrix, member_maps5, member_maps10 = ({}, {}, {})
            if REFS_MEMBER_NMEP_ENABLED:
                member_matrix, member_maps5, member_maps10 = (
                    fetch_rrfsens_member_nmep() if RRFS_MEMBER_NMEP_ENABLED else ({}, {}, {}))

            if member_matrix:
                # TRUE member time-lagged NMEP at the real 5/10 nm radius. For each site-hour count
                # the fraction of members whose in-ring MAX echo top reaches the isotherm height —
                # a direct height comparison, so there is no fixed-knot 20 kft floor (no '*').
                refs_cumulus_maps = member_maps5      # -10C/5nm hover -> P(top>20kft) ~5nm
                refs_cumulus_maps10 = member_maps10    # -20C/10nm hover -> P(top>25kft) ~10nm
                for sid, hours in member_matrix.items():
                    refs_rows = (combined_data.get(sid, {}) or {}).get("refs")
                    if not isinstance(refs_rows, dict):
                        continue
                    for row_key, ets in hours.items():
                        p = refs_rows.get(row_key)
                        if not isinstance(p, dict):
                            continue
                        h10 = p.get("hght_10c")  # kft, REFS-mean isotherm heights
                        h20 = p.get("hght_20c")
                        et5 = ets.get("et5") or []
                        et10 = ets.get("et10") or []
                        if h10 and et5:
                            n = len(et5)
                            h10m = h10 * KFT_TO_M
                            p["cuP_neg10_5nm"] = round(100.0 * sum(1 for e in et5 if e >= h10m) / n)
                            p["cuP_neg10_5nm_q"] = f"nmep{n}"
                        if h20 and et10:
                            n = len(et10)
                            h20m = h20 * KFT_TO_M
                            p["cuP_neg20_10nm"] = round(100.0 * sum(1 for e in et10 if e >= h20m) / n)
                            p["cuP_neg20_10nm_q"] = f"nmep{n}"
                        filled += 1
                logging.info(f"REFS cumulus (member NMEP) computed for {filled} REFS site-hours.")
            else:
                # Fallback: the 40 km enspost exceedance-curve method (interpolate P(top>H) at the
                # isotherm height). Kept so the column never goes dark if members are unavailable.
                etop, refs_cumulus_maps = fetch_refs_echotop_probs()
                refs_cumulus_maps10 = refs_cumulus_maps  # fallback has only the one 40 km field
                for sid, hours in (etop or {}).items():
                    refs_rows = (combined_data.get(sid, {}) or {}).get("refs")
                    if not isinstance(refs_rows, dict):
                        continue
                    for row_key, curves in hours.items():
                        p = refs_rows.get(row_key)
                        if not isinstance(p, dict):
                            continue
                        h10 = p.get("hght_10c")  # kft, REFS-mean isotherm heights
                        h20 = p.get("hght_20c")
                        if h10:
                            val, q = _interp_exceedance(curves.get("c5", {}), h10 * KFT_TO_M)
                            p["cuP_neg10_5nm"] = val
                            p["cuP_neg10_5nm_q"] = q
                        if h20:
                            val, q = _interp_exceedance(curves.get("c10", {}), h20 * KFT_TO_M)
                            p["cuP_neg20_10nm"] = val
                            p["cuP_neg20_10nm_q"] = q
                        filled += 1
                logging.info(f"REFS cumulus (enspost 40km fallback) computed for {filled} REFS site-hours.")
        except Exception as e:
            logging.error(f"REFS cumulus probabilities failed, continuing without them: {e}")

    current_timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    current_entry = {
        "timestamp": current_timestamp,
        "data": combined_data,
        # HREF lightning point/percentage data DOES participate in dprog/dt history.
        "href_lightning": href_lightning,
        # Calibrated-thunder point probabilities (1hr + 4hr) also participate in history.
        "ct1_points": ct1_points,
        "ct4_points": ct4_points,
    }

    if not history_runs or history_runs[0]["timestamp"] != current_timestamp:
        history_runs.insert(0, current_entry)
    history_runs = history_runs[:5]

    # The HREF spatial PNG maps are NOT part of dprog/dt history — they always reflect
    # only the latest run and get fully overwritten (and pruned) each pipeline pass.
    blank_basemap_path = generate_blank_basemap()
    href_maps_latest = {
        "timestamp": current_timestamp,
        "href_maps": href_maps,
        "ct1_maps": ct1_maps,
        "ct4_maps": ct4_maps,
        "refs_cumulus_maps": refs_cumulus_maps,
        "refs_cumulus_maps10": refs_cumulus_maps10,
        "blank_map": blank_basemap_path,
    }

    # Launch-thermo panel (KXMR, 10Z per day) — latest run only, like the maps.
    try:
        refs_member_rows = fetch_refs_member_thermo(site="kxmr", assess_hour=10)
    except Exception as e:
        logging.error(f"REFS member thermo fetch failed: {e}")
        refs_member_rows = {}
    try:
        gefs_member_rows, gefs_cycle_key = _run_with_deadline(
            lambda: fetch_gefs_member_thermo(
                site="kxmr", assess_hour=10, cache=prior_gefs_cache),
            SOURCE_DEADLINE_S, "GEFS", default=({}, None))
    except Exception as e:
        logging.error(f"GEFS member thermo fetch failed: {e}")
        gefs_member_rows, gefs_cycle_key = {}, None
    # Same cache-preservation rule as ECMWF ENS below.
    if not gefs_member_rows and isinstance(prior_gefs_cache, dict) and prior_gefs_cache.get("rows"):
        gefs_member_rows = prior_gefs_cache["rows"]
        gefs_cycle_key = prior_gefs_cache.get("cycle")
        logging.warning(f"GEFS: fetch produced nothing; reusing cached cycle "
                        f"{gefs_cycle_key} ({len(gefs_member_rows)} rows).")
    try:
        ecens_member_rows, ecens_cycle_key = _run_with_deadline(
            lambda: fetch_ecmwf_ens_member_thermo(
                site="kxmr", assess_hour=10, cache=prior_ecens_cache),
            ECMWF_ENS_DEADLINE_S, "ECMWF ENS", default=({}, None))
    except Exception as e:
        logging.error(f"ECMWF ENS member thermo fetch failed: {e}")
        ecens_member_rows, ecens_cycle_key = {}, None
    # A failed ENS fetch must not also DESTROY the cache. Without this, one timeout wrote
    # ecens_cache=None, which forced a full ~940 MB refetch on the next hourly run, which
    # timed out the same way — the column stayed missing until a cycle happened to land
    # inside the deadline. Falling back to the previous cycle's rows keeps the panel
    # populated with a real (older) forecast and keeps the cache alive.
    if not ecens_member_rows and isinstance(prior_ecens_cache, dict) and prior_ecens_cache.get("rows"):
        ecens_member_rows = prior_ecens_cache["rows"]
        ecens_cycle_key = prior_ecens_cache.get("cycle")
        logging.warning(f"ECMWF ENS: fetch produced nothing; reusing cached cycle "
                        f"{ecens_cycle_key} ({len(ecens_member_rows)} rows) rather than "
                        f"dropping the column and discarding the cache.")
    try:
        launch_thermo = build_launch_thermo(combined_data, site="kxmr", assess_hour=10,
                                            refs_member_rows=refs_member_rows,
                                            gefs_member_rows=gefs_member_rows,
                                            ecens_member_rows=ecens_member_rows)
        logging.info(f"Launch thermo: {len(launch_thermo['models'])} models, "
                     f"rows/model={ {m: len(launch_thermo['by_model'][m]) for m in launch_thermo['models']} }")
    except Exception as e:
        logging.error(f"Launch thermo build failed: {e}")
        launch_thermo = {"site": "KXMR", "hour": 10, "models": [], "by_model": {}}

    # Skew-T export MUST happen here: it reuses the very `_layers` the next block deletes.
    try:
        skewt = build_soundings_export(combined_data, site=SKEWT_SITE)
        if skewt:
            with open(SKEWT_FILE, "w") as f:
                json.dump(skewt, f, separators=(",", ":"))
            logging.info(f"Skew-T soundings written to {SKEWT_FILE} "
                         f"({os.path.getsize(SKEWT_FILE) / 1e6:.1f} MB).")
    except Exception as e:
        logging.error(f"Skew-T export failed (dashboard continues): {e}")

    # Strip the raw sounding layers stashed for on-demand thermo — they must NOT bloat history.json.
    for _sid, _models in combined_data.items():
        if not isinstance(_models, dict):
            continue
        for _mk, _rows in _models.items():
            if not isinstance(_rows, dict):
                continue
            for _prof in _rows.values():
                if isinstance(_prof, dict):
                    _prof.pop("_layers", None)
                    _prof.pop("av_u", None)
                    _prof.pop("av_v", None)

    # Panel DPROG/DT: unlike the spatial maps, the 10Z panel DOES keep run history so trends in
    # Thompson/PWAT/lightning can be eyeballed run over run. Current + 3 back.
    thermo_runs = [r for r in prior_thermo_runs
                   if isinstance(r, dict) and r.get("timestamp") != current_timestamp]
    thermo_runs.insert(0, {"timestamp": current_timestamp, "thermo": launch_thermo})
    thermo_runs = thermo_runs[:LAUNCH_THERMO_HISTORY_RUNS]

    # Which model run produced each column, so the frontend can label "HRRR (12Z)".
    # Panel-only ensembles report their own cycle separately — they are not matrix columns.
    panel_cycles = {}
    if gefs_cycle_key:
        panel_cycles["gefs"] = gefs_cycle_key
    elif isinstance(prior_gefs_cache, dict) and prior_gefs_cache.get("cycle"):
        panel_cycles["gefs"] = prior_gefs_cache["cycle"]
    if ecens_cycle_key:
        panel_cycles["ecens"] = ecens_cycle_key
    elif isinstance(prior_ecens_cache, dict) and prior_ecens_cache.get("cycle"):
        panel_cycles["ecens"] = prior_ecens_cache["cycle"]

    payload = {
        "runs": history_runs,
        "model_cycles": _cycles_payload(),
        "panel_cycles": panel_cycles,
        "href_maps_latest": href_maps_latest,
        "launch_thermo": launch_thermo,
        "launch_thermo_runs": thermo_runs,
        "refc_maps": refc_maps,
        # {model: {"cycle": "YYYYMMDD HHz", "maps": {"f024": "maps/ivt/gfs/....png"}}}
        "ivt_maps": ivt_map_set,
        # GEFS cycles 6-hourly while this runs hourly; cache the rows so the fetch is skipped
        # until a new cycle posts.
        "gefs_cache": ({"cycle": gefs_cycle_key, "rows": gefs_member_rows}
                       if (GEFS_CACHE_ENABLED and gefs_cycle_key and gefs_member_rows) else None),
        "ecens_cache": ({"cycle": ecens_cycle_key, "rows": ecens_member_rows}
                        if (ECMWF_ENS_CACHE_ENABLED and ecens_cycle_key and ecens_member_rows) else None),
        # Monthly percentile distributions, shipped so the panel can draw box-and-whisker plots
        # from exactly the same numbers the percentile badges use.
        "climo": {
            "points": CLIMO_PCTL_POINTS_15,
            "thompson": THOMPSON_CLIMO_XMR,
            "pwat": PWAT_CLIMO_XMR,
            "rh75": RH75_CLIMO_XMR,
        },
    }

    with open(HISTORY_FILE, "w") as f:
        json.dump(payload, f, indent=2)

    # Keep EVERY current-run PNG: density thresholds (nested), 1-hr CT and 4-hr CT (flat),
    # plus the blank basemap. Collected recursively so no map type gets dropped from the
    # keep-set — the previous `{**href_maps, **ct_all_maps}` merge silently discarded density
    # maps (and ct1 maps) wherever a row key was shared, so they were pruned right after
    # being generated. That was why only a handful of density hours survived on disk.
    # ivt_map_set is NESTED ({model: {"maps": {...}}}), not flat like the others, so the
    # inner dicts are handed to _collect_map_paths directly. Leaving it out entirely is the
    # exact failure the comment above describes: the PNGs are written, then pruned in the
    # same run, and the panel shows broken images with no error anywhere in the log.
    _ivt_paths = [v.get("maps", {}) for v in (ivt_map_set or {}).values()]
    referenced_paths = _collect_map_paths(href_maps, ct1_maps, ct4_maps, blank_basemap_path,
                                          refc_maps, *_ivt_paths)
    prune_stale_maps(referenced_paths)
    logging.info("Dashboard matrix completely compiled and written to history.json.")


def run_pipeline():
    logging.info("Starting complete structural iteration run...")
    purge_workspace()
    sounding_matrix = {stn: {mdl: {} for mdl in MODELS} for stn in STATIONS}

    temp_time_rows_set = set()
    with requests.Session() as session:
        # Our own backoff handles retries; let urllib3 fail fast rather than silently
        # stacking a second retry layer on top of an already-throttled host.
        session.mount("https://", requests.adapters.HTTPAdapter(
            pool_connections=BUFKIT_MAX_CONCURRENCY,
            pool_maxsize=BUFKIT_MAX_CONCURRENCY,
            max_retries=0))
        # The pool is sized to the semaphore so we don't park a dozen threads on a gate
        # they can't pass; _BUFKIT_GATE is the hard guarantee either way.
        with concurrent.futures.ThreadPoolExecutor(max_workers=BUFKIT_MAX_CONCURRENCY) as executor:
            futures = [
                executor.submit(fetch_station_model, session, s, m)
                for s in STATIONS
                for m in BUFKIT_MODELS
            ]
            for future in concurrent.futures.as_completed(futures):
                stn, model, data = future.result()
                if data:
                    sounding_matrix[stn][model] = data
                    temp_time_rows_set.update(data.keys())

    ok = sum(1 for s in STATIONS for m in BUFKIT_MODELS if sounding_matrix[s].get(m))
    total = len(STATIONS) * len(BUFKIT_MODELS)
    logging.info(f"BUFKIT: {ok}/{total} station-model columns fetched.")
    if ok < total:
        # PSU throttling should degrade the board, not blank it.
        carry_forward_missing(sounding_matrix, models_to_check=BUFKIT_MODELS)
        for s in STATIONS:
            for m in BUFKIT_MODELS:
                temp_time_rows_set.update((sounding_matrix[s].get(m) or {}).keys())


    time_rows = sorted(list(temp_time_rows_set))
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    trimmed_rows = []
    for row in time_rows:
        try:
            d_part, h_part = map(int, row.split("/"))
            if d_part < now_utc.day and now_utc.day - d_part < 25:
                continue
            if d_part == now_utc.day and h_part < now_utc.hour:
                continue
            trimmed_rows.append(row)
        except Exception:
            trimmed_rows.append(row)
    if trimmed_rows:
        time_rows = trimmed_rows

    # Fetch launch-pad soundings from raw GRIB2 (additive; independent of BUFKIT stations).
    try:
        pad_matrix = fetch_all_pad_soundings()
        pad_hours = sum(len(m.get("hrrr", {})) for m in pad_matrix.values())
        logging.info(f"Launch-pad soundings assembled ({pad_hours} HRRR pad-hours across {len(pad_matrix)} pads).")
    except Exception as e:
        logging.error(f"Launch-pad sounding fetch failed, continuing without pads: {e}")
        pad_matrix = None

    # Fetch RRFS + REFS + HRRR columns from AWS (single idx-based pass). RRFS/REFS are
    # point-extracted for BOTH pads and airports; AWS HRRR is applied to PADS ONLY (the
    # airports already have superior BUFKIT HRRR soundings, so we don't overwrite those).
    if RRFS_ENABLED or REFS_ENABLED:
        try:
            aws_matrix = fetch_all_rrfs_refs_soundings(include_hrrr=True)
            if pad_matrix is None:
                pad_matrix = {pid: {} for pid in LAUNCH_PADS}
            for sid, kinds in aws_matrix.items():
                is_pad = sid in LAUNCH_PADS
                target = pad_matrix if is_pad else sounding_matrix
                target.setdefault(sid, {})
                for kind, rows in kinds.items():
                    if not rows:
                        continue
                    # AWS HRRR normally fills pad columns only, because the airports have
                    # richer ~40-level BUFKIT HRRR soundings. When BUFKIT is unavailable
                    # (PSU closed public access to the archive in Aug 2026), an airport HRRR
                    # column would otherwise sit empty once carry-forward ages out — so fall
                    # back to the AWS column rather than showing nothing. Mandatory isobaric
                    # levels give only 2-3 points below 2,000 ft, so LLWS on a fallback column
                    # is the same bulk estimate the pads use; it is tagged so the frontend can
                    # say so rather than letting it pass as a BUFKIT-quality number.
                    if kind == "hrrr" and not is_pad:
                        if target[sid].get("hrrr"):
                            continue  # a real BUFKIT column arrived this run; don't clobber it
                        rows = {rk: dict(p, coarse="AWS HRRR (isobaric levels)")
                                for rk, p in rows.items() if isinstance(p, dict)}
                        logging.info(f"HRRR {sid}: no BUFKIT column, using AWS isobaric fallback "
                                     f"({len(rows)} hours).")
                    target[sid][kind] = rows
            r_hours = sum(len(k.get("rrfs", {})) for k in aws_matrix.values())
            e_hours = sum(len(k.get("refs", {})) for k in aws_matrix.values())
            h_hours = sum(len(aws_matrix[p].get("hrrr", {})) for p in LAUNCH_PADS if p in aws_matrix)
            logging.info(f"AWS columns merged (RRFS {r_hours}, REFS {e_hours} site-hours; HRRR {h_hours} pad-hours).")
        except Exception as e:
            logging.error(f"AWS RRFS/REFS/HRRR fetch failed, continuing without them: {e}")

    # Fetch the ECMWF IFS column (additive; point-extracted for pads + airports) from ECMWF
    # Open Data. Merged under the "ecmwf" key exactly like the AWS columns; total isolation via
    # try/except so any ECMWF outage or missing dependency leaves the rest of the run intact.
    if ECMWF_ENABLED:
        try:
            # ECMWF gets its OWN deadline, sized above its internal budget. The shared 300 s
            # was below the time a 49-step retrieve needs on a normal day, so the backstop
            # was firing as routine behaviour and taking the whole column with it.
            ecmwf_matrix = _run_with_deadline(
                fetch_all_ecmwf_soundings, ECMWF_DEADLINE_S, "ECMWF", default={})
            if ecmwf_matrix:
                if pad_matrix is None:
                    pad_matrix = {pid: {} for pid in LAUNCH_PADS}
                for sid, rows in ecmwf_matrix.items():
                    if not rows:
                        continue
                    is_pad = sid in LAUNCH_PADS
                    target = pad_matrix if is_pad else sounding_matrix
                    target.setdefault(sid, {})
                    target[sid]["ecmwf"] = rows
                merged = sum(len(r) for r in ecmwf_matrix.values())
                logging.info(f"ECMWF column merged ({merged} site-hours).")
        except Exception as e:
            logging.error(f"ECMWF fetch failed, continuing without it: {e}")

    generate_aviation_dashboard(STATIONS, MODELS, sounding_matrix, time_rows, pad_matrix=pad_matrix)


if __name__ == "__main__":
    run_pipeline()
