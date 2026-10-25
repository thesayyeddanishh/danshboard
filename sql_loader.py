"""
Load a PostgreSQL dump (custom ``pg_dump -Fc`` or plain-text format) of the Hawk-Eye
``cricketdatabase`` and turn it into the same DataFrame the dashboard gets
from a CSV export.

Only the Python standard library + pandas are used (no Postgres / pg_restore
needed).  Public entry point:

    df = load_dashboard_df(uploaded_bytes)

The DataFrame has the dashboard's REQUIRED_COLS (plus ``Country``), named
exactly like the CSV export, so every page works unchanged.
"""
from __future__ import annotations

import csv
import io
import itertools
import math
import re
import struct
import zlib
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class DumpReadError(Exception):
    """Raised with a user-friendly message when the dump cannot be read."""


# --------------------------------------------------------------------------
# Column mapping: database column -> name used by the dashboard / CSV export
# (taken from "name_in_exported_csv" in columns.csv)
# --------------------------------------------------------------------------
CRICKET_EVENT_MAP = {
    "score": "Runs",
    "swing": "Swing",
    "deviation": "Deviation",
    "bounce_x": "BounceX",
    "bounce_y": "BounceY",
    "stumps_y": "StumpsY",
    "stumps_z": "StumpsZ",
    "landing_x": "LandingX",
    "landing_y": "LandingY",
    "interception_x": "InterceptionX",
    "interception_y": "InterceptionY",
    "interception_z": "InterceptionZ",
    "crease_y": "CreaseY",
    "crease_z": "CreaseZ",
    "release_y": "ReleaseY",
    "release_z": "ReleaseZ",
    "release_speed": "ReleaseSpeed",
    "delivery_type": "DeliveryType",
    "wicket": "Wicket",
    "batsman_hand": "IsBatsmanRightHanded",  # 't' = right-handed, 'f' = left-handed
    "bowler_hand": "IsBowlerRightHanded",  # 't' = right-handed, 'f' = left-handed
}
NUMERIC_EVENT_COLS = [
    "Runs", "Swing", "Deviation", "BounceX", "BounceY", "StumpsY", "StumpsZ",
    "LandingX", "LandingY", "InterceptionX", "InterceptionY", "InterceptionZ",
    "CreaseY", "CreaseZ", "ReleaseY", "ReleaseZ", "ReleaseSpeed",
]
BOOL_EVENT_COLS = ["Wicket", "IsBatsmanRightHanded", "IsBowlerRightHanded"]

# columns we need to read from each table
TABLE_COLUMNS: Dict[str, List[str]] = {
    "cricket_event": ["id", "action_id", "batsman_lineup_id", "bowler_lineup_id"]
    + list(CRICKET_EVENT_MAP),
    "generic_action": ["id", "segment_id"],
    "generic_segment": ["id", "name", "sequence", "parent_id", "contest_id"],
    "generic_lineup": ["id", "person_id", "team_id"],
    "generic_person": ["id", "name"],
    "generic_team": ["id", "name"],
    "generic_contest": ["id", "name", "fixture_id", "surface_id"],
    "generic_fixture": ["id", "name", "competition_id"],
    "generic_competition": ["id", "name", "start_date", "entity_id"],
    "generic_entity": ["id", "name"],
    "generic_surface": ["id", "name", "venue_id"],
    "generic_venue": ["id", "name"],
}

FINAL_COLS = [
    "Innings", "Over", "Ball", "BowlerName", "BatsmanName", "BowlingTeam",
    "BattingTeam", "IsBowlerRightHanded", "IsBatsmanRightHanded", "Wicket",
    "Runs", "BounceX", "BounceY", "InterceptionX", "InterceptionY",
    "InterceptionZ", "CreaseY", "CreaseZ", "StumpsY", "StumpsZ", "ReleaseY",
    "ReleaseZ", "LandingX", "LandingY", "ReleaseSpeed", "Deviation",
    "DeliveryType", "Swing", "Ground", "Tour", "Year", "Match", "Country",
]

class _Bad(Exception):
    """Internal: a structural problem found while reading the archive."""

    def __init__(self, where: str, detail: str):
        super().__init__(f"{where}: {detail}")
        self.where, self.detail = where, detail


def newline_stats(data: bytes):
    """(CRLF pairs, standalone LF bytes). A binary dump normally has plenty of
    standalone LF bytes; a file whose newlines were converted has none."""
    crlf = data.count(b"\r\n")
    lone_lf = len(re.findall(rb"(?<!\r)\n", data))
    return crlf, lone_lf


def _explain(data: bytes, err: "_Bad", version: str = "") -> str:
    crlf, lone_lf = newline_stats(data)
    if crlf >= 50 and lone_lf == 0 and err.where.startswith("table "):
        return (
            "This dump was damaged before upload (every newline byte was converted to CRLF, "
            f"{crlf:,} places). The dashboard repaired it automatically where it could, but "
            f"{err.where} still has damage that could not be resolved: {err.detail}. "
            "If you can, export it again with `pg_dump -Fc -f file.dump ...` (not with `>`)."
        )
    if crlf >= 50 and lone_lf == 0:
        return (
            f"The file was damaged before it reached the dashboard: it contains "
            f"{crlf:,} CRLF pairs and no standalone newline bytes, so every newline "
            "byte in the binary dump was converted to CRLF (a text-mode transfer, e.g. "
            "`pg_dump ... > file` in Windows PowerShell/cmd). Re-export with "
            "`pg_dump -Fc -f file.dump ...` and upload that file."
        )
    ver = f" (dump format {version})" if version else ""
    return (
        f"The dump could not be read{ver}: {err.where} - {err.detail}. "
        "The file does not look newline-damaged, so this may be a dump variant "
        "this loader does not handle yet."
    )


# --------------------------------------------------------------------------
# Low-level reader for the custom archive format
# --------------------------------------------------------------------------
class _Buf:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.int_size = 4
        self.off_size = 8

    def byte(self) -> int:
        if self.pos >= len(self.data):
            raise _Bad("unexpected end of file", f"at byte {self.pos:,}")
        b = self.data[self.pos]
        self.pos += 1
        return b

    def read(self, n: int) -> bytes:
        if n < 0 or self.pos + n > len(self.data):
            raise _Bad("bad length field", f"{n:,} bytes requested at byte {self.pos:,}")
        out = self.data[self.pos : self.pos + n]
        self.pos += n
        return out

    def int(self) -> int:
        sign = self.byte()
        raw = self.read(self.int_size)
        v = int.from_bytes(raw, "little")
        return -v if sign else v

    def string(self) -> Optional[str]:
        n = self.int()
        if n < 0:
            return None
        return self.read(n).decode("utf-8", "replace")

    def offset(self):
        flag = self.byte()
        v = int.from_bytes(self.read(self.off_size), "little")
        return flag, v


class _Toc:
    def __init__(self):
        self.version = (0, 0, 0)
        self.compressed = True
        self.dbname = ""
        self.entries: List[dict] = []
        self.data_start = 0


def _parse_toc(buf: _Buf) -> _Toc:
    toc = _Toc()
    if buf.read(5) != b"PGDMP":
        raise DumpReadError(
            "This is not a PostgreSQL custom-format dump. Please upload the "
            "output of `pg_dump -Fc` (plain-text .sql files are not supported)."
        )
    vmaj, vmin, vrev = buf.byte(), buf.byte(), buf.byte()
    toc.version = (vmaj, vmin, vrev)
    if vmaj != 1 or vmin < 10:
        raise DumpReadError(f"Unsupported dump version {vmaj}.{vmin}.{vrev}.")
    buf.int_size = buf.byte()
    buf.off_size = buf.byte()
    fmt = buf.byte()
    if fmt != 1:
        raise DumpReadError("Only the custom dump format (pg_dump -Fc) is supported.")
    if vmin >= 15:
        algo = buf.byte()  # 0 none, 1 gzip, 2 lz4, 3 zstd
        if algo not in (0, 1):
            raise DumpReadError(
                "This dump uses lz4/zstd compression. Re-export with "
                "`pg_dump -Fc --compress=gzip`."
            )
        toc.compressed = algo == 1
    else:
        level = buf.int()
        toc.compressed = level != 0
    for _ in range(7):  # creation timestamp
        buf.int()
    toc.dbname = buf.string() or ""
    buf.string()  # server version
    buf.string()  # pg_dump version
    count = buf.int()
    if not 0 < count < 1_000_000:
        raise _Bad("table of contents", f"implausible entry count {count}")
    for _ in range(count):
        e = {}
        e["id"] = buf.int()
        buf.int()  # had dumper
        buf.string()  # table oid
        buf.string()  # oid
        e["tag"] = buf.string()
        e["desc"] = buf.string()
        buf.int()  # section
        e["defn"] = buf.string()
        buf.string()  # drop statement
        e["copy"] = buf.string()
        e["namespace"] = buf.string()
        buf.string()  # tablespace
        if vmin >= 14:
            buf.string()  # table access method
        if vmin >= 16:
            buf.int()  # relkind
        buf.string()  # owner
        buf.string()  # with oids
        while buf.string() is not None:  # dependencies
            pass
        e["flag"], e["off"] = buf.offset()
        toc.entries.append(e)
    toc.data_start = buf.pos
    return toc


def _read_block(buf: _Buf, toc: _Toc, entry: dict, pos: int) -> bytes:
    where = f"table {entry['tag']}"
    buf.pos = pos
    kind = buf.byte()
    if kind != 1:  # BLK_DATA
        raise _Bad(where, f"no data block at offset {pos:,} (found type {kind})")
    found = buf.int()
    if found != entry["id"]:
        raise _Bad(where, f"data block id {found} does not match table of contents id {entry['id']}")
    d = zlib.decompressobj() if toc.compressed else None
    parts = []
    while True:
        n = buf.int()
        if n == 0:
            break
        chunk = buf.read(n)
        if d is not None:
            try:
                parts.append(d.decompress(chunk))
            except zlib.error as exc:
                raise _Bad(where, f"compressed data is corrupt ({exc})") from exc
        else:
            parts.append(chunk)
    if d is not None and not d.eof:
        raise _Bad(where, "compressed data ends unexpectedly")
    return b"".join(parts)


def _table_blocks(buf: _Buf, toc: _Toc, wanted: List[str]) -> Dict[str, bytes]:
    """Return {table_name: COPY text bytes} for the wanted tables."""
    data_entries = [e for e in toc.entries if e["desc"] == "TABLE DATA"]
    out: Dict[str, bytes] = {}
    sequential_pos = toc.data_start
    for e in data_entries:
        if e["flag"] == 2:  # offset known (seekable dump)
            pos = e["off"]
        else:  # blocks follow the TOC in order
            pos = sequential_pos
        if e["tag"] in wanted:
            out[e["tag"]] = _read_block(buf, toc, e, pos)
            sequential_pos = buf.pos
        elif e["flag"] != 2:
            _read_block(buf, toc, e, pos)  # must skip to find the next one
            sequential_pos = buf.pos
    missing = [t for t in wanted if t not in out]
    if missing:
        raise DumpReadError(
            "The dump is missing required table(s): " + ", ".join(missing)
        )
    return out


# --------------------------------------------------------------------------
# Automatic repair of newline-damaged dumps
#
# Some tools write the binary dump in text mode: every 0x0A byte that is not
# already preceded by 0x0D becomes 0x0D 0x0A.  Reading it back, a "\r\n" pair
# is therefore EITHER an inserted CR (the original byte was a lone LF - the
# overwhelmingly common case) OR a genuine CR LF in the original (about 1 in
# 65,000 byte pairs of compressed data).  The archive itself tells us which:
#   * every data chunk has a known length, and the next chunk header must
#     appear exactly where that many ORIGINAL bytes end  -> how many pairs in a
#     chunk are genuine;
#   * only the correct placement decompresses into COPY rows with the right
#     number of tab-separated columns;
#   * zlib's Adler-32 checksum confirms the whole stream at the end.
# --------------------------------------------------------------------------
_AMB = re.compile(rb"(?<!\r)\r\n")
_CTRL = re.compile(rb"[\x00-\x08\x0b-\x1f]")
_MAX_KEEPS = 6
_MAX_COMBOS = 300_000
_MAX_HYPOTHESES = 32


def is_newline_damaged(data: bytes) -> bool:
    if data[:5] != b"PGDMP":
        return False
    crlf, lone_lf = newline_stats(data)
    return crlf >= 50 and lone_lf == 0


class _Hyp:
    __slots__ = ("pos", "d", "carry", "parts", "ended")

    def __init__(self, pos, d, carry, parts, ended=False):
        self.pos, self.d, self.carry, self.parts, self.ended = pos, d, carry, parts, ended


class _Repairer:
    def __init__(self, data: bytes):
        self.C = data
        self._progress = 0.0
        self.amb = np.fromiter((m.start() for m in _AMB.finditer(data)), dtype=np.int64)

    # -- locating a block --------------------------------------------------
    def _locate(self, entry: dict) -> List[int]:
        off = entry["off"]
        c = off
        for _ in range(100):
            nc = off + int(np.searchsorted(self.amb, c, side="left"))
            if nc == c:
                break
            c = nc
        slack = 3 * (c // 65536) + 120
        lo, hi = max(0, c - slack), c + 16
        idb = (b"\x01" if entry["id"] < 0 else b"\x00") + abs(entry["id"]).to_bytes(4, "little")
        variants = {idb}
        for i, b in enumerate(idb):  # a 0x0A inside the id shows up as CR LF
            if b == 0x0A:
                variants.add(idb[:i] + b"\r\n" + idb[i + 1 :])
        found = []
        for v in variants:
            pat = b"\x01" + v
            i = self.C.find(pat, lo, hi + len(pat))
            while i >= 0:
                found.append((i, len(pat)))
                i = self.C.find(pat, i + 1, hi + len(pat))
        return sorted(found)

    # -- helpers -------------------------------------------------------------
    def _hdr(self, q: int):
        """Chunk header (sign + 4-byte int), treating CR LF as an inserted CR."""
        C = self.C
        out = bytearray()
        i = q
        for _ in range(5):
            if i >= len(C):
                return None
            if C[i : i + 2] == b"\r\n":
                out.append(0x0A)
                i += 2
            else:
                out.append(C[i])
                i += 1
        if out[0] != 0:
            return None
        return int.from_bytes(out[1:5], "little"), i

    def _next_ok(self, q: int, n_cur: int) -> bool:
        h = self._hdr(q)
        return h is not None and 0 <= h[0] <= n_cur

    def _candidates(self, p: int, n: int):
        win = self.C[p : p + n + n // 8 + 64]
        amb = [m.start() for m in _AMB.finditer(win)]
        L = len(amb)
        res = []
        for m in range(L + 1):
            for k in range(min(m, _MAX_KEEPS) + 1):
                e = n + m - k
                if e > len(win):
                    continue
                if m > 0 and amb[m - 1] + 2 > e:
                    continue
                if m < L and e > amb[m]:
                    continue
                if self._next_ok(p + e, n):
                    res.append((k, e, m))
        res.sort()
        return win, amb, res

    @staticmethod
    def _placements(win, amb, e, m, k):
        if k == 0:
            yield _AMB.sub(b"\n", win[:e])
            return
        for keep in itertools.combinations(range(m), k):
            ks = set(keep)
            parts, cpos = [], 0
            for i in range(m):
                a = amb[i]
                parts.append(win[cpos:a])
                parts.append(b"\r\n" if i in ks else b"\n")
                cpos = a + 2
            parts.append(win[cpos:e])
            yield b"".join(parts)

    @staticmethod
    def _advance(h: _Hyp, o: bytes, ntab: int, new_pos: int):
        d = h.d.copy()
        try:
            out = d.decompress(o)
        except zlib.error:
            return None
        if _CTRL.search(out):
            return None
        buf = h.carry + out
        nl = buf.rfind(b"\n")
        ended = h.ended
        if nl >= 0:
            if not ended:
                for ln in buf[:nl].split(b"\n"):
                    if ln == b"\\.":
                        ended = True
                        break
                    if ln.count(b"\t") != ntab:
                        return None
            carry = buf[nl + 1 :]
        else:
            carry = buf
        if not ended and carry.count(b"\t") > ntab:
            return None
        return _Hyp(new_pos, d, carry, (h.parts, out), ended)

    # -- one block -----------------------------------------------------------
    def read_block(self, entry: dict, ntab: int) -> bytes:
        where = f"table {entry['tag']}"
        starts = self._locate(entry)
        if not starts:
            raise _Bad(where, "could not locate its data block")
        last_err = "no valid reconstruction"
        for start, patlen in starts:
            try:
                return self._read_from(start + patlen, ntab, where)
            except _Bad as exc:
                last_err = exc.detail
        raise _Bad(where, last_err)

    def _read_from(self, q: int, ntab: int, where: str) -> bytes:
        h0 = self._hdr(q)
        if h0 is None:
            raise _Bad(where, "bad chunk header")
        hyps = [_Hyp(q, zlib.decompressobj(), b"", None)]
        finished: List[_Hyp] = []
        block_end = len(self.C)
        self._progress = 0.0
        while hyps:
            self._progress = max(self._progress, (hyps[0].pos - q) / max(1, block_end - q))
            nxt: List[_Hyp] = []
            for h in hyps:
                hd = self._hdr(h.pos)
                if hd is None:
                    continue
                n, p = hd
                if n == 0:
                    finished.append(h)
                    continue
                win, amb, cands = self._candidates(p, n)
                combos = 0
                for k, e, m in cands:
                    combos += math.comb(m, k)
                    if combos > _MAX_COMBOS:
                        break
                    for o in self._placements(win, amb, e, m, k):
                        nh = self._advance(h, o, ntab, p + e)
                        if nh is not None:
                            nxt.append(nh)
                            if len(nxt) >= _MAX_HYPOTHESES:
                                break
                    if len(nxt) >= _MAX_HYPOTHESES:
                        break
            hyps = nxt
            if len(hyps) > 1:  # drop duplicates that ended at the same place with the same text tail
                seen, uniq = set(), []
                for h in hyps:
                    key = (h.pos, h.carry[-64:], id(h.parts))
                    if key not in seen:
                        seen.add(key)
                        uniq.append(h)
                hyps = uniq
        for h in finished:
            if h.d.eof:
                chunks, node = [], h.parts
                while node is not None:
                    node, out = node
                    chunks.append(out)
                return b"".join(reversed(chunks))
        raise _Bad(
            where,
            f"repair stopped after {self._progress:.0%} of this table's data "
            "(unexplained byte-level damage)",
        )

    def blocks(self, toc: "_Toc", wanted: List[str]) -> Dict[str, bytes]:
        entries = {e["tag"]: e for e in toc.entries if e["desc"] == "TABLE DATA"}
        out = {}
        for name in wanted:
            e = entries.get(name)
            if e is None:
                raise DumpReadError("The dump is missing required table(s): " + name)
            if e["flag"] != 2:
                raise _Bad(f"table {name}", "dump has no block offsets (not seekable)")
            ntab = len(_copy_columns(e)) - 1
            out[name] = self.read_block(e, ntab)
        return out


# --------------------------------------------------------------------------
# Plain-text dumps (pgAdmin "Plain" format / pg_dump -Fp)
# Tables are stored as COPY ... FROM stdin; blocks of tab-separated rows.
# Line endings may have been converted to CRLF - harmless here, because a real
# CR inside a value is always written as the two characters \r in COPY text.
# --------------------------------------------------------------------------
_COPY_RE = re.compile(
    rb'^COPY\s+((?:"?[\w$]+"?\.)?"?[\w$]+"?)\s*\(([^)]*)\)\s+FROM\s+stdin;[ \t]*\r?\n',
    re.M | re.I,
)


def _read_plain(data: bytes) -> Dict[str, pd.DataFrame]:
    text = data.replace(b"\r\n", b"\n")
    found: Dict[str, tuple] = {}
    for m in _COPY_RE.finditer(text):
        name = m.group(1).split(b".")[-1].strip(b'"').decode("utf-8", "replace")
        cols = [c.strip().strip(b'"').decode("utf-8", "replace") for c in m.group(2).split(b",")]
        end = text.find(b"\n\\.\n", m.end() - 1)
        if end < 0:
            end = len(text)
        body = text[m.end() : end + 1] if end >= m.end() else b""
        found[name] = (cols, body)
    if not found:
        if re.search(rb"^INSERT\s+INTO", text, re.M | re.I):
            raise DumpReadError(
                "This SQL file stores its data as INSERT statements, which are not supported. "
                "In pgAdmin's Backup dialog leave 'Use INSERT commands' switched off."
            )
        raise DumpReadError(
            "No table data found. Upload a pgAdmin backup in 'Custom' or 'Plain' format "
            "(plain files must contain COPY ... FROM stdin; sections)."
        )
    missing = [t for t in TABLE_COLUMNS if t not in found]
    if missing:
        raise DumpReadError("The dump is missing required table(s): " + ", ".join(missing))
    out: Dict[str, pd.DataFrame] = {}
    for name, wanted in TABLE_COLUMNS.items():
        cols, body = found[name]
        try:
            out[name] = _copy_to_df(body, cols, wanted)
        except DumpReadError:
            raise
        except Exception as exc:
            raise DumpReadError(
                f"Could not parse the data of table {name}: {type(exc).__name__}: {exc}"
            ) from exc
    return out


# --------------------------------------------------------------------------
# COPY text -> DataFrame
# --------------------------------------------------------------------------
_ESC = {"b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t", "v": "\v", "\\": "\\"}


def _unescape(s):
    if not isinstance(s, str) or "\\" not in s:
        return s

    def sub(m):
        t = m.group(1)
        if t[0] in _ESC:
            return _ESC[t[0]]
        if t[0] == "x":
            return chr(int(t[1:], 16))
        if t[0] in "01234567":
            return chr(int(t, 8))
        return t

    return re.sub(r"\\(x[0-9A-Fa-f]{1,2}|[0-7]{1,3}|.)", sub, s)


def _copy_columns(entry: dict) -> List[str]:
    m = re.search(r"\((.*?)\)", entry.get("copy") or "", re.S)
    if not m:
        raise DumpReadError(f"Could not read column list of {entry['tag']}.")
    return [c.strip().strip('"') for c in m.group(1).split(",")]


def _copy_to_df(text: bytes, all_cols: List[str], wanted: List[str]) -> pd.DataFrame:
    # strip the "\." end marker
    if text.startswith(b"\\.\n") or text.strip() == b"\\.":
        text = b""
    else:
        idx = text.find(b"\n\\.\n")
        if idx >= 0:
            text = text[: idx + 1]
    absent = [c for c in wanted if c not in all_cols]
    if absent:
        raise DumpReadError("Missing column(s) in dump: " + ", ".join(absent))
    if not text.strip():
        return pd.DataFrame(columns=wanted)
    usecols = [all_cols.index(c) for c in wanted]
    df = pd.read_csv(
        io.BytesIO(text),
        sep="\t",
        header=None,
        names=all_cols,
        usecols=usecols,
        dtype=str,
        na_values=["\\N"],
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
        engine="c",
    )
    df = df[wanted]
    for c in df.columns:
        if df[c].str.contains("\\\\", regex=True, na=False).any():
            df[c] = df[c].map(_unescape)
    return df


def _to_bool(series: pd.Series):
    mapped = series.map({"t": True, "f": False})
    return mapped.astype(bool) if not mapped.isna().any() else mapped


def _num(series: pd.Series):
    return pd.to_numeric(series, errors="coerce")


# --------------------------------------------------------------------------
# Joins -> dashboard DataFrame
# --------------------------------------------------------------------------
def _ordinal(seg: pd.DataFrame, ids: pd.Series) -> pd.Series:
    return ids.map(seg["sequence"])


def build_dashboard_df(t: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    ev = t["cricket_event"].copy()
    if ev.empty:
        raise DumpReadError("The dump has no rows in cricket_event.")

    # ---- Innings / Over / Ball from the generic_segment hierarchy -------
    seg = t["generic_segment"].copy()
    seg["sequence"] = _num(seg["sequence"])
    seg = seg.set_index("id")
    act = t["generic_action"].set_index("id")

    ev["segment_id"] = ev["action_id"].map(act["segment_id"])
    ev["Ball"] = ev["segment_id"].map(seg["sequence"])
    ev["_over_id"] = ev["segment_id"].map(seg["parent_id"])
    ev["Over"] = ev["_over_id"].map(seg["sequence"])
    ev["_inn_id"] = ev["_over_id"].map(seg["parent_id"])
    ev["Innings"] = ev["_inn_id"].map(seg["sequence"])
    ev["_contest_id"] = ev["segment_id"].map(seg["contest_id"])

    ev = ev.dropna(subset=["Innings", "Over", "Ball"]).copy()
    if ev.empty:
        raise DumpReadError(
            "Could not link deliveries to innings/overs/balls "
            "(cricket_event -> generic_action -> generic_segment)."
        )
    # The CSV export is 1-based (Innings 1-2, Over 1-20, Ball 1-n).
    for c in ("Innings", "Over", "Ball"):
        if ev[c].min() == 0:
            ev[c] = ev[c] + 1
        ev[c] = ev[c].astype(int)

    # ---- Players and teams via the lineup rows --------------------------
    lu = t["generic_lineup"].set_index("id")
    person = t["generic_person"].set_index("id")["name"]
    team = t["generic_team"].set_index("id")["name"]

    bat_lu, bowl_lu = ev["batsman_lineup_id"], ev["bowler_lineup_id"]
    ev["BatsmanName"] = bat_lu.map(lu["person_id"]).map(person)
    ev["BowlerName"] = bowl_lu.map(lu["person_id"]).map(person)
    ev["BattingTeam"] = bat_lu.map(lu["team_id"]).map(team)
    ev["BowlingTeam"] = bowl_lu.map(lu["team_id"]).map(team)

    # ---- Match / Ground / Tour / Year / Country --------------------------
    contest = t["generic_contest"].set_index("id")
    fixture = t["generic_fixture"].set_index("id")
    comp = t["generic_competition"].set_index("id")
    surface = t["generic_surface"].set_index("id")
    venue = t["generic_venue"].set_index("id")["name"]
    entity = t["generic_entity"].set_index("id")["name"]

    cid = ev["_contest_id"]
    ev["Match"] = cid.map(contest["name"])
    fix_id = cid.map(contest["fixture_id"])
    surf_id = cid.map(contest["surface_id"])

    ground = surf_id.map(surface["name"])
    ev["Ground"] = ground.fillna(surf_id.map(surface["venue_id"]).map(venue))

    comp_id = fix_id.map(fixture["competition_id"])
    tour = fix_id.map(fixture["name"])
    ev["Tour"] = tour.fillna(comp_id.map(comp["name"]))
    start = comp_id.map(comp["start_date"])
    ev["Year"] = pd.to_numeric(start.str[:4], errors="coerce")
    ev["Country"] = comp_id.map(comp["entity_id"]).map(entity)

    # ---- rename + type the cricket_event columns ------------------------
    ev = ev.rename(columns=CRICKET_EVENT_MAP)
    for c in NUMERIC_EVENT_COLS:
        ev[c] = _num(ev[c])
    for c in BOOL_EVENT_COLS:
        ev[c] = _to_bool(ev[c])
    if ev["Year"].notna().all():
        ev["Year"] = ev["Year"].astype(int)

    ev["_event_id"] = _num(ev["id"])
    ev = ev.sort_values(["_contest_id", "Innings", "Over", "Ball", "_event_id"])
    return ev[FINAL_COLS].reset_index(drop=True)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def read_tables(data: bytes) -> Dict[str, pd.DataFrame]:
    """Parse the dump and return the needed tables as DataFrames of strings."""
    version = ""
    if data[:5] != b"PGDMP":
        return _read_plain(data)
    try:
        if is_newline_damaged(data):
            # parse the table of contents from a simple CR LF -> LF copy, then
            # rebuild each needed data block exactly (see _Repairer)
            buf = _Buf(_AMB.sub(b"\n", data))
            toc = _parse_toc(buf)
            version = ".".join(str(v) for v in toc.version)
            blocks = _Repairer(data).blocks(toc, list(TABLE_COLUMNS))
        else:
            buf = _Buf(data)
            toc = _parse_toc(buf)
            version = ".".join(str(v) for v in toc.version)
            blocks = _table_blocks(buf, toc, list(TABLE_COLUMNS))
    except _Bad as err:
        raise DumpReadError(_explain(data, err, version)) from err
    except (IndexError, ValueError, struct.error) as exc:
        raise DumpReadError(
            _explain(data, _Bad("archive structure", repr(exc)), version)
        ) from exc

    entries = {e["tag"]: e for e in toc.entries if e["desc"] == "TABLE DATA"}
    out: Dict[str, pd.DataFrame] = {}
    for name, cols in TABLE_COLUMNS.items():
        try:
            out[name] = _copy_to_df(blocks[name], _copy_columns(entries[name]), cols)
        except DumpReadError:
            raise
        except Exception as exc:  # pandas parsing problems etc.
            raise DumpReadError(
                f"Could not parse the data of table {name} (dump format {version}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    return out


def load_dashboard_df(data: bytes) -> pd.DataFrame:
    """Dump bytes -> DataFrame with the same columns as the CSV export."""
    df = build_dashboard_df(read_tables(data))
    df.attrs["sql_repaired"] = is_newline_damaged(data)
    return df
