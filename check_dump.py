"""
Diagnose why a dump will not load in the dashboard.

Usage:   python check_dump.py file1.sql [file2.sql ...]
"""
import sys
import zlib

import sql_loader as L


def check(path: str) -> None:
    print("=" * 70)
    print(path)
    data = open(path, "rb").read()
    print(f"size: {len(data):,} bytes   first bytes: {data[:8]!r}")
    crlf, lone_lf = L.newline_stats(data)
    print(f"CRLF pairs: {crlf:,}   standalone LF bytes: {lone_lf:,}")
    if crlf >= 50 and lone_lf == 0:
        print("=> newline-converted (damaged) binary file; re-export it with pg_dump -Fc -f")
    if not data.startswith(b"PGDMP"):
        print("=> not a pg_dump custom-format file (does not start with PGDMP)")
        return
    try:
        buf = L._Buf(data)
        toc = L._parse_toc(buf)
    except (L._Bad, L.DumpReadError) as exc:
        print("table of contents: FAILED -", exc)
        return
    print(f"dump format {'.'.join(map(str, toc.version))}, database '{toc.dbname}', "
          f"{len(toc.entries)} TOC entries, compressed={toc.compressed}")
    tables = {e["tag"]: e for e in toc.entries if e["desc"] == "TABLE DATA"}
    needed = list(L.TABLE_COLUMNS)
    print(f"{len(tables)} tables with data; missing needed: {[t for t in needed if t not in tables] or 'none'}")
    for name in needed:
        e = tables.get(name)
        if e is None:
            continue
        try:
            text = L._read_block(buf, toc, e, e["off"] if e["flag"] == 2 else buf.pos)
            rows = text.count(b"\n") - 1 if text.strip() != b"\\." else 0
            print(f"  {name:22s} OK   ~{max(rows, 0):,} rows")
        except (L._Bad, zlib.error) as exc:
            print(f"  {name:22s} FAILED - {exc}")
    try:
        df = L.load_dashboard_df(data)
        print(f"full load: OK  {df.shape[0]:,} deliveries x {df.shape[1]} columns")
    except L.DumpReadError as exc:
        print("full load: FAILED -", exc)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        check(p)
