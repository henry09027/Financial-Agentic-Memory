#!/usr/bin/env python3
"""Step 0: build the company universe shared by both downloaders.

Both data streams (SEC EDGAR facts and Yahoo broker actions) read the SAME
universe file, so every company is identified by both a 10-digit CIK and a
listed ticker, and benchmark splits can be company-disjoint across streams.

    python build_universe.py                                   # current S&P 500 from Wikipedia
    python build_universe.py --source notebook --user-agent "Name email@example.com"
    python build_universe.py --source file --input companies.csv [--user-agent ...]

Sources
  wikipedia  Current S&P 500 constituents table (has Symbol and CIK). Current
             membership at retrieval time, NOT historical membership.
  notebook   The fixed 500-CIK list from the original research notebook (used by
             the earlier EDGAR download). Tickers are resolved with SEC's
             company_tickers.json, so --user-agent is required.
  file       .csv with ticker and/or cik columns (optional name, sector), .json
             array of such objects or a {ticker: cik} map, or .txt with one
             ticker per line (# comments allowed). Missing tickers/CIKs are
             resolved through SEC, which requires --user-agent.

Output: artifacts/raw/universe.json (fam-envelope-v1, kind=universe). One record per
listing: ticker (BRK.B), yahoo_symbol (BRK-B), cik, name, sector, primary_listing.
When several listings share a CIK (GOOG/GOOGL), exactly one is primary: the
shortest symbol, then alphabetical. EDGAR uses primary listings; Yahoo uses all.
Exit codes: 0 ok, 1 error, 130 interrupted.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import re
import time

from pipeline_common import (DEFAULT_PATHS, PipelineError, add_common_arguments, check_output, exclusive_lock,
                             get_logger, normalize_cik, normalize_ticker, producer, read_envelope, require,
                             run_main, setup_logging, utc_now, write_envelope, yahoo_symbol)

LOG = get_logger("build_universe")
WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
# Fixed universe copied from the research notebook; membership date was not recorded.
NOTEBOOK_CIKS = '''
66740 91142 1800 1551152 1467373 796343 2488 874761 4977 1090872 2969 1559720 1086222 915913
1035443 1097149 1579241 352541 899051 1652044 764180 1018724 1748790 1002910 4904 4962 5272 1053507
1410636 820027 1037868 318154 820313 6281 315293 1841666 1858681 320193 6951 1751008 1521332 947484
7084 1176948 1596532 354190 1267238 732717 731802 769397 8670 866787 8818 1069183 1701605 9389
70858 10456 10795 1067983 764478 842023 875045 2012383 1393818 1512673 1664703 1390777 12927 1075531
885725 14272 1730168 1383312 79282 14693 1996862 1037540 1043277 813672 906345 927628 721371 815097
1783180 1690820 726958 18230 1374310 1138118 1402057 1140859 1071739 1130310 1324404 1100682 316709 1091667
93410 1058090 896159 313927 936395 1739940 20286 723254 858877 831001 759944 21076 1156375 811156
21344 1058290 820318 1679788 21665 1166691 1035983 1163165 1047862 16918 1868275 711404 900075 24741
1175454 1057352 909832 849395 1535527 1051470 277948 26172 64803 313616 940944 1561550 927066 910521
315189 1571996 27904 1090012 1093557 1539838 1297996 29534 935703 715957 1286681 1792789 29905 1751788
882184 936340 1326160 1666700 1551182 1065088 1415404 31462 827052 1099800 1156039 105634 32604 65984
821189 33213 33185 1101239 922621 920522 1001250 1095073 1711269 1474432 72741 1109357 895126 1324424
746515 1289490 2115436 1048695 1013237 814547 815556 34903 1048911 2082247 2011641 1136893 35527 1274494
1031296 798354 866374 37996 1262039 1659166 1754301 38777 831259 1121788 749251 40545 1932393 1996810
849399 1474735 40533 40704 1467858 40987 882095 1123360 320335 1609711 886982 45012 874766 46080
860730 765880 1000228 47111 1645590 1585689 354950 2089271 773840 48465 1070750 4281 47217 48898
49071 49196 1501585 51143 832101 874716 49826 1110803 879169 1699150 1145197 50863 1381197 1571949
51253 51434 896878 1035267 914208 1687229 1478242 1020569 728535 898293 779152 52988 200406 833444
19617 1944048 1418135 91576 1601046 55785 879101 1506307 1404912 319201 1637459 56873 202058 920148
707549 1300514 1336920 920760 1069202 59478 1707925 1335258 936468 60086 60667 1397187 1633978 1489393
36270 1510295 1048286 62709 916076 1835632 62996 1141391 63754 63908 927653 1613103 310158 1326801
1099219 1037646 789570 827054 723125 789019 912595 1682852 1103982 1280452 865752 1059556 895421 1285785
68505 1408198 1120193 1002047 1065280 1164727 1564708 753308 320187 1111711 72331 702165 73124 1133421
1513761 1013871 73309 1045810 906163 1413447 898173 797468 878927 29989 1097864 1039684 1341439 1781335
75362 75677 1321655 1327567 76334 723531 1633917 77360 77476 78003 1004980 1413329 1534701 764622
713676 79879 922224 1126328 80424 80661 1045609 1137774 857005 788784 1393311 822416 2058873 804328
1050915 1022079 1037038 720005 1713445 726728 910606 872589 1281761 1060391 943819 31791 1783879 1024478
84839 882835 745732 884887 101829 64040 1108524 2023554 1034054 87347 1137789 1032208 1373715 89800
1063761 2041610 4127 91419 2005951 91440 1964738 92122 92380 93556 829224 93751 1022671 1757898
310764 1375365 1601712 883241 96021 1283699 1113169 946581 1116132 1389170 27419 1385157 1094285 97210
1318605 97476 1811074 217346 97745 109198 1973266 916365 1466258 1260221 86312 864749 92230 1447669
860731 100493 36104 1543151 74208 1403568 100885 100517 1090727 1067701 731766 352915 1035002 1393052
740260 1967680 1014473 1442145 732712 875320 1674101 1792044 1705696 1403161 1692819 906107 1396009 2128626
11544 277135 943452 104169 1744489 823768 1000697 783325 72971 766704 105770 106040 106535 719955
107263 1140536 1327811 1174922 72903 1524472 1041061 877212 1136869 1555280
'''.split()


def http_get(url: str, user_agent: str, attempts: int = 4) -> str:
    import requests
    last = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, timeout=(10, 60), headers={"User-Agent": user_agent})
            if response.status_code == 200:
                return response.text
            last = f"HTTP {response.status_code}"
            if response.status_code in (401, 403):
                break
        except requests.RequestException as error:
            last = str(error)
        if attempt + 1 < attempts:
            time.sleep(min(60.0, 2.0 ** (attempt + 1)))
    raise PipelineError(f"Could not fetch {url}: {last}")


def sec_directory(user_agent: str | None, text: str | None = None) -> tuple[dict, dict]:
    """Returns (yahoo_symbol -> (cik, title), cik -> first (ticker, title))."""
    if text is None:
        require(user_agent, "Resolving tickers/CIKs through SEC requires --user-agent 'Name email@example.com'")
        text = http_get(SEC_TICKERS_URL, user_agent)
    by_symbol, by_cik = {}, {}
    rows = json.loads(text)
    for key in sorted(rows, key=int):  # file order is SEC's ranking; first listing per CIK is the common stock
        row = rows[key]
        cik, ticker = normalize_cik(row["cik_str"]), normalize_ticker(row["ticker"])
        by_symbol.setdefault(yahoo_symbol(ticker), (cik, row.get("title")))
        by_cik.setdefault(cik, (ticker.replace("-", "."), row.get("title")))
    return by_symbol, by_cik


def from_wikipedia(html: str) -> list[dict]:
    import pandas as pd
    tables = pd.read_html(io.StringIO(html))
    matches = [table for table in tables if {"Symbol", "Security", "CIK"} <= set(map(str, table.columns))]
    require(len(matches) == 1, "Could not uniquely identify the S&P 500 constituents table; use --source file")
    rows = [{"ticker": row["Symbol"], "cik": row["CIK"], "name": row["Security"], "sector": row.get("GICS Sector")}
            for row in matches[0].to_dict("records")]
    require(450 <= len(rows) <= 550, f"Unexpected constituent count {len(rows)}; verify the page or use --source file")
    return rows


def from_file(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".csv":
        reader = csv.DictReader(io.StringIO(text))
        columns = {name.lower().strip(): name for name in reader.fieldnames or []}
        require("ticker" in columns or "cik" in columns, "CSV needs a ticker and/or cik column")
        return [{key: (row.get(columns[key]) or None) if key in columns else None for key in ("ticker", "cik", "name", "sector")}
                for row in reader]
    if path.suffix.lower() == ".json":
        value = json.loads(text)
        if isinstance(value, dict):
            return [{"ticker": ticker, "cik": cik, "name": None, "sector": None} for ticker, cik in value.items()]
        require(isinstance(value, list), "JSON universe must be an array or a {ticker: cik} map")
        return [{"ticker": item.get("ticker"), "cik": item.get("cik"), "name": item.get("name"), "sector": item.get("sector")}
                if isinstance(item, dict) else {"ticker": None, "cik": item, "name": None, "sector": None} for item in value]
    tickers = [line.split("#", 1)[0].strip() for line in text.splitlines()]
    return [{"ticker": ticker, "cik": None, "name": None, "sector": None} for ticker in tickers if ticker]


def resolve(rows: list[dict], directory: tuple[dict, dict] | None) -> list[dict]:
    resolved, unresolved = [], []
    for row in rows:
        ticker = normalize_ticker(row["ticker"]) if row.get("ticker") else None
        cik = normalize_cik(row["cik"]) if row.get("cik") not in (None, "") else None
        name, sector = row.get("name"), row.get("sector")
        if ticker is None or cik is None:
            require(directory is not None, "Some rows lack a ticker or CIK; pass --user-agent to resolve them through SEC")
            by_symbol, by_cik = directory
            if cik is None and ticker is not None and yahoo_symbol(ticker) in by_symbol:
                cik, title = by_symbol[yahoo_symbol(ticker)]
                name = name or title
            elif ticker is None and cik is not None and cik in by_cik:
                ticker, title = by_cik[cik]
                name = name or title
        if ticker is None or cik is None:
            unresolved.append(row.get("ticker") or row.get("cik"))
            continue
        resolved.append({"ticker": ticker, "yahoo_symbol": yahoo_symbol(ticker), "cik": cik,
                         "name": name.strip() if isinstance(name, str) and name.strip() else None,
                         "sector": str(sector).strip() if isinstance(sector, str) and sector.strip() else None})
    require(not unresolved, f"Could not resolve {len(unresolved)} entries: {unresolved[:20]}")
    return finalize(resolved)


def finalize(records: list[dict]) -> list[dict]:
    by_ticker = {}
    for record in records:
        previous = by_ticker.setdefault(record["ticker"], record)
        require(previous["cik"] == record["cik"], f"Ticker {record['ticker']} maps to several CIKs")
    groups: dict[str, list[dict]] = {}
    for record in by_ticker.values():
        groups.setdefault(record["cik"], []).append(record)
    for listings in groups.values():
        primary = min(listings, key=lambda item: (len(item["yahoo_symbol"]), item["yahoo_symbol"]))
        for item in listings:
            item["primary_listing"] = item is primary
    require(len({record["yahoo_symbol"] for record in by_ticker.values()}) == len(by_ticker), "Yahoo symbol collision")
    return sorted(by_ticker.values(), key=lambda item: item["ticker"])


def self_test() -> None:
    html = ("<table><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>CIK</th></tr>"
            + "".join(f"<tr><td>T{i}</td><td>Co {i}</td><td>Tech</td><td>{1000 + i}</td></tr>" for i in range(498))
            + "<tr><td>BRK.B</td><td>Berkshire</td><td>Financials</td><td>1067983</td></tr>"
            + "<tr><td>GOOGL</td><td>Alphabet A</td><td>Tech</td><td>1652044</td></tr>"
            + "<tr><td>GOOG</td><td>Alphabet C</td><td>Tech</td><td>1652044</td></tr></table>")
    records = resolve(from_wikipedia(html), None)
    by_ticker = {record["ticker"]: record for record in records}
    require(len(records) == 501 and by_ticker["BRK.B"]["yahoo_symbol"] == "BRK-B", "Normalization failed")
    require(by_ticker["GOOG"]["primary_listing"] and not by_ticker["GOOGL"]["primary_listing"], "Primary listing rule failed")
    sec = json.dumps({"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
                      "1": {"cik_str": 1067983, "ticker": "BRK-B", "title": "Berkshire"},
                      "2": {"cik_str": 1067983, "ticker": "BRK-A", "title": "Berkshire"}})
    directory = sec_directory(None, sec)
    records = resolve([{"ticker": "AAPL", "cik": None}, {"ticker": None, "cik": "1067983"}], directory)
    require([record["ticker"] for record in records] == ["AAPL", "BRK.B"], "SEC resolution failed")
    try:
        resolve([{"ticker": "MISSING", "cik": None}], directory)
    except PipelineError:
        pass
    else:
        raise PipelineError("Unresolved ticker was accepted")
    require(len(set(NOTEBOOK_CIKS)) == 500, "Embedded notebook universe must have 500 unique CIKs")
    LOG.info("Universe self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["wikipedia", "notebook", "file"], default="wikipedia")
    parser.add_argument("--input", type=Path, help="Universe file for --source file")
    parser.add_argument("--output", type=Path, default=DEFAULT_PATHS["universe"])
    parser.add_argument("--user-agent", help="SEC contact, 'Name email@example.com' (env SEC_USER_AGENT)")
    add_common_arguments(parser)
    args = parser.parse_args()
    setup_logging(args.log_level)
    if args.self_test:
        self_test()
        return 0
    import os
    user_agent = args.user_agent or os.environ.get("SEC_USER_AGENT")
    if user_agent and not re.search(r"\S+@\S+\.\S+", user_agent):
        parser.error("--user-agent must contain a contact email")
    if (args.source == "file") != (args.input is not None):
        parser.error("--input is required with, and only with, --source file")
    with exclusive_lock(args.output):
        check_output(args.output, args.overwrite)
        if args.source == "wikipedia":
            rows = from_wikipedia(http_get(WIKIPEDIA_URL, "Mozilla/5.0 (compatible; FAM-universe/2.0)"))
            records = resolve(rows, None)
            source = {"source": "wikipedia", "url": WIKIPEDIA_URL,
                      "membership": "current S&P 500 at retrieval time; not historical membership"}
        elif args.source == "notebook":
            records = resolve([{"ticker": None, "cik": cik} for cik in NOTEBOOK_CIKS], sec_directory(user_agent))
            source = {"source": "notebook", "membership": "fixed research-notebook CIK list; membership date unknown",
                      "ticker_resolution": SEC_TICKERS_URL}
        else:
            rows = from_file(args.input)
            needs_sec = any(not row.get("ticker") or not row.get("cik") for row in rows)
            records = resolve(rows, sec_directory(user_agent) if needs_sec else None)
            source = {"source": "file", "path": str(args.input), "membership": "user supplied",
                      "ticker_resolution": SEC_TICKERS_URL if needs_sec else None}
        metadata = dict(source, retrieved_at=utc_now(), producer=producer(__file__),
                        listings=len(records), companies=len({record["cik"] for record in records}),
                        primary_listing_rule="shortest Yahoo symbol, then alphabetical, among listings sharing a CIK")
        write_envelope(args.output, "universe", metadata, records)
        read_envelope(args.output, "universe")
    LOG.info("Saved %s: %d listings, %d companies", args.output, len(records), metadata["companies"])
    return 0


if __name__ == "__main__":
    run_main(main, LOG)
