# Phase 5 Atom 5.1: Data Quality Report
import pandas as pd
from pathlib import Path

CSV = 'data/eurusd_unified.csv'

def main():
    if not Path(CSV).exists():
        print('[FAIL] ' + CSV + ' not found')
        return

    df = pd.read_csv(CSV, index_col=0, parse_dates=True)

    print('=' * 60)
    print('  EUR/USD DATA QUALITY REPORT')
    print('=' * 60)

    # 1. Basic
    print('\n[1] BASIC')
    print('    Rows: ' + str(len(df)) + '  Columns: ' + str(len(df.columns)))
    dt0 = df.index.min()
    dt1 = df.index.max()
    days = (dt1 - dt0).days
    print('    From: ' + str(dt0))
    print('    To:   ' + str(dt1))
    print('    Span: ' + str(days) + ' days (~' + str(round(days / 365.25, 1)) + ' years)')

    # 2. Timezone
    print('\n[2] TIMEZONE')
    tz = df.index.tz
    if tz is None:
        print('    [WARN] No timezone in file')
        df.index = pd.to_datetime(df.index, utc=True)
        print('    Forced UTC for analysis')
    else:
        print('    ' + str(tz))

    # 3. Gaps
    print('\n[3] GAPS')
    diffs = df.index.to_series().diff().dropna()
    h1 = pd.Timedelta(hours=1)
    normal = (diffs == h1).sum()
    weekend = (diffs > pd.Timedelta(hours=48)).sum()
    mid_gaps = diffs[(diffs > h1) & (diffs <= pd.Timedelta(hours=48))]
    print('    H1 intervals: ' + str(normal) + '/' + str(len(diffs)))
    print('    Weekend gaps (>48h): ' + str(weekend))
    if len(mid_gaps) > 0:
        print('    [WARN] Mid-size gaps (2-48h): ' + str(len(mid_gaps)))
        for t in mid_gaps.index[:5]:
            hrs = mid_gaps[t].total_seconds() / 3600
            print('      Gap ' + str(round(hrs, 1)) + 'h at ' + str(t))
    else:
        print('    No suspicious gaps. PASS')

    # 4. Duplicates
    print('\n[4] DUPLICATES')
    dup = df.index.duplicated().sum()
    if dup > 0:
        print('    [WARN] ' + str(dup) + ' duplicate timestamps')
    else:
        print('    None. PASS')

    # 5. Nulls
    print('\n[5] MISSING VALUES')
    nulls = df.isnull().sum().sum()
    if nulls > 0:
        print('    [WARN] ' + str(nulls) + ' null values:')
        for c in df.columns:
            n = df[c].isnull().sum()
            if n > 0:
                print('      ' + c + ': ' + str(n))
    else:
        print('    None. PASS')

    # 6. OHLC integrity
    print('\n[6] OHLC INTEGRITY')
    bad = 0
    hl = (df['high'] < df['low']).sum()
    cl = (df['close'] < df['low']).sum()
    ch = (df['close'] > df['high']).sum()
    ol = (df['open'] < df['low']).sum()
    oh = (df['open'] > df['high']).sum()
    flat = ((df['open'] == df['high']) & (df['high'] == df['low']) & (df['low'] == df['close'])).sum()
    if hl > 0: print('    [FAIL] High < Low: ' + str(hl)); bad += hl
    if cl > 0: print('    [WARN] Close < Low: ' + str(cl)); bad += cl
    if ch > 0: print('    [WARN] Close > High: ' + str(ch)); bad += ch
    if ol > 0: print('    [WARN] Open < Low: ' + str(ol)); bad += ol
    if oh > 0: print('    [WARN] Open > High: ' + str(oh)); bad += oh
    if flat > 0: print('    [INFO] Flat candles (O=H=L=C): ' + str(flat))
    if bad == 0: print('    No anomalies. PASS')

    # 7. Price stats
    print('\n[7] PRICE STATISTICS')
    rng = df['high'] - df['low']
    pips = rng * 10000
    print('    Close range: ' + '{:.5f}'.format(df['close'].min()) + ' to ' + '{:.5f}'.format(df['close'].max()))
    print('    Mean range: ' + '{:.1f}'.format(pips.mean()) + ' pips')
    print('    Median range: ' + '{:.1f}'.format(pips.median()) + ' pips')
    print('    Max range: ' + '{:.1f}'.format(pips.max()) + ' pips')
    print('    Min range: ' + '{:.1f}'.format(pips.min()) + ' pips')
    atr14 = rng.rolling(14).mean() * 10000
    print('    ATR(14) mean: ' + '{:.1f}'.format(atr14.mean()) + ' pips')

    # 8. Hour distribution
    print('\n[8] HOURLY DISTRIBUTION')
    hc = df.index.hour.value_counts().sort_index()
    missing = [h for h in range(24) if h not in hc.index]
    if missing:
        print('    [WARN] Missing hours: ' + str(missing))
    else:
        print('    All 24 hours present. PASS')
    print('    Min/hour: ' + str(hc.min()) + ' (h' + str(hc.idxmin()) + ')')
    print('    Max/hour: ' + str(hc.max()) + ' (h' + str(hc.idxmax()) + ')')

    # Summary
    issues = []
    if dup > 0: issues.append('duplicates')
    if nulls > 0: issues.append('nulls')
    if bad > 0: issues.append('ohlc_anomalies')
    if len(mid_gaps) > 0: issues.append('gaps')
    if tz is None: issues.append('no_timezone')
    if missing: issues.append('missing_hours')

    print('\n' + '=' * 60)
    if issues:
        print('  RESULT: ISSUES -> ' + ', '.join(issues))
    else:
        print('  RESULT: ALL CHECKS PASSED')
    print('=' * 60)

main()