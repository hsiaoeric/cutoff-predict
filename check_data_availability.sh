#!/usr/bin/env bash
#
# TMU 志願錄取權重 — Data Availability Scanner (High Performance)
# ================================================================
# Scans 民國100–115 (semesters 1 & 2) for available cutoff weight data.
#
# REQUIREMENTS:
#   - Active TMU session cookies (update COOKIES below)
#   - Get fresh cookies from browser DevTools after logging in
#
# Usage:
#   chmod +x check_data_availability.sh && ./check_data_availability.sh
#

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
CONCURRENCY=5
START_YEAR=100
END_YEAR=115
SEMESTERS=(1 2)

BASE_URL="https://newacademic.tmu.edu.tw/Application/TKE/TKE20/TKE2030_02.aspx?Language="

# ──── UPDATE THESE COOKIES WITH YOUR ACTIVE SESSION ────
COOKIES='_ga_N8XN48BCER=GS2.3.s1769502302$o2$g1$t1769502304$j58$l0$h0; _ga_F783CES2HW=GS2.1.s1769759386$o1$g0$t1769759387$j59$l0$h0; _ga_L97GC65PLY=GS2.1.s1769759393$o1$g0$t1769759398$j55$l0$h0; _ga_M4L8RV5BT7=GS2.1.s1770003168$o2$g1$t1770003179$j49$l0$h0; _ga_L75BJGL5RY=GS2.1.s1770280985$o7$g0$t1770280985$j60$l0$h0; ASP.NET_SessionId=imgbhpvlft010u3ew35nspqb; framepage_func=hideTimeOut(); _ga=GA1.3.1978173950.1768355872; _gid=GA1.3.1311773854.1770981604; GCLB=CN-z2tHaz6vDowEQAw; _ga_P2WSG5V5BG=GS2.3.s1770996007$o2$g0$t1770996007$j60$l0$h0; .ASPXAUTH=F5B02B989AB45058781190D30126A0CEB3031E1EC6D38BB136D2690F65D298DE69EE3448630247EFE9ACADEC79C6C5AAC3F9071F18E739FC386B487828627E8B255F6F781CAE393C35ADBCDB612D0F31FC61F5D3'
# ───────────────────────────────────────────────────────

RESULTS_DIR="/Users/hsiaoeric/projects/cutoff_predict/availability_results"
RESULTS_FILE="${RESULTS_DIR}/data_availability.csv"
SUMMARY_FILE="${RESULTS_DIR}/summary.txt"
RAW_DIR="${RESULTS_DIR}/raw"

# ─── Setup ────────────────────────────────────────────────────────────────────
mkdir -p "${RAW_DIR}"

echo "╔══════════════════════════════════════════════════════════════╗" >&2
echo "║  TMU 志願錄取權重 — Data Availability Scanner               ║" >&2
echo "║  Scanning 民國${START_YEAR}–${END_YEAR}, semesters 1 & 2              ║" >&2
echo "║  Concurrency: ${CONCURRENCY} parallel requests                       ║" >&2
echo "╚══════════════════════════════════════════════════════════════╝" >&2
echo "" >&2

# ─── Step 1: Get fresh viewstate from the page ────────────────────────────────
echo "→ Step 1: Fetching initial page to extract ASP.NET tokens..." >&2

INIT_PAGE="${RAW_DIR}/_init.html"
curl -s -o "${INIT_PAGE}" \
    --max-time 30 \
    -b "${COOKIES}" \
    "${BASE_URL}" \
    -H 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36'

# Check if we got redirected to login
if grep -q 'logout\|登入\|login\|逾時' "${INIT_PAGE}" 2>/dev/null; then
    echo "❌ ERROR: Session expired or invalid cookies." >&2
    echo "" >&2
    echo "   To update cookies:" >&2
    echo "   1. Log into https://newacademic.tmu.edu.tw" >&2
    echo "   2. Navigate to 志願錄取權重查詢 (TKE2030)" >&2
    echo "   3. Open DevTools (F12) → Network tab" >&2
    echo "   4. Click 查詢 and copy the Cookie header from the request" >&2
    echo "   5. Paste into the COOKIES variable in this script" >&2
    exit 1
fi

# Extract __VIEWSTATE
VIEWSTATE=$(sed -n 's/.*id="__VIEWSTATE"[^"]*value="\([^"]*\)".*/\1/p' "${INIT_PAGE}" | head -1)
if [[ -z "${VIEWSTATE}" ]]; then
    VIEWSTATE=$(sed -n 's/.*name="__VIEWSTATE"[^"]*value="\([^"]*\)".*/\1/p' "${INIT_PAGE}" | head -1)
fi

# Extract __EVENTVALIDATION
EVENTVALIDATION=$(sed -n 's/.*id="__EVENTVALIDATION"[^"]*value="\([^"]*\)".*/\1/p' "${INIT_PAGE}" | head -1)
if [[ -z "${EVENTVALIDATION}" ]]; then
    EVENTVALIDATION=$(sed -n 's/.*name="__EVENTVALIDATION"[^"]*value="\([^"]*\)".*/\1/p' "${INIT_PAGE}" | head -1)
fi

VIEWSTATEGENERATOR=$(sed -n 's/.*id="__VIEWSTATEGENERATOR"[^"]*value="\([^"]*\)".*/\1/p' "${INIT_PAGE}" | head -1)
VIEWSTATEGENERATOR="${VIEWSTATEGENERATOR:-828EEBBB}"

if [[ -z "${VIEWSTATE}" ]] || [[ -z "${EVENTVALIDATION}" ]]; then
    echo "❌ ERROR: Could not extract ASP.NET tokens." >&2
    echo "   Response preview:" >&2
    head -5 "${INIT_PAGE}" >&2
    exit 1
fi

echo "  ✅ Session valid. Tokens extracted." >&2
echo "     __VIEWSTATE length:       ${#VIEWSTATE}" >&2
echo "     __EVENTVALIDATION length: ${#EVENTVALIDATION}" >&2
echo "" >&2

# URL-encode tokens
VS_ENCODED=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.stdin.read().strip(), safe=''))" <<< "${VIEWSTATE}")
EV_ENCODED=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.stdin.read().strip(), safe=''))" <<< "${EVENTVALIDATION}")

# ─── Write CSV header ────────────────────────────────────────────────────────
echo "year_roc,semester,ayearsms,status,num_courses,sample_courses" > "${RESULTS_FILE}"

# ─── Create a temporary helper script for xargs ──────────────────────────────
# (xargs on macOS can't easily call exported bash functions with all args,
#  so we write a small helper script)
HELPER="${RESULTS_DIR}/_fetch_one.sh"
cat > "${HELPER}" << 'HELPER_EOF'
#!/usr/bin/env bash
set -uo pipefail

ayearsms="$1"
vs_encoded="$2"
ev_encoded="$3"
vsg="$4"
base_url="$5"
raw_dir="$6"
cookies="$7"

year_roc="${ayearsms%?}"
sem="${ayearsms: -1}"
raw_file="${raw_dir}/${ayearsms}.html"
year_ad=$((year_roc + 1911))

# Build POST body
post_data="ScriptManager1=AjaxPanel%7CQUERY_BTN1"
post_data+="&__EVENTTARGET=&__EVENTARGUMENT=&__LASTFOCUS="
post_data+="&__VIEWSTATE=${vs_encoded}"
post_data+="&__VIEWSTATEGENERATOR=${vsg}"
post_data+="&__VIEWSTATEENCRYPTED="
post_data+="&__EVENTVALIDATION=${ev_encoded}"
post_data+="&ActivePageControl=&ColumnFilter=&M_PKNO="
post_data+="&Mode=ADD&ROWSTAMP=&LOTFACULTY=&LOTGROUP="
post_data+="&M_ASYS_CODE=&M_DEGREE_CODE=&M_COLLEGE_CODE="
post_data+="&M_TEACH_GROUP_CODE=&M_TEACH_GRP=&H_CRD=&Language="
post_data+="&Q_AYEARSMS=${ayearsms}"
post_data+="&Q_COSID=&Q_CH_LESSON="
post_data+="&__ASYNCPOST=true"
post_data+="&QUERY_BTN1=%E6%9F%A5%E8%A9%A2"

# Execute curl
http_code=$(curl -s -o "${raw_file}" -w "%{http_code}" \
    --max-time 30 \
    --retry 2 \
    --retry-delay 3 \
    -b "${cookies}" \
    "${base_url}" \
    -H 'accept: */*' \
    -H 'content-type: application/x-www-form-urlencoded; charset=UTF-8' \
    -H 'origin: https://newacademic.tmu.edu.tw' \
    -H "referer: ${base_url}" \
    -H 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36' \
    -H 'x-microsoftajax: Delta=true' \
    -H 'x-requested-with: XMLHttpRequest' \
    --data-raw "${post_data}" \
    2>/dev/null || echo "000")

# Parse
status="NO_DATA"
num_courses=0
sample_courses=""

if [[ "${http_code}" == "200" ]] && [[ -f "${raw_file}" ]]; then
    if grep -q 'logout' "${raw_file}" 2>/dev/null || grep -q '逾時' "${raw_file}" 2>/dev/null; then
        status="SESSION_EXPIRED"
    else
        # Count data rows — macOS grep compatible (tail -1 ensures single value)
        num_courses=$(grep -c 'class="td[GW]' "${raw_file}" 2>/dev/null | tail -1 || echo 0)
        num_courses=$(echo "${num_courses}" | tr -d '[:space:]')
        [[ -z "${num_courses}" ]] && num_courses=0

        if [[ "${num_courses}" -gt 0 ]]; then
            status="HAS_DATA"
            sample_courses=$(grep -o '<td>[^<]*</td>' "${raw_file}" 2>/dev/null \
                | sed 's/<[^>]*>//g' \
                | awk 'NR % 10 == 5' \
                | head -3 \
                | tr '\n' '|' \
                | sed 's/|$//')
        else
            if grep -q 'DataGrid' "${raw_file}" 2>/dev/null; then
                status="NO_DATA"
            else
                status="ERROR_OR_EMPTY"
            fi
        fi
    fi
elif [[ "${http_code}" == "000" ]]; then
    status="TIMEOUT"
elif [[ "${http_code}" == "500" ]]; then
    status="SERVER_ERROR"
else
    status="HTTP_${http_code}"
fi

# CSV output to stdout
echo "${year_roc},${sem},${ayearsms},${status},${num_courses},\"${sample_courses}\""

# Progress to stderr
icon="❌"
[[ "${status}" == "HAS_DATA" ]] && icon="✅"
[[ "${status}" == "TIMEOUT" ]] && icon="⏱️ "
[[ "${status}" == "SERVER_ERROR" || "${status}" == "SESSION_EXPIRED" ]] && icon="💥"
printf "  %s [%3s] 民國%3s年 第%s學期 (%s) → %-16s (%s courses)\n" \
    "${icon}" "${http_code}" "${year_roc}" "${sem}" "${year_ad}" "${status}" "${num_courses}" >&2

# Small delay to be polite
sleep 0.3
HELPER_EOF
chmod +x "${HELPER}"

# ─── Generate semester codes and run ──────────────────────────────────────────
semester_codes=()
for year in $(seq "${START_YEAR}" "${END_YEAR}"); do
    for sem in "${SEMESTERS[@]}"; do
        semester_codes+=("${year}${sem}")
    done
done

total=${#semester_codes[@]}
echo "→ Step 2: Scanning ${total} semesters (${CONCURRENCY} parallel)..." >&2
echo "" >&2

START_TIME=$(date +%s)

printf '%s\n' "${semester_codes[@]}" \
    | xargs -P "${CONCURRENCY}" -I {} \
        "${HELPER}" {} "${VS_ENCODED}" "${EV_ENCODED}" \
        "${VIEWSTATEGENERATOR}" "${BASE_URL}" "${RAW_DIR}" "${COOKIES}" \
    >> "${RESULTS_FILE}"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# ─── Sort CSV ─────────────────────────────────────────────────────────────────
# Filter out any malformed lines (no comma = bad data)
{
    head -1 "${RESULTS_FILE}"
    tail -n +2 "${RESULTS_FILE}" | grep ',' | sort -t',' -k3 -n
} > "${RESULTS_FILE}.tmp" && mv "${RESULTS_FILE}.tmp" "${RESULTS_FILE}"

# Cleanup helper
rm -f "${HELPER}"

# ─── Generate Summary ────────────────────────────────────────────────────────
echo "" >&2

has_data_count=$(grep -c 'HAS_DATA' "${RESULTS_FILE}" 2>/dev/null | tail -1 || echo 0)
no_data_count=$(grep -c 'NO_DATA\|NO_TABLE\|ERROR_OR_EMPTY' "${RESULTS_FILE}" 2>/dev/null | tail -1 || echo 0)
error_count=$(grep -c 'TIMEOUT\|SERVER_ERROR\|HTTP_\|SESSION_EXPIRED' "${RESULTS_FILE}" 2>/dev/null | tail -1 || echo 0)

{
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  TMU 志願錄取權重 — Data Availability Summary"
    echo "  Scan completed: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Duration: ${ELAPSED}s (${total} semesters)"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    echo "  Total semesters scanned:  ${total}"
    echo "  ✅ With data:             ${has_data_count}"
    echo "  ❌ No data:               ${no_data_count}"
    echo "  ⚠️  Errors/Timeouts:       ${error_count}"
    echo ""
    echo "───────────────────────────────────────────────────────────"
    echo "  Semesters WITH data:"
    echo "───────────────────────────────────────────────────────────"
    grep 'HAS_DATA' "${RESULTS_FILE}" | while IFS=',' read -r yr sem code status ncourses samples; do
        yr_ad=$((yr + 1911))
        printf "  民國%-4s 第%s學期 (%s)  │ %4s courses\n" "${yr}" "${sem}" "${yr_ad}" "${ncourses}"
    done
    echo ""
    echo "───────────────────────────────────────────────────────────"
    echo "  Semesters WITHOUT data or errors:"
    echo "───────────────────────────────────────────────────────────"
    grep -v 'HAS_DATA\|year_roc' "${RESULTS_FILE}" | while IFS=',' read -r yr sem code status ncourses samples; do
        yr_ad=$((yr + 1911))
        printf "  民國%-4s 第%s學期 (%s)  │ %s\n" "${yr}" "${sem}" "${yr_ad}" "${status}"
    done
    echo ""
    echo "───────────────────────────────────────────────────────────"
    echo "  Data Year Range:"
    echo "───────────────────────────────────────────────────────────"
    first_data=$(grep 'HAS_DATA' "${RESULTS_FILE}" | head -1 | cut -d',' -f3)
    last_data=$(grep 'HAS_DATA' "${RESULTS_FILE}" | tail -1 | cut -d',' -f3)
    if [[ -n "${first_data:-}" ]] && [[ -n "${last_data:-}" ]]; then
        f_yr="${first_data%?}"; f_sem="${first_data: -1}"
        l_yr="${last_data%?}"; l_sem="${last_data: -1}"
        echo "  Earliest: 民國${f_yr}年 第${f_sem}學期 ($((f_yr + 1911)))"
        echo "  Latest:   民國${l_yr}年 第${l_sem}學期 ($((l_yr + 1911)))"
    else
        echo "  No data found in any semester."
    fi
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  CSV output:  ${RESULTS_FILE}"
    echo "  Raw HTML:    ${RAW_DIR}/"
    echo "═══════════════════════════════════════════════════════════"
} | tee "${SUMMARY_FILE}"

echo "" >&2
echo "✅ Done! Results saved to ${RESULTS_FILE}" >&2
echo "📄 Summary saved to ${SUMMARY_FILE}" >&2
