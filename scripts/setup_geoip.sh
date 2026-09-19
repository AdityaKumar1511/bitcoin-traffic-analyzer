#!/usr/bin/env bash
# ==============================================================================
# GeoIP Database Setup Script
# Downloads and installs MaxMind GeoLite2 databases for offline IP enrichment.
#
# Usage:
#   ./scripts/setup_geoip.sh                     # Interactive (prompts for key)
#   ./scripts/setup_geoip.sh --license-key YOUR_KEY  # Non-interactive
#
# MaxMind requires a free account to download GeoLite2 databases:
#   1. Sign up at https://www.maxmind.com/en/geolite2/signup
#   2. Generate a license key at https://www.maxmind.com/en/accounts/current/license-key
#   3. Run this script with your key
#
# Once downloaded, the databases are used fully offline — no internet needed
# for runtime IP lookups. This aligns with NTRO's air-gapped requirement.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GEOIP_DIR="$PROJECT_ROOT/data/geoip"
TEMP_DIR="$(mktemp -d)"

# Cleanup temp directory on exit
trap 'rm -rf "$TEMP_DIR"' EXIT

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

echo "=================================================================="
echo "    MaxMind GeoLite2 Database Setup — Bitcoin Traffic Analyzer"
echo "=================================================================="
echo ""

# Parse arguments
LICENSE_KEY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --license-key)
            LICENSE_KEY="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--license-key YOUR_MAXMIND_LICENSE_KEY]"
            echo ""
            echo "Downloads MaxMind GeoLite2-City and GeoLite2-ASN databases."
            echo "Sign up for a free key at: https://www.maxmind.com/en/geolite2/signup"
            exit 0
            ;;
        *)
            log_error "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Prompt for license key if not provided
if [ -z "$LICENSE_KEY" ]; then
    echo "MaxMind requires a free license key to download GeoLite2 databases."
    echo "Sign up at: https://www.maxmind.com/en/geolite2/signup"
    echo ""
    read -rp "Enter your MaxMind License Key: " LICENSE_KEY
    if [ -z "$LICENSE_KEY" ]; then
        log_error "No license key provided. Aborting."
        exit 1
    fi
fi

# Create GeoIP directory
mkdir -p "$GEOIP_DIR"

# Database editions to download
EDITIONS=("GeoLite2-City" "GeoLite2-ASN")
BASE_URL="https://download.maxmind.com/app/geoip_download"

for EDITION in "${EDITIONS[@]}"; do
    DOWNLOAD_URL="${BASE_URL}?edition_id=${EDITION}&license_key=${LICENSE_KEY}&suffix=tar.gz"
    ARCHIVE_FILE="$TEMP_DIR/${EDITION}.tar.gz"
    DB_FILE="${EDITION}.mmdb"

    # Check if database already exists
    if [ -f "$GEOIP_DIR/$DB_FILE" ]; then
        log_warn "$DB_FILE already exists. Re-downloading for update..."
    fi

    log_info "Downloading $EDITION database..."
    if ! curl -fsSL "$DOWNLOAD_URL" -o "$ARCHIVE_FILE" 2>/dev/null; then
        log_error "Failed to download $EDITION. Check your license key and network."
        log_error "URL: $BASE_URL?edition_id=${EDITION}&suffix=tar.gz"
        exit 1
    fi

    log_info "Extracting $EDITION..."
    tar -xzf "$ARCHIVE_FILE" -C "$TEMP_DIR"

    # Find the .mmdb file in extracted directory
    MMDB_PATH=$(find "$TEMP_DIR" -name "$DB_FILE" -type f 2>/dev/null | head -1)
    if [ -z "$MMDB_PATH" ]; then
        log_error "Could not find $DB_FILE in extracted archive."
        exit 1
    fi

    cp "$MMDB_PATH" "$GEOIP_DIR/$DB_FILE"
    log_ok "$DB_FILE installed to $GEOIP_DIR/"
done

# Update .gitignore to exclude GeoIP databases (they're large binary files)
GITIGNORE="$PROJECT_ROOT/.gitignore"
if ! grep -q "data/geoip/" "$GITIGNORE" 2>/dev/null; then
    echo "" >> "$GITIGNORE"
    echo "# MaxMind GeoIP databases (binary, ~70MB)" >> "$GITIGNORE"
    echo "data/geoip/*.mmdb" >> "$GITIGNORE"
    log_info "Added data/geoip/*.mmdb to .gitignore"
fi

echo ""
echo "=================================================================="
log_ok "GeoLite2 databases installed successfully!"
echo ""
echo "  Location:  $GEOIP_DIR/"
echo "  Files:"
for EDITION in "${EDITIONS[@]}"; do
    DB_FILE="${EDITION}.mmdb"
    if [ -f "$GEOIP_DIR/$DB_FILE" ]; then
        SIZE=$(du -h "$GEOIP_DIR/$DB_FILE" | cut -f1)
        echo "    ✓ $DB_FILE ($SIZE)"
    fi
done
echo ""
echo "  These databases are used fully offline for IP → Country/ASN"
echo "  enrichment during transaction ingestion."
echo "=================================================================="
