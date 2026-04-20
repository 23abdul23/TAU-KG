#!/usr/bin/env bash
set -Eeuo pipefail

# Reset persisted development data inside a running TAU-KG dev container.
# This clears directory contents, not mount points.

CONTAINER_NAME="taukg-streamlit-dev"
APP_ROOT="/app"
ASSUME_YES="false"
DRY_RUN="false"

usage() {
  cat <<'EOF'
Usage: scripts/reset_dev_container_data.sh [options]

Options:
  -c, --container NAME   Container name (default: taukg-streamlit-dev)
  -y, --yes              Skip confirmation prompt
  -n, --dry-run          Show what would be removed without deleting
  -h, --help             Show this help

This script clears contents of:
  /app/chroma_db
  /app/data
inside the selected container.

It does not remove mount directories themselves.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--container)
      CONTAINER_NAME="$2"
      shift 2
      ;;
    -y|--yes)
      ASSUME_YES="true"
      shift
      ;;
    -n|--dry-run)
      DRY_RUN="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! docker ps --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
  echo "Container is not running: $CONTAINER_NAME" >&2
  echo "Start it first, for example: docker compose -f compose.dev.yml up -d" >&2
  exit 2
fi

TARGETS=("$APP_ROOT/chroma_db" "$APP_ROOT/data")

if [[ "$ASSUME_YES" != "true" ]]; then
  echo "Container: $CONTAINER_NAME"
  echo "This will clear contents of:"
  printf '  - %s\n' "${TARGETS[@]}"
  read -r -p "Type 'yes' to continue: " reply
  if [[ "$reply" != "yes" ]]; then
    echo "Cancelled."
    exit 1
  fi
fi

if [[ "$DRY_RUN" == "true" ]]; then
  docker exec "$CONTAINER_NAME" sh -lc '
    set -eu
    for d in /app/chroma_db /app/data; do
      echo "--- $d"
      if [ -d "$d" ]; then
        find "$d" -mindepth 1 -maxdepth 1 -print
      else
        echo "(missing)"
      fi
    done
  '
  exit 0
fi

docker exec "$CONTAINER_NAME" sh -lc '
  set -eu
  for d in /app/chroma_db /app/data; do
    if [ -d "$d" ]; then
      # Clear all entries, including dotfiles, but keep directory itself.
      find "$d" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
    fi
  done
'

echo "Reset complete for container: $CONTAINER_NAME"
