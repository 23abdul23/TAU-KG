#!/usr/bin/env bash
set -Eeuo pipefail

# Copy data directory contents from the running dev container to local ./data.

CONTAINER_NAME="taukg-streamlit-dev"
CONTAINER_DATA_DIR="/app/data"
LOCAL_DATA_DIR="./data"
CLEAN_DESTINATION="false"
ASSUME_YES="false"

usage() {
  cat <<'EOF'
Usage: scripts/copy_dev_container_data.sh [options]

Options:
  -c, --container NAME   Running container name or name fragment (default: taukg-streamlit-dev)
  -s, --source PATH      Source data directory inside container (default: /app/data)
  -d, --dest PATH        Local destination directory (default: ./data)
      --clean            Remove existing local destination contents before copy
  -y, --yes              Skip confirmation prompts (used with --clean)
  -h, --help             Show this help

Examples:
  scripts/copy_dev_container_data.sh
  scripts/copy_dev_container_data.sh -c streamlit-dev
  scripts/copy_dev_container_data.sh --clean -y
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--container)
      CONTAINER_NAME="$2"
      shift 2
      ;;
    -s|--source)
      CONTAINER_DATA_DIR="$2"
      shift 2
      ;;
    -d|--dest)
      LOCAL_DATA_DIR="$2"
      shift 2
      ;;
    --clean)
      CLEAN_DESTINATION="true"
      shift
      ;;
    -y|--yes)
      ASSUME_YES="true"
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

resolve_container_name() {
  local requested="$1"
  local exact_match
  exact_match="$(docker ps --format '{{.Names}}' | grep -Fx "$requested" || true)"
  if [[ -n "$exact_match" ]]; then
    echo "$exact_match"
    return
  fi

  local partial_match
  partial_match="$(docker ps --format '{{.Names}}' | grep -F "$requested" | head -n 1 || true)"
  if [[ -n "$partial_match" ]]; then
    echo "$partial_match"
    return
  fi

  echo ""
}

RESOLVED_CONTAINER="$(resolve_container_name "$CONTAINER_NAME")"
if [[ -z "$RESOLVED_CONTAINER" ]]; then
  echo "No running container found matching: $CONTAINER_NAME" >&2
  echo "Tip: run 'docker ps --format {{.Names}}' to list running containers." >&2
  exit 2
fi

if ! docker exec "$RESOLVED_CONTAINER" sh -lc "test -d '$CONTAINER_DATA_DIR'"; then
  echo "Container path not found: $CONTAINER_DATA_DIR (container: $RESOLVED_CONTAINER)" >&2
  exit 3
fi

mkdir -p "$LOCAL_DATA_DIR"

if [[ "$CLEAN_DESTINATION" == "true" ]]; then
  if [[ "$ASSUME_YES" != "true" ]]; then
    echo "Container: $RESOLVED_CONTAINER"
    echo "Source:    $CONTAINER_DATA_DIR"
    echo "Dest:      $LOCAL_DATA_DIR"
    echo "This will delete existing contents in destination before copy."
    read -r -p "Type 'yes' to continue: " reply
    if [[ "$reply" != "yes" ]]; then
      echo "Cancelled."
      exit 1
    fi
  fi
  find "$LOCAL_DATA_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
fi

docker cp "${RESOLVED_CONTAINER}:${CONTAINER_DATA_DIR}/." "$LOCAL_DATA_DIR"

echo "Copied data from ${RESOLVED_CONTAINER}:${CONTAINER_DATA_DIR} to ${LOCAL_DATA_DIR}"
