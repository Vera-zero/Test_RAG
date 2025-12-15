#!/usr/bin/env bash
set -euo pipefail

# Rebuild work_dir vector DB and related storage by backing up existing files
# then running examples/openai_all.py which will re-index using your local BGE embeddings.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TIMESTAMP="$(date +%Y%m%d%H%M%S)"
BACKUP_DIR="$ROOT_DIR/work_dir_backup_$TIMESTAMP"
mkdir -p "$BACKUP_DIR"

echo "Backing up and removing existing index/storage files to: $BACKUP_DIR"
FILES=(
  "work_dir/vdb_events.json"
  "work_dir/kv_store_full_docs.json"
  "work_dir/kv_store_text_chunks.json"
  "work_dir/kv_store_llm_response_cache.json"
  "work_dir/graph_event_dynamic_graph.graphml"
)

for f in "${FILES[@]}"; do
  if [ -f "$f" ]; then
    echo "  backing up: $f"
    mv "$f" "$BACKUP_DIR/"
  else
    echo "  not present: $f"
  fi
done

echo "Backup complete. Now running examples/openai_all.py to rebuild indexes."

cd "$ROOT_DIR/examples"
python openai_all.py

EXIT_CODE=$?
echo "examples/openai_all.py exited with code $EXIT_CODE"
exit $EXIT_CODE
