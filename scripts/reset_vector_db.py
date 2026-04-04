"""
Reset the local Chroma vector database used by TAU-KG.

This script deletes the main collections used by the app and then removes the
persisted Chroma directory so the next app start recreates a clean database.
"""

from __future__ import annotations

import argparse
import gc
import shutil
import time
from pathlib import Path

import chromadb
from chromadb.config import Settings


DEFAULT_DB_PATH = "./chroma_db"
DEFAULT_COLLECTIONS = ("gene_proteins", "papers", "paper_entities")


def _delete_known_collections(client: chromadb.PersistentClient, collection_names: tuple[str, ...]) -> list[str]:
    """Delete known collections if they exist and return the deleted names."""
    deleted = []
    for name in collection_names:
        try:
            client.delete_collection(name=name)
            deleted.append(name)
        except Exception:
            continue
    return deleted


def _best_effort_directory_remove(target_path: Path, retries: int = 3, delay_seconds: float = 0.5) -> tuple[bool, str]:
    """Try to remove the persisted Chroma directory, handling Windows file locks cleanly."""
    last_error = ""
    for attempt in range(retries + 1):
        try:
            if target_path.exists():
                shutil.rmtree(target_path, ignore_errors=False)
            return (not target_path.exists(), "")
        except PermissionError as exc:
            last_error = str(exc)
            if attempt >= retries:
                break
            time.sleep(delay_seconds * (attempt + 1))
        except FileNotFoundError:
            return (True, "")
    return (False, last_error)


def reset_vector_db(db_path: str, collection_names: tuple[str, ...]) -> dict[str, object]:
    """Delete Chroma collections and remove the on-disk persistence directory."""
    target_path = Path(db_path).resolve()
    deleted_collections: list[str] = []
    reset_used = False
    reset_error = ""
    directory_remove_error = ""

    if target_path.exists():
        client = None
        try:
            client = chromadb.PersistentClient(
                path=str(target_path),
                settings=Settings(allow_reset=True),
            )
            try:
                client.reset()
                reset_used = True
                deleted_collections = list(collection_names)
            except Exception as exc:
                reset_error = str(exc)
                deleted_collections = _delete_known_collections(client, collection_names)
        except Exception as exc:
            reset_error = str(exc)
        finally:
            client = None
            gc.collect()
            time.sleep(0.25)

        directory_removed, directory_remove_error = _best_effort_directory_remove(target_path)
    else:
        directory_removed = True

    return {
        "db_path": str(target_path),
        "deleted_collections": deleted_collections,
        "directory_removed": directory_removed,
        "reset_used": reset_used,
        "reset_error": reset_error,
        "directory_remove_error": directory_remove_error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reset the local Chroma vector database.")
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help=f"Path to the Chroma persistence directory. Default: {DEFAULT_DB_PATH}",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_path = Path(args.db_path).resolve()

    if not args.yes:
        confirmation = input(
            f"This will permanently delete the vector DB at '{target_path}'. Type 'yes' to continue: "
        ).strip().lower()
        if confirmation != "yes":
            print("Reset cancelled.")
            return 1

    result = reset_vector_db(args.db_path, DEFAULT_COLLECTIONS)
    print(f"Vector DB path: {result['db_path']}")
    print(f"Reset API used: {result['reset_used']}")
    print(f"Deleted collections: {', '.join(result['deleted_collections']) or 'none found'}")
    print(f"Persistence directory removed: {result['directory_removed']}")
    if result["reset_error"]:
        print(f"Reset note: {result['reset_error']}")
    if result["directory_remove_error"]:
        print("Directory removal note: the DB file is still locked by another process.")
        print(result["directory_remove_error"])
        print("Close Streamlit, Python shells, or any running app using Chroma, then rerun the script.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
