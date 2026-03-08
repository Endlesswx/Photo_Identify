from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from photo_identify.storage import Storage


@dataclass(slots=True)
class PersonSource:
    db_path: str
    person_id: str
    name: str
    display_name: str
    face_count: int
    path: str
    bbox: str
    cover_image_id: int | None
    cover_face_id: int | None
    sort_order: int
    is_deleted: int


@dataclass(slots=True)
class CombinedPersonModel:
    person_id: str
    name: str
    display_name: str
    face_count: int = 0
    is_pinned: bool = False
    path: str = ""
    bbox: str = "[]"
    cover_image_id: int | None = None
    cover_face_id: int | None = None
    source_dbs: list[str] = field(default_factory=list)
    sources: list[PersonSource] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.person_id,
            "name": self.name,
            "display_name": self.display_name,
            "face_count": self.face_count,
            "is_pinned": self.is_pinned,
            "path": self.path,
            "bbox": self.bbox,
            "cover_image_id": self.cover_image_id,
            "cover_face_id": self.cover_face_id,
            "source_dbs": list(self.source_dbs),
            "sources": [
                {
                    "db_path": source.db_path,
                    "person_id": source.person_id,
                    "name": source.name,
                    "display_name": source.display_name,
                    "face_count": source.face_count,
                    "path": source.path,
                    "bbox": source.bbox,
                    "cover_image_id": source.cover_image_id,
                    "cover_face_id": source.cover_face_id,
                    "sort_order": source.sort_order,
                    "is_deleted": source.is_deleted,
                }
                for source in self.sources
            ],
        }


@dataclass(slots=True)
class SimilarPersonCandidate:
    similarity: float
    left_id: str
    left_name: str
    left_display_name: str
    left_db_path: str
    right_id: str
    right_name: str
    right_display_name: str
    right_db_path: str

    def to_dict(self) -> dict:
        return {
            "similarity": self.similarity,
            "left_id": self.left_id,
            "left_name": self.left_name,
            "left_display_name": self.left_display_name,
            "left_db_path": self.left_db_path,
            "right_id": self.right_id,
            "right_name": self.right_name,
            "right_display_name": self.right_display_name,
            "right_db_path": self.right_db_path,
        }


def normalize_db_paths(db_paths: Iterable[str]) -> list[str]:
    resolved_paths: list[str] = []
    seen: set[str] = set()
    for db_path in db_paths:
        normalized = str(db_path or "").strip()
        if not normalized:
            continue
        resolved = str(Path(normalized).resolve())
        if resolved in seen:
            continue
        if not Path(resolved).exists():
            continue
        seen.add(resolved)
        resolved_paths.append(resolved)
    return resolved_paths


def load_combined_persons(db_paths: Iterable[str], include_deleted: bool = False) -> list[dict]:
    combined: dict[str, CombinedPersonModel] = {}
    all_sources: list[PersonSource] = []

    for db_path in normalize_db_paths(db_paths):
        storage = Storage(db_path)
        try:
            for row in storage.get_all_persons(include_deleted=include_deleted):
                source = PersonSource(
                    db_path=db_path,
                    person_id=str(row["id"]),
                    name=str(row.get("name") or "未命名人物"),
                    display_name=str(row.get("name") or "未命名人物"),
                    face_count=int(row.get("face_count") or 0),
                    path=str(row.get("path") or ""),
                    bbox=str(row.get("bbox") or "[]"),
                    cover_image_id=row.get("cover_image_id"),
                    cover_face_id=row.get("cover_face_id"),
                    sort_order=int(row.get("sort_order") or 0),
                    is_deleted=int(row.get("is_deleted") or 0),
                )
                all_sources.append(source)

                item = combined.get(source.person_id)
                if item is None:
                    item = CombinedPersonModel(
                        person_id=source.person_id,
                        name=source.name,
                        display_name=source.display_name,
                        face_count=source.face_count,
                        is_pinned=int(source.sort_order or 0) < 0,
                        path=source.path,
                        bbox=source.bbox,
                        cover_image_id=source.cover_image_id,
                        cover_face_id=source.cover_face_id,
                        source_dbs=[db_path],
                        sources=[source],
                    )
                    combined[source.person_id] = item
                else:
                    item.face_count += source.face_count
                    item.sources.append(source)
                    if db_path not in item.source_dbs:
                        item.source_dbs.append(db_path)
                    if (not item.path) and source.path:
                        item.path = source.path
                        item.bbox = source.bbox
                        item.cover_image_id = source.cover_image_id
                        item.cover_face_id = source.cover_face_id
        finally:
            storage.close()

    for item in combined.values():
        item.display_name = item.name
        item.is_pinned = any(int(source.sort_order or 0) < 0 for source in item.sources)

    return [
        item.to_dict()
        for item in sorted(
            combined.values(),
            key=lambda row: (
                0 if row.is_pinned else 1,
                -row.face_count,
                row.display_name.lower(),
                row.person_id,
            ),
        )
    ]


def load_deleted_persons(db_paths: Iterable[str]) -> list[dict]:
    deleted_items: list[dict] = []
    for db_path in normalize_db_paths(db_paths):
        storage = Storage(db_path)
        try:
            for row in storage.get_deleted_persons():
                name = str(row.get("name") or "未命名人物")
                deleted_items.append(
                    {
                        **row,
                        "id": str(row["id"]),
                        "display_name": f"{name} ({Path(db_path).name})",
                        "db_path": db_path,
                    }
                )
        finally:
            storage.close()
    deleted_items.sort(key=lambda row: (-int(row.get("face_count") or 0), row.get("display_name") or ""))
    return deleted_items


def _load_combined_person_vectors(db_paths: Iterable[str]) -> list[dict]:
    combined: dict[str, dict] = {}
    for db_path in normalize_db_paths(db_paths):
        storage = Storage(db_path)
        try:
            for row in storage.get_person_feature_vectors(include_deleted=False):
                vector = row.get("embedding_vector")
                if vector is None:
                    continue
                person_id = str(row["id"])
                item = combined.setdefault(
                    person_id,
                    {
                        "id": person_id,
                        "name": str(row.get("name") or "未命名人物"),
                        "path": str(row.get("path") or ""),
                        "bbox": str(row.get("bbox") or "[]"),
                        "face_count": 0,
                        "_vectors": [],
                    },
                )
                item["face_count"] += int(row.get("face_count") or 0)
                if (not item["path"]) and row.get("path"):
                    item["path"] = str(row.get("path") or "")
                    item["bbox"] = str(row.get("bbox") or "[]")
                item["_vectors"].append(np.asarray(vector, dtype=np.float32))
        finally:
            storage.close()

    results: list[dict] = []
    for item in combined.values():
        vectors = item.pop("_vectors", [])
        if not vectors:
            continue
        mean_vector = np.mean(np.stack(vectors), axis=0)
        mean_norm = float(np.linalg.norm(mean_vector))
        if mean_norm <= 1e-8:
            continue
        item["embedding_vector"] = mean_vector / mean_norm
        results.append(item)
    return results



def build_merge_candidates_for_target(db_paths: Iterable[str], target_person_id: str) -> list[dict]:
    target_id = str(target_person_id or "").strip()
    if not target_id:
        raise ValueError("目标人物 ID 不能为空")

    persons = _load_combined_person_vectors(db_paths)
    target_person = next((item for item in persons if str(item["id"]) == target_id), None)
    if target_person is None:
        return []

    target_vector = target_person.get("embedding_vector")
    if target_vector is None:
        return []

    candidates: list[dict] = []
    for item in persons:
        person_id = str(item["id"])
        if person_id == target_id:
            continue
        candidate_vector = item.get("embedding_vector")
        if candidate_vector is None:
            continue
        candidates.append(
            {
                "id": person_id,
                "name": item.get("name") or "未命名人物",
                "path": item.get("path") or "",
                "bbox": item.get("bbox") or "[]",
                "face_count": int(item.get("face_count") or 0),
                "similarity": float(np.dot(target_vector, candidate_vector)),
            }
        )

    candidates.sort(
        key=lambda item: (
            -float(item.get("similarity") or 0.0),
            -int(item.get("face_count") or 0),
            str(item.get("name") or "").lower(),
            str(item.get("id") or ""),
        )
    )
    return candidates



def build_similarity_candidates(db_paths: Iterable[str], threshold: float = 0.9) -> list[dict]:
    vectors: list[dict] = []
    for db_path in normalize_db_paths(db_paths):
        storage = Storage(db_path)
        try:
            for row in storage.get_person_feature_vectors(include_deleted=False):
                vectors.append(
                    {
                        **row,
                        "db_path": db_path,
                        "display_name": f"{row['name']} ({Path(db_path).name})",
                    }
                )
        finally:
            storage.close()

    candidates: list[SimilarPersonCandidate] = []
    for index, left in enumerate(vectors):
        left_vector = left.get("embedding_vector")
        if left_vector is None:
            continue
        for right in vectors[index + 1 :]:
            right_vector = right.get("embedding_vector")
            if right_vector is None:
                continue
            if left["id"] == right["id"]:
                continue
            similarity = float(np.dot(left_vector, right_vector))
            if similarity <= threshold:
                continue
            candidates.append(
                SimilarPersonCandidate(
                    similarity=similarity,
                    left_id=str(left["id"]),
                    left_name=str(left["name"]),
                    left_display_name=str(left["display_name"]),
                    left_db_path=str(left["db_path"]),
                    right_id=str(right["id"]),
                    right_name=str(right["name"]),
                    right_display_name=str(right["display_name"]),
                    right_db_path=str(right["db_path"]),
                )
            )

    candidates.sort(key=lambda item: (-item.similarity, item.left_display_name, item.right_display_name))
    return [item.to_dict() for item in candidates]
