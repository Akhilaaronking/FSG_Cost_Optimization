from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json


def _canonical_json(
    value,
) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    )


def _mutable_design_view(
    bom: dict,
    search_space: dict | None = None,
) -> dict:
    if search_space is None:
        return deepcopy(bom)

    parts_by_id = {
        part["part_id"]: part
        for part in bom.get("parts", [])
    }

    design_parts = []

    for part_space in search_space.get(
        "parts",
        [],
    ):
        part_id = part_space["part_id"]
        part = parts_by_id[part_id]
        item = {
            "part_id": part_id,
        }

        if "material_choices" in part_space:
            item["material_id"] = part.get(
                "material_id"
            )

        if "process_choices" in part_space:
            item["process_id"] = part.get(
                "process_id"
            )

        if "geometry_variables" in part_space:
            geometry = part.get(
                "geometry",
                {},
            )
            item["geometry"] = {
                field_name: geometry.get(
                    field_name
                )
                for field_name in part_space[
                    "geometry_variables"
                ]
            }

        if "fastener_choices" in part_space:
            item["fasteners"] = deepcopy(
                part.get("fasteners", [])
            )

        design_parts.append(item)

    return {
        "parts": design_parts,
    }


def design_fingerprint(
    bom: dict,
    search_space: dict | None = None,
) -> str:
    payload = _mutable_design_view(
        bom,
        search_space=search_space,
    )

    return hashlib.sha256(
        _canonical_json(payload).encode(
            "utf-8"
        )
    ).hexdigest()


@dataclass
class OptimizationCandidate:
    candidate_id: str
    bom: dict
    generation: int = 0
    parent_ids: tuple = ()
    metadata: dict = field(
        default_factory=dict
    )

    def __post_init__(self):
        self.bom = deepcopy(self.bom)
        self.parent_ids = tuple(
            self.parent_ids
        )
        self.metadata = deepcopy(
            self.metadata
        )

    def copy_with(
        self,
        candidate_id: str,
        generation: int | None = None,
        parent_ids: tuple | None = None,
        metadata: dict | None = None,
    ):
        return OptimizationCandidate(
            candidate_id=candidate_id,
            bom=deepcopy(self.bom),
            generation=(
                self.generation
                if generation is None
                else generation
            ),
            parent_ids=(
                self.parent_ids
                if parent_ids is None
                else parent_ids
            ),
            metadata=(
                self.metadata
                if metadata is None
                else metadata
            ),
        )

    def fingerprint(
        self,
        search_space: dict | None = None,
    ) -> str:
        return design_fingerprint(
            self.bom,
            search_space=search_space,
        )
