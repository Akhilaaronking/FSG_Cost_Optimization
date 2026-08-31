from src.data.registry import DataRegistry


MM3_PER_M3 = 1_000_000_000.0


def _get_volume_m3(part: dict) -> float:
    """
    Return finished part volume in cubic metres.

    Supports:
    1. Frozen B4 pilot format:
       part["volume_m3"]

    2. Canonical BOM format:
       part["geometry"]["finished_volume_mm3"]
    """

    if "volume_m3" in part:
        volume_m3 = float(part["volume_m3"])

    elif (
        "geometry" in part
        and "finished_volume_mm3" in part["geometry"]
    ):
        volume_mm3 = float(
            part["geometry"]["finished_volume_mm3"]
        )
        volume_m3 = volume_mm3 / MM3_PER_M3

    else:
        raise ValueError(
            "Part does not contain a supported volume field"
        )

    if volume_m3 <= 0:
        raise ValueError(
            "Part volume must be greater than zero"
        )

    return volume_m3


def calculate_unit_mass(
    part: dict,
    registry: DataRegistry | None = None,
) -> float:
    """
    Calculate the mass of one physical part.

    mass = density × finished volume

    Stored values such as part['mass_kg'] are intentionally
    ignored because mass must be deterministically recomputed.
    """

    registry = registry or DataRegistry()

    material_id = part.get("material_id")

    if not material_id:
        raise ValueError(
            "Part is missing material_id"
        )

    material = registry.get_material(material_id)

    density = material.get("density_kg_m3")

    if density is None:
        raise ValueError(
            f"Material {material_id} has no density"
        )

    density = float(density)

    if density <= 0:
        raise ValueError(
            f"Material {material_id} density must be positive"
        )

    volume_m3 = _get_volume_m3(part)

    return density * volume_m3


def calculate_part_mass(
    part: dict,
    registry: DataRegistry | None = None,
) -> float:
    """
    Calculate total mass considering part quantity.

    The frozen B4 pilot parts currently have no quantity field,
    so quantity defaults to 1.
    """

    registry = registry or DataRegistry()

    quantity = part.get("quantity", 1)

    if not isinstance(quantity, int) or quantity < 1:
        raise ValueError(
            "Part quantity must be a positive integer"
        )

    unit_mass = calculate_unit_mass(
        part,
        registry=registry,
    )

    return unit_mass * quantity


def calculate_bom_mass(
    bom: dict,
    registry: DataRegistry | None = None,
) -> float:
    """
    Calculate total BOM mass from every part.
    """

    registry = registry or DataRegistry()

    parts = bom.get("parts")

    if not isinstance(parts, list) or not parts:
        raise ValueError(
            "BOM must contain a non-empty parts list"
        )

    total_mass = 0.0

    for part in parts:
        total_mass += calculate_part_mass(
            part,
            registry=registry,
        )

    return total_mass
