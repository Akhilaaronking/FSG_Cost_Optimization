from decimal import Decimal, ROUND_HALF_UP

from src.data.registry import DataRegistry
from src.mass_engine.mass_calculator import calculate_part_mass


CENT = Decimal("0.01")


def _money(value: Decimal) -> float:
    return float(
        value.quantize(
            CENT,
            rounding=ROUND_HALF_UP,
        )
    )


def _decimal(value) -> Decimal:
    return Decimal(str(value))


def calculate_material_cost(
    part: dict,
    registry: DataRegistry | None = None,
) -> Decimal:
    registry = registry or DataRegistry()

    material_id = part.get("material_id")

    if not material_id:
        raise ValueError("Part is missing material_id")

    material = registry.get_material(material_id)

    price = material.get(
        "unit_price_eur_per_kg"
    )

    if price is None:
        raise ValueError(
            f"Material {material_id} has no "
            "unit_price_eur_per_kg"
        )

    mass_kg = calculate_part_mass(
        part,
        registry=registry,
    )

    return (
        _decimal(mass_kg)
        * _decimal(price)
    )


def _get_process_input(
    part: dict,
    canonical_field: str,
    benchmark_field: str,
):
    process_inputs = part.get(
        "process_inputs",
        {},
    )

    if canonical_field in process_inputs:
        return process_inputs[canonical_field]

    manual = part.get(
        "manual_calculation",
        {},
    )

    if benchmark_field in manual:
        return manual[benchmark_field]

    raise ValueError(
        f"Missing process input: {canonical_field}"
    )


def calculate_process_cost(
    part: dict,
    registry: DataRegistry | None = None,
) -> Decimal:
    registry = registry or DataRegistry()

    process_id = part.get("process_id")

    if not process_id:
        raise ValueError("Part is missing process_id")

    process = registry.get_process(
        process_id
    )

    rate = _decimal(process["rate"])
    rate_unit = process["rate_unit"]

    quantity = _decimal(
        part.get("quantity", 1)
    )

    if rate_unit == "EUR_per_min":
        minutes = _decimal(
            _get_process_input(
                part,
                "time_min",
                "process_time_min_estimate",
            )
        )

        cost = rate * minutes

    elif rate_unit == "EUR_per_hour":
        minutes = _decimal(
            _get_process_input(
                part,
                "time_min",
                "process_time_min_estimate",
            )
        )

        cost = (
            rate
            * minutes
            / Decimal("60")
        )

    elif rate_unit == "EUR_per_meter":
        length_m = _decimal(
            _get_process_input(
                part,
                "length_m",
                "cut_length_m_estimate",
            )
        )

        cost = rate * length_m

    elif rate_unit == "EUR_per_gram":
        mass_g = _decimal(
            _get_process_input(
                part,
                "mass_g",
                "process_mass_g_estimate",
            )
        )

        cost = rate * mass_g

    elif rate_unit == "EUR_per_m2":
        area_m2 = _decimal(
            _get_process_input(
                part,
                "area_m2",
                "process_area_m2_estimate",
            )
        )

        cost = rate * area_m2

    else:
        raise ValueError(
            f"Unsupported process rate unit: "
            f"{rate_unit}"
        )

    return cost * quantity


def calculate_fastener_cost(
    part: dict,
    registry: DataRegistry | None = None,
) -> Decimal:
    registry = registry or DataRegistry()

    total = Decimal("0")

    part_quantity = _decimal(
        part.get("quantity", 1)
    )

    for fastener in part.get(
        "fasteners",
        [],
    ):
        fastener_id = fastener[
            "fastener_id"
        ]

        quantity = fastener.get(
            "qty",
            fastener.get("quantity", 1),
        )

        record = registry.get_fastener(
            fastener_id
        )

        unit_price = record.get(
            "unit_price_eur"
        )

        if unit_price is None:
            raise ValueError(
                f"Fastener {fastener_id} "
                "has no unit price"
            )

        total += (
            _decimal(unit_price)
            * _decimal(quantity)
            * part_quantity
        )

    return total


def calculate_part_cost(
    part: dict,
    registry: DataRegistry | None = None,
) -> dict:
    registry = registry or DataRegistry()

    material_raw = calculate_material_cost(
        part,
        registry,
    )

    process_raw = calculate_process_cost(
        part,
        registry,
    )

    fastener_raw = calculate_fastener_cost(
        part,
        registry,
    )

    total_raw = (
        material_raw
        + process_raw
        + fastener_raw
    )

    return {
        "part_id": part.get("part_id"),
        "material_cost_eur": _money(
            material_raw
        ),
        "process_cost_eur": _money(
            process_raw
        ),
        "fastener_cost_eur": _money(
            fastener_raw
        ),
        "total_cost_eur": _money(
            total_raw
        ),
        "process_id": part.get(
            "process_id"
        ),
    }


def calculate_bom_cost(
    bom: dict,
    registry: DataRegistry | None = None,
) -> dict:
    registry = registry or DataRegistry()

    parts = bom.get("parts")

    if not isinstance(parts, list) or not parts:
        raise ValueError(
            "BOM must contain a non-empty parts list"
        )

    results = []
    total_raw = Decimal("0")

    for part in parts:
        material_raw = calculate_material_cost(
            part,
            registry,
        )
        process_raw = calculate_process_cost(
            part,
            registry,
        )
        fastener_raw = calculate_fastener_cost(
            part,
            registry,
        )

        part_total_raw = (
            material_raw
            + process_raw
            + fastener_raw
        )

        total_raw += part_total_raw

        results.append({
            "part_id": part.get("part_id"),
            "material_cost_eur": _money(
                material_raw
            ),
            "process_cost_eur": _money(
                process_raw
            ),
            "fastener_cost_eur": _money(
                fastener_raw
            ),
            "total_cost_eur": _money(
                part_total_raw
            ),
            "process_id": part.get(
                "process_id"
            ),
        })

    return {
        "parts": results,
        "total_cost_eur": _money(
            total_raw
        ),
    }
