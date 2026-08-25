def build_engineering_query(
    part: dict,
    change_type: str | None = None,
    target_field: str | None = None,
    user_intent: str | None = None,
) -> str:
    pieces = [
        "Formula Student cost and manufacturing evidence relevant to an engineering candidate."
    ]

    if part.get("part_id"):
        pieces.append(
            f"Part ID: {part['part_id']}."
        )

    name = part.get(
        "part_name",
        part.get("name"),
    )
    if name:
        pieces.append(f"Part name: {name}.")

    if part.get("material_id"):
        pieces.append(
            f"Current material: {part['material_id']}."
        )

    if part.get("process_id"):
        pieces.append(
            f"Current process: {part['process_id']}."
        )

    if change_type:
        pieces.append(
            f"Proposed change type: {change_type}."
        )

    if target_field:
        pieces.append(
            f"Target field: {target_field}."
        )

    if user_intent:
        pieces.append(
            f"Information need: {user_intent}."
        )

    pieces.append(
        "Retrieve evidence about cost realism, bought or made status, manufacturing process, BOM requirements, metric units, catalog availability, source traceability and applicable constraints."
    )

    return " ".join(pieces)
