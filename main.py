from src.validation.schema_validator import validate_bom, print_validation_report

if __name__ == "__main__":
    bom, errors = validate_bom()
    ok = print_validation_report(bom, errors)
    raise SystemExit(0 if ok else 1)
