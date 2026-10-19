"""Command-line interface for GraspRL."""

import json
import sys
from pathlib import Path
import typer
from grasp_rl.data.validator import validate_data

app = typer.Typer(help="GraspRL: Robot arm reinforcement learning toolkit")
data_app = typer.Typer(help="Data validation and preprocessing")
app.add_typer(data_app, name="data")


@app.command()
def main() -> None:
    """GraspRL CLI entry point."""
    typer.echo("GraspRL v0.1.0")
    typer.echo("Use grasp-rl --help for available commands.")


@data_app.command()
def validate(
    path: Path = typer.Argument(..., help="Path to demonstration data file (JSON)"),
) -> None:
    """Validate demonstration data against schema.

    Returns non-zero exit code on validation failure.
    """
    if not path.exists():
        typer.echo(f"Error: File not found: {path}", err=True)
        raise typer.Exit(code=1)

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as e:
        typer.echo(f"Error reading file: {e}", err=True)
        raise typer.Exit(code=1)

    result = validate_data(data)

    if result.valid:
        typer.echo(f"✓ Validation passed: {path}")
        typer.echo(f"  Schema version: {data.get('schema_version', 'unknown')}")
        raise typer.Exit(code=0)
    else:
        typer.echo(f"✗ Validation failed: {path}", err=True)
        typer.echo(f"  Errors found: {result.error_count}")
        for error in result.errors[:10]:  # Show first 10 errors
            typer.echo(
                f"  - Episode: {error.episode_id}, Step: {error.step_index}, "
                f"Error: {error.message}",
                err=True,
            )
        if result.error_count > 10:
            typer.echo(f"  ... and {result.error_count - 10} more errors", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
