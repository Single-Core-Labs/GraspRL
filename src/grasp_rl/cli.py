"""Command-line interface for GraspRL."""

import typer

app = typer.Typer(help="GraspRL: Robot arm reinforcement learning toolkit")


@app.command()
def main() -> None:
    """GraspRL CLI entry point."""
    typer.echo("GraspRL v0.1.0")
    typer.echo("Use grasp-rl --help for available commands.")


if __name__ == "__main__":
    app()
