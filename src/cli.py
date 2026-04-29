"""Command-line interface for the Intelligent Routing System."""
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich import box

from .clients.ollama_client import OllamaClient
from .clients.claude_client import ClaudeClient
from .router import Router
from .utils.logger import RequestLogger
from .config import config

console = Console()


class IRS:
    """Intelligent Routing System orchestrator."""

    def __init__(self):
        """Initialize the routing system."""
        self.ollama_client = OllamaClient()
        self.claude_client = ClaudeClient()
        self.router = Router()
        self.logger = RequestLogger()
        self.last_model_used = None

    def generate(self, prompt: str, force_model: str = None, continue_conversation: bool = False) -> dict:
        """Generate code using intelligent routing.

        Args:
            prompt: The coding prompt
            force_model: Force a specific model ('local' or 'cloud')
            continue_conversation: If True, continue the previous conversation

        Returns:
            Dictionary with response and metadata
        """
        if continue_conversation and self.last_model_used:
            target = self.last_model_used
            reason = f"Continuing conversation with {target} model"
            decision = None
        elif force_model:
            decision = None
            target = force_model
            reason = f"Forced to {force_model}"
        else:
            decision = self.router.route(prompt)
            target = decision.target
            reason = decision.reason

        if target == "local":
            result = self.ollama_client.generate(prompt, continue_conversation=continue_conversation)
            model_used = "local"
        else:
            result = self.claude_client.generate(prompt, continue_conversation=continue_conversation)
            model_used = "cloud"

        self.last_model_used = model_used

        self.logger.log_request(
            prompt=prompt,
            routing_decision={
                "target": target,
                "reason": reason,
                "confidence": decision.confidence if decision else 1.0,
                "features": decision.features if decision else {}
            },
            response=result.get("response", ""),
            model_used=model_used,
            latency_ms=result.get("latency_ms", 0),
            success=result.get("success", False),
            error=result.get("error")
        )

        return {
            "response": result.get("response", ""),
            "model_used": model_used,
            "latency_ms": result.get("latency_ms", 0),
            "success": result.get("success", False),
            "error": result.get("error"),
            "routing_reason": reason,
            "routing_decision": decision
        }


@click.group()
def cli():
    """Intelligent Routing System - Route coding queries between local and cloud models."""
    pass


@cli.command()
@click.argument("prompt", required=False)
@click.option("--force-local", is_flag=True, help="Force routing to local model")
@click.option("--force-cloud", is_flag=True, help="Force routing to cloud model")
@click.option("--interactive", "-i", is_flag=True, help="Interactive mode")
def generate(prompt, force_local, force_cloud, interactive):
    """Generate code from a prompt."""
    irs = IRS()

    if not irs.ollama_client.health_check():
        console.print("[red]Error: Ollama is not running or model not found![/red]")
        console.print(f"[yellow]Make sure Ollama is running and '{config.ollama_model}' is available.[/yellow]")
        console.print(f"[yellow]Run: ollama pull {config.ollama_model}[/yellow]")
        return

    if interactive:
        console.print("[bold cyan]Interactive Mode[/bold cyan] - Conversations continue automatically")
        console.print("[dim]Type 'exit' or 'quit' to stop, 'new' to start fresh conversation[/dim]\n")
        first_message = True
        while True:
            try:
                prompt = console.input("[bold green]Prompt:[/bold green] ")
                if prompt.lower() in ["exit", "quit"]:
                    break

                if prompt.lower() == "new":
                    irs = IRS()
                    first_message = True
                    console.print("[yellow]Started new conversation[/yellow]\n")
                    continue

                if not prompt.strip():
                    continue

                continue_conv = not first_message
                _generate_and_display(irs, prompt, force_local, force_cloud, continue_conversation=continue_conv)
                first_message = False

            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted[/yellow]")
                break
        return

    if not prompt:
        console.print("[red]Error: No prompt provided[/red]")
        console.print("Usage: irs generate 'your prompt here' or use --interactive")
        return

    _generate_and_display(irs, prompt, force_local, force_cloud)


def _generate_and_display(irs: IRS, prompt: str, force_local: bool, force_cloud: bool, continue_conversation: bool = False):
    """Generate and display the result."""
    force_model = None
    if force_local:
        force_model = "local"
    elif force_cloud:
        force_model = "cloud"

    prompt_title = "[bold]Prompt[/bold]" if not continue_conversation else "[bold]Follow-up[/bold]"
    console.print(Panel(prompt, title=prompt_title, border_style="blue"))

    with console.status("[bold yellow]Thinking...[/bold yellow]"):
        result = irs.generate(prompt, force_model=force_model, continue_conversation=continue_conversation)

    model_color = "green" if result["model_used"] == "local" else "cyan"
    status_icon = "✓" if result["success"] else "✗"
    status_color = "green" if result["success"] else "red"

    console.print(f"\n[{status_color}]{status_icon}[/{status_color}] "
                  f"[{model_color}]Model: {result['model_used'].upper()}[/{model_color}] "
                  f"[dim]({result['latency_ms']}ms)[/dim]")
    console.print(f"[dim]Reason: {result['routing_reason']}[/dim]\n")

    if result["success"]:
        console.print(Panel(
            Markdown(result["response"]),
            title="[bold]Response[/bold]",
            border_style=model_color
        ))
    else:
        console.print(Panel(
            f"[red]Error: {result['error']}[/red]",
            title="[bold]Error[/bold]",
            border_style="red"
        ))


@cli.command()
def stats():
    """Show routing statistics."""
    logger = RequestLogger()
    stats = logger.get_statistics()

    if stats["total_requests"] == 0:
        console.print("[yellow]No requests logged yet.[/yellow]")
        return

    table = Table(title="Routing Statistics", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Requests", str(stats["total_requests"]))
    table.add_row("Success Rate", f"{stats['success_rate']:.1f}%")
    table.add_row("", "")
    table.add_row("Local Requests", f"{stats['local_requests']} ({stats['local_percentage']:.1f}%)")
    table.add_row("Cloud Requests", f"{stats['cloud_requests']} ({stats['cloud_percentage']:.1f}%)")
    table.add_row("", "")
    table.add_row("Avg Latency", f"{stats['avg_latency_ms']:.0f}ms")
    table.add_row("Avg Local Latency", f"{stats['avg_local_latency_ms']:.0f}ms")
    table.add_row("Avg Cloud Latency", f"{stats['avg_cloud_latency_ms']:.0f}ms")

    console.print(table)


@cli.command()
@click.option("--limit", "-n", default=10, help="Number of recent requests to show")
def history(limit):
    """Show recent request history."""
    logger = RequestLogger()
    logs = logger.get_recent_logs(limit)

    if not logs:
        console.print("[yellow]No requests logged yet.[/yellow]")
        return

    table = Table(title=f"Recent Requests (last {len(logs)})", box=box.SIMPLE)
    table.add_column("Time", style="dim")
    table.add_column("Model", style="cyan")
    table.add_column("Latency", style="green")
    table.add_column("Prompt Preview", style="white")
    table.add_column("Status", style="green")

    for log in logs:
        timestamp = log["timestamp"].split("T")[1].split(".")[0]
        model = log["model_used"]
        latency = f"{log['latency_ms']}ms"
        prompt_preview = log["prompt"][:50] + "..." if len(log["prompt"]) > 50 else log["prompt"]
        status = "✓" if log["success"] else "✗"

        table.add_row(timestamp, model, latency, prompt_preview, status)

    console.print(table)


@cli.command()
def check():
    """Check system health."""
    console.print("[bold]System Health Check[/bold]\n")

    ollama = OllamaClient()
    ollama_status = ollama.health_check()

    if ollama_status:
        console.print("[green]✓[/green] Ollama: Running")
        models = ollama.list_models()
        console.print(f"  [dim]Available models: {', '.join(models)}[/dim]")
    else:
        console.print("[red]✗[/red] Ollama: Not available")

    claude = ClaudeClient()
    claude_status = claude.health_check()

    if claude_status:
        console.print("[green]✓[/green] Claude API: Configured")
    else:
        console.print("[yellow]○[/yellow] Claude API: Not configured (optional for Phase 1)")

    console.print(f"\n[bold]Configuration:[/bold]")
    console.print(f"  Ollama URL: {config.ollama_base_url}")
    console.print(f"  Ollama Model: {config.ollama_model}")
    console.print(f"  Routing Strategy: {config.routing_strategy}")
    console.print(f"  Prompt Threshold: {config.prompt_length_threshold} tokens")


if __name__ == "__main__":
    cli()
