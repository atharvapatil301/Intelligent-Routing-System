"""Command-line interface for the Intelligent Routing System."""
import json
from pathlib import Path
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from .clients.ollama_client import OllamaClient
from .clients.claude_client import ClaudeClient
from .router import Router
from .utils.logger import RequestLogger
from .config import config
from .dataset_generator import DatasetGenerator
from .feature_extractor import FeatureExtractor
from .vector_db import QueryVectorDB

console = Console()


class IRS:
    """Intelligent Routing System orchestrator."""

    def __init__(self, use_enhanced_routing: bool = True):
        """Initialize the routing system.

        Args:
            use_enhanced_routing: If True, use enhanced routing with features and vector DB
        """
        self.ollama_client = OllamaClient()
        self.claude_client = ClaudeClient()

        # Use enhanced routing with Phase 2 features
        strategy = "enhanced" if use_enhanced_routing else config.routing_strategy
        self.router = Router(
            strategy=strategy,
            use_features=use_enhanced_routing,
            use_vector_db=use_enhanced_routing
        )

        self.logger = RequestLogger()
        self.vector_db = QueryVectorDB() if use_enhanced_routing else None
        self.feature_extractor = FeatureExtractor() if use_enhanced_routing else None
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

        # Save to vector database if enabled (Phase 2)
        if self.vector_db is not None and self.feature_extractor is not None:
            try:
                # Extract features with embedding
                features = self.feature_extractor.extract(prompt, generate_embedding=True)

                # Generate unique query ID
                import uuid
                query_id = str(uuid.uuid4())

                # Store in vector DB
                self.vector_db.add_query(
                    query_id=query_id,
                    prompt=prompt,
                    embedding=features.embedding,
                    metadata={
                        "model_used": model_used,
                        "success": result.get("success", False),
                        "latency_ms": result.get("latency_ms", 0),
                        "has_complexity_keywords": features.has_complexity_keywords,
                        "has_concurrency": features.has_concurrency_mentions,
                        "token_count": features.token_count,
                    }
                )
            except Exception as e:
                # Don't fail the request if vector DB storage fails
                console.print(f"[yellow]Warning: Failed to save to vector DB: {e}[/yellow]")

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


@cli.command()
@click.option("--prompts-file", "-f", default="data/seed_prompts.json", help="Path to prompts JSON file")
@click.option("--num-samples", "-n", type=int, help="Number of samples to generate (default: all)")
@click.option("--output", "-o", default="dataset.jsonl", help="Output filename")
@click.option("--evaluation", "-e", default="heuristic", type=click.Choice(["heuristic", "unit_test", "llm_judge"]), help="Evaluation method")
@click.option("--category", "-c", help="Only generate from this category")
def generate_dataset(prompts_file, num_samples, output, evaluation, category):
    """Generate training dataset by running both models (Phase 3)."""
    console.print("[bold cyan]Dataset Generation - Phase 3[/bold cyan]\n")

    # Load prompts
    prompts_path = Path(prompts_file)
    if not prompts_path.exists():
        console.print(f"[red]Error: Prompts file not found: {prompts_file}[/red]")
        return

    with open(prompts_path) as f:
        data = json.load(f)

    # Collect prompts
    all_prompts = []
    if category:
        if category in data.get("categories", {}):
            all_prompts = data["categories"][category]
            console.print(f"[yellow]Using category: {category}[/yellow]")
        else:
            console.print(f"[red]Error: Category '{category}' not found[/red]")
            console.print(f"[yellow]Available categories: {', '.join(data.get('categories', {}).keys())}[/yellow]")
            return
    else:
        for cat_prompts in data.get("categories", {}).values():
            all_prompts.extend(cat_prompts)
        console.print(f"[yellow]Using all categories ({len(data.get('categories', {}))} categories)[/yellow]")

    if num_samples:
        all_prompts = all_prompts[:num_samples]

    console.print(f"[green]Generating {len(all_prompts)} samples using {evaluation} evaluation[/green]\n")

    # Check models
    ollama = OllamaClient()
    if not ollama.health_check():
        console.print("[red]Error: Ollama is not running![/red]")
        return

    # Generate dataset
    generator = DatasetGenerator()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("Generating samples...", total=len(all_prompts))

        for i, prompt in enumerate(all_prompts, 1):
            try:
                generator.generate_sample(prompt, evaluation_method=evaluation)
                progress.update(task, advance=1)

                # Auto-save every 10 samples
                if i % 10 == 0:
                    generator.save_dataset(f"progress_{output}")

            except Exception as e:
                console.print(f"\n[red]Error on sample {i}: {str(e)}[/red]")
                progress.update(task, advance=1)
                continue

    # Save final dataset
    output_path = generator.save_dataset(output)

    # Show statistics
    stats = generator.get_statistics()
    console.print("\n[bold]Dataset Statistics:[/bold]")
    table = Table(box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Samples", str(stats["total"]))
    table.add_row("Local Sufficient", f"{stats['local_sufficient']} ({stats['local_percentage']:.1f}%)")
    table.add_row("Cloud Needed", f"{stats['cloud_needed']} ({stats['cloud_percentage']:.1f}%)")
    table.add_row("", "")
    table.add_row("Qwen Success Rate", f"{stats['qwen_success_rate']:.1f}%")
    table.add_row("Claude Success Rate", f"{stats['claude_success_rate']:.1f}%")
    table.add_row("", "")
    table.add_row("Avg Qwen Score", f"{stats['avg_qwen_score']:.2f}")
    table.add_row("Avg Claude Score", f"{stats['avg_claude_score']:.2f}")

    console.print(table)
    console.print(f"\n[green]✓ Dataset saved to: {output_path}[/green]")


@cli.command()
@click.argument("dataset_file")
@click.option("--train-ratio", default=0.7, help="Training set ratio")
@click.option("--val-ratio", default=0.15, help="Validation set ratio")
@click.option("--test-ratio", default=0.15, help="Test set ratio")
@click.option("--prefix", "-p", default="dataset", help="Output file prefix")
def split_dataset(dataset_file, train_ratio, val_ratio, test_ratio, prefix):
    """Split dataset into train/val/test sets."""
    console.print("[bold cyan]Splitting Dataset[/bold cyan]\n")

    if not Path(dataset_file).exists():
        console.print(f"[red]Error: Dataset file not found: {dataset_file}[/red]")
        return

    generator = DatasetGenerator()
    generator.load_dataset(dataset_file)

    train, val, test = generator.create_splits(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        shuffle=True
    )

    paths = generator.save_splits(train, val, test, prefix=prefix)

    console.print("\n[green]✓ Dataset split complete![/green]")
    for split_name, path in paths.items():
        console.print(f"  {split_name}: {path}")


@cli.command()
@click.argument("prompt")
def extract_features(prompt):
    """Extract features from a prompt (Phase 2)."""
    console.print("[bold cyan]Feature Extraction - Phase 2[/bold cyan]\n")

    extractor = FeatureExtractor()

    with console.status("[yellow]Extracting features...[/yellow]"):
        features = extractor.extract(prompt, generate_embedding=True)

    # Display features
    console.print(Panel(prompt, title="[bold]Prompt[/bold]", border_style="blue"))

    # Basic features
    table = Table(title="Basic Features", box=box.ROUNDED)
    table.add_column("Feature", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Character Count", str(features.char_count))
    table.add_row("Token Count", str(features.token_count))
    table.add_row("Word Count", str(features.word_count))
    table.add_row("Line Count", str(features.line_count))
    table.add_row("Has Code Block", str(features.has_code_block))
    table.add_row("Num Code Blocks", str(features.num_code_blocks))

    console.print(table)

    # Structural features
    table2 = Table(title="Structural Features", box=box.ROUNDED)
    table2.add_column("Feature", style="cyan")
    table2.add_column("Value", style="green")

    table2.add_row("Functions Requested", str(features.num_functions_requested))
    table2.add_row("Classes Requested", str(features.num_classes_requested))
    table2.add_row("Question Types", ", ".join(features.question_types) if features.question_types else "None")

    console.print(table2)

    # Complexity signals
    table3 = Table(title="Complexity Signals", box=box.ROUNDED)
    table3.add_column("Feature", style="cyan")
    table3.add_column("Value", style="green")

    table3.add_row("Has Complexity Keywords", str(features.has_complexity_keywords))
    if features.matched_keywords:
        table3.add_row("Matched Keywords", ", ".join(features.matched_keywords))
    table3.add_row("Has Concurrency", str(features.has_concurrency_mentions))
    table3.add_row("Has Algorithm Complexity", str(features.has_algorithm_complexity))
    table3.add_row("Has Reasoning Keywords", str(features.has_reasoning_keywords))

    console.print(table3)

    # Embedding info
    if features.embedding is not None:
        console.print(f"\n[green]✓ Embedding generated: {features.embedding.shape[0]} dimensions[/green]")


@cli.command()
@click.option("--action", type=click.Choice(["stats", "clear", "export", "import"]), required=True, help="Action to perform")
@click.option("--file", "-f", help="File path for export/import")
def vector_db(action, file):
    """Manage vector database."""
    console.print("[bold cyan]Vector Database Management[/bold cyan]\n")

    db = QueryVectorDB()

    if action == "stats":
        stats = db.get_stats()
        table = Table(title="Vector DB Statistics", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Queries", str(stats["total_queries"]))
        table.add_row("Local Queries", str(stats["local_queries"]))
        table.add_row("Cloud Queries", str(stats["cloud_queries"]))
        table.add_row("Success Rate", f"{stats['success_rate']:.1%}")

        console.print(table)

    elif action == "clear":
        if click.confirm("Are you sure you want to clear the vector database?"):
            db.clear()
            console.print("[green]✓ Vector database cleared[/green]")
        else:
            console.print("[yellow]Cancelled[/yellow]")

    elif action == "export":
        if not file:
            console.print("[red]Error: --file required for export[/red]")
            return
        db.export_to_json(file)

    elif action == "import":
        if not file:
            console.print("[red]Error: --file required for import[/red]")
            return
        if not Path(file).exists():
            console.print(f"[red]Error: File not found: {file}[/red]")
            return
        db.import_from_json(file)


if __name__ == "__main__":
    cli()
