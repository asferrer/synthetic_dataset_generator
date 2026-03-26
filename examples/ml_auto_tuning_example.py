#!/usr/bin/env python3
"""
ML Auto-Tuning Example Script
==============================
Ejemplo completo de cómo usar el sistema de auto-tuning predictivo
para optimizar la configuración de efectos y minimizar el domain gap.

Uso:
    python ml_auto_tuning_example.py --reference-dir /path/to/real/images \\
                                      --synthetic-dir /path/to/synthetic/images \\
                                      --trials 20

Requisitos:
    pip install httpx rich
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import httpx
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn


console = Console()


class MLAutoTuner:
    """Cliente para el sistema de auto-tuning ML."""

    def __init__(self, gateway_url: str = "http://localhost:8000"):
        self.gateway_url = gateway_url
        self.client = httpx.Client(base_url=gateway_url, timeout=30.0)

    def create_reference_set(
        self,
        name: str,
        directory: str,
        domain_id: str = "default",
    ) -> str:
        """Crear reference set desde directorio de imágenes reales.

        Returns:
            Reference set ID.
        """
        console.print(f"[cyan]Creando reference set desde:[/cyan] {directory}")

        response = self.client.post(
            "/domain-gap/references/from-directory",
            json={
                "name": name,
                "directory_path": directory,
                "domain_id": domain_id,
            },
        )
        response.raise_for_status()
        data = response.json()

        ref_set_id = data["set_id"]
        image_count = data["image_count"]

        console.print(f"[green]✓[/green] Reference set creado: {ref_set_id} ({image_count} imágenes)")
        return ref_set_id

    def measure_baseline_gap(
        self,
        synthetic_dir: str,
        reference_set_id: str,
        max_images: int = 100,
    ) -> float:
        """Medir gap score del baseline (sin optimización).

        Returns:
            Overall gap score (0-100).
        """
        console.print(f"[cyan]Midiendo gap score baseline...[/cyan]")

        response = self.client.post(
            "/domain-gap/metrics/compute",
            json={
                "synthetic_dir": synthetic_dir,
                "reference_set_id": reference_set_id,
                "max_images": max_images,
                "compute_radio_mmd": True,
                "compute_fd_radio": True,
                "compute_fid": False,
                "compute_kid": False,
                "compute_color_distribution": True,
                "compute_prdc": True,
            },
        )
        response.raise_for_status()
        data = response.json()

        gap_score = data["overall_gap_score"]
        gap_level = data["gap_level"]

        console.print(f"[yellow]Baseline gap score:[/yellow] {gap_score:.1f} ({gap_level})")
        return gap_score

    def start_ml_optimization(
        self,
        synthetic_dir: str,
        reference_set_id: str,
        n_trials: int = 20,
        probe_size: int = 50,
        warm_start: bool = True,
        current_config: Optional[dict] = None,
    ) -> str:
        """Iniciar optimización ML-guided.

        Returns:
            Job ID de optimización.
        """
        console.print(f"[cyan]Iniciando optimización ML con {n_trials} trials...[/cyan]")

        response = self.client.post(
            "/ml-optimize/start",
            json={
                "synthetic_dir": synthetic_dir,
                "reference_set_id": reference_set_id,
                "n_trials": n_trials,
                "probe_size": probe_size,
                "warm_start": warm_start,
                "current_config": current_config,
            },
        )
        response.raise_for_status()
        data = response.json()

        job_id = data["job_id"]
        console.print(f"[green]✓[/green] Optimización iniciada: {job_id}")
        return job_id

    def wait_for_optimization(self, job_id: str) -> dict:
        """Esperar a que la optimización termine, mostrando progreso.

        Returns:
            Job status final.
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Optimizando...", total=None)

            while True:
                response = self.client.get(f"/ml-optimize/jobs/{job_id}")
                response.raise_for_status()
                status = response.json()

                if status["status"] == "completed":
                    progress.update(task, description="[green]✓ Optimización completada")
                    break
                elif status["status"] == "failed":
                    progress.update(task, description="[red]✗ Optimización falló")
                    break
                elif status["status"] == "cancelled":
                    progress.update(task, description="[yellow]⚠ Optimización cancelada")
                    break

                current = status.get("current_trial", 0)
                total = status.get("total_trials", 0)
                best = status.get("best_gap_score")

                desc = f"Trial {current}/{total}"
                if best is not None:
                    desc += f" | Mejor gap: {best:.1f}"

                progress.update(task, description=desc)
                time.sleep(5)

        return status

    def get_feature_importance(self, source: str = "combined") -> dict:
        """Obtener análisis de importancia de parámetros.

        Args:
            source: "bayesian", "ml_predictor", o "combined"

        Returns:
            Feature importance dict.
        """
        response = self.client.get(
            "/ml-optimize/feature-importance",
            params={"source": source},
        )
        response.raise_for_status()
        return response.json()

    def predict_gap_score(self, config: dict) -> Optional[float]:
        """Predecir gap score para una configuración (sin generar datos).

        Args:
            config: Configuración de efectos.

        Returns:
            Predicted gap score, o None si falla.
        """
        response = self.client.post(
            "/ml-optimize/predict",
            json={"config": config},
        )
        response.raise_for_status()
        data = response.json()

        if not data["success"]:
            console.print(f"[yellow]⚠[/yellow] {data['message']}")
            return None

        return data["predicted_score"]

    def display_results(self, status: dict, baseline_gap: Optional[float] = None):
        """Mostrar resultados de optimización en tabla bonita."""
        best_gap = status.get("best_gap_score")
        best_config = status.get("best_config", {})
        history = status.get("optimization_history", [])

        # Tabla de resumen
        table = Table(title="Resultados de Optimización ML", show_header=True)
        table.add_column("Métrica", style="cyan")
        table.add_column("Valor", style="green")

        if baseline_gap is not None:
            table.add_row("Baseline Gap Score", f"{baseline_gap:.1f}")

        if best_gap is not None:
            table.add_row("Mejor Gap Score", f"{best_gap:.1f}")

            if baseline_gap is not None:
                improvement = (baseline_gap - best_gap) / baseline_gap * 100
                table.add_row("Mejora", f"{improvement:.1f}%")

        table.add_row("Trials Completados", str(len(history)))

        console.print(table)

        # Mostrar mejor configuración
        if best_config:
            console.print("\n[cyan]Mejor Configuración:[/cyan]")
            for param, value in sorted(best_config.items()):
                if isinstance(value, float):
                    console.print(f"  {param}: {value:.3f}")
                else:
                    console.print(f"  {param}: {value}")

    def display_feature_importance(self, importance_data: dict):
        """Mostrar importancia de parámetros en tabla."""
        if not importance_data["success"]:
            console.print("[yellow]⚠ No hay datos de feature importance disponibles[/yellow]")
            return

        importances = importance_data["importances"]
        top_features = importance_data["top_features"]

        table = Table(title="Importancia de Parámetros", show_header=True)
        table.add_column("Parámetro", style="cyan")
        table.add_column("Importancia", style="green", justify="right")
        table.add_column("Impacto", style="yellow")

        for param in top_features:
            importance = importances[param]

            # Visual bar
            bar_length = int(importance * 20)
            bar = "█" * bar_length

            # Impact level
            if importance > 0.20:
                impact = "Crítico"
            elif importance > 0.10:
                impact = "Alto"
            else:
                impact = "Moderado"

            table.add_row(param, f"{importance:.3f}", f"{bar} {impact}")

        console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="ML Auto-Tuning para minimizar domain gap"
    )
    parser.add_argument(
        "--reference-dir",
        required=True,
        help="Directorio con imágenes reales de referencia",
    )
    parser.add_argument(
        "--synthetic-dir",
        required=True,
        help="Directorio con imágenes sintéticas a optimizar",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Número de trials de optimización (default: 20)",
    )
    parser.add_argument(
        "--probe-size",
        type=int,
        default=50,
        help="Imágenes por trial (default: 50)",
    )
    parser.add_argument(
        "--no-warm-start",
        action="store_true",
        help="Deshabilitar warm-start con predictor ML",
    )
    parser.add_argument(
        "--gateway-url",
        default="http://localhost:8000",
        help="URL del gateway service (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--reference-name",
        default="Auto-created References",
        help="Nombre para el reference set",
    )

    args = parser.parse_args()

    # Validar directorios
    if not Path(args.reference_dir).is_dir():
        console.print(f"[red]Error:[/red] {args.reference_dir} no es un directorio válido")
        return 1

    if not Path(args.synthetic_dir).is_dir():
        console.print(f"[red]Error:[/red] {args.synthetic_dir} no es un directorio válido")
        return 1

    # Inicializar cliente
    tuner = MLAutoTuner(gateway_url=args.gateway_url)

    try:
        # 1. Crear reference set
        ref_set_id = tuner.create_reference_set(
            name=args.reference_name,
            directory=args.reference_dir,
        )

        # 2. Medir baseline gap
        baseline_gap = tuner.measure_baseline_gap(
            synthetic_dir=args.synthetic_dir,
            reference_set_id=ref_set_id,
        )

        # 3. Iniciar optimización ML
        job_id = tuner.start_ml_optimization(
            synthetic_dir=args.synthetic_dir,
            reference_set_id=ref_set_id,
            n_trials=args.trials,
            probe_size=args.probe_size,
            warm_start=not args.no_warm_start,
        )

        # 4. Esperar a que termine
        status = tuner.wait_for_optimization(job_id)

        # 5. Mostrar resultados
        console.print("\n")
        tuner.display_results(status, baseline_gap=baseline_gap)

        # 6. Mostrar feature importance
        console.print("\n")
        importance_data = tuner.get_feature_importance()
        tuner.display_feature_importance(importance_data)

        # 7. Guardar configuración óptima
        best_config = status.get("best_config")
        if best_config:
            output_file = Path("best_config.json")
            import json
            with open(output_file, "w") as f:
                json.dump(best_config, f, indent=2)
            console.print(f"\n[green]✓[/green] Configuración óptima guardada en: {output_file}")

        console.print("\n[green bold]✓ Auto-tuning completado exitosamente[/green bold]")
        return 0

    except httpx.HTTPStatusError as e:
        console.print(f"[red]Error HTTP {e.response.status_code}:[/red] {e.response.text}")
        return 1
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
