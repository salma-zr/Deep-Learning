import argparse
import subprocess
from pathlib import Path

from src.utils import load_config, log_config, log_versions, setup_logger


def run_command(cmd: list[str], cwd: str | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PDF report.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("build_report")
    log_versions(logger)
    log_config(logger, cfg)

    report_tex = Path(cfg["report_tex"])
    report_dir = report_tex.parent

    run_command(["python", "-m", "src.report.report_tables", "--config", args.config])
    run_command(["python", "-m", "src.report.make_figures", "--config", args.config])

    output_dir = Path(cfg.get("output_dir", "report/output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        run_command(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={output_dir}",
                report_tex.name,
            ],
            cwd=str(report_dir),
        )
        logger.info("PDF generated at %s", output_dir / report_tex.with_suffix(".pdf").name)
    except subprocess.CalledProcessError as exc:
        logger.error("LaTeX build failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
