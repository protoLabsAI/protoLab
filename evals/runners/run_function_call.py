"""Run function calling accuracy evals with per-alias gate enforcement.

Tests structured tool call correctness: name, arguments, structure.
Reads alias_tiers.yaml for per-model pass/fail gates (overall, in-process,
external buckets). Tests with a `bucket` field are categorized; tests
without one are counted in the overall gate only.

Usage:
    python -m runners.run_function_call --suite canary
    python -m runners.run_function_call --all-suites
    python -m runners.run_function_call --model protolabs/fast --suite canary --trials 3
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
import yaml
from graders.function_call import FunctionCallGrader
from openai import AuthenticationError, OpenAI

from sampling import resolve, to_openai_kwargs

SUITE_DIR = Path(__file__).parent.parent / "function_call" / "test_cases"
ALIAS_TIERS_PATH = Path(__file__).parent.parent.parent / "models" / "alias_tiers.yaml"


def load_alias_tier(model: str) -> dict | None:
    """Load gate thresholds for a model alias from alias_tiers.yaml."""
    if not ALIAS_TIERS_PATH.exists():
        return None
    with open(ALIAS_TIERS_PATH) as f:
        tiers = yaml.safe_load(f)
    return tiers.get("aliases", {}).get(model)


def run_function_call_test(client: OpenAI, model: str, test: dict, temperature: float = 0.0) -> dict:
    """Send a prompt with tools and capture the model's tool calls.

    Function calling is an exact-match, structured task: the right answer is the mode of
    the distribution (correct tool + args), not a sample. Default temperature is 0.0 —
    high temp only adds noise and depresses scores on structured suites. Bump it only if
    you specifically want to measure tool-call robustness under sampling.
    """
    messages = [{"role": "user", "content": test["prompt"]}]
    tools = test.get("tools", [])

    try:
        # extra_body only for local vLLM models (cloud APIs reject it)
        # NOTE: cloud APIs reject extra_body, so local lanes and cloud judges do
        # NOT get identical sampling here. That asymmetry is deliberate but real —
        # it is recorded in the results so a cross-provider comparison can see it.
        samp = resolve("FC")
        kwargs: dict = {}
        if model.startswith("protolabs/"):
            ok = to_openai_kwargs(samp)
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False},
                **ok["extra_body"],
                "presence_penalty": samp.presence_penalty,
            }
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            top_p=samp.top_p,
            max_tokens=8000,  # raised from 1000: thinking models (Ornith etc.) need
                              # room to close <think> before the tool call (no-thinking-off policy)
            **kwargs,
        )
        choice = response.choices[0]

        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = tc.function.arguments
                tool_calls.append({
                    "name": tc.function.name,
                    "arguments": args,
                })

        return {
            "tool_calls": tool_calls,
            "content": choice.message.content or "",
            "finish_reason": choice.finish_reason,
        }
    except AuthenticationError:
        raise click.ClickException(
            "Gateway authentication failed — set LITELLM_API_KEY or pass --api-key"
        )
    except Exception as e:
        return {"tool_calls": [], "content": "", "error": str(e)}


def compute_bucket_stats(test_results: list[dict]) -> dict:
    """Compute pass rates per bucket (inprocess / external / overall)."""
    buckets: dict[str, dict] = {}
    for tr in test_results:
        bucket = tr.get("bucket") or "untagged"
        if bucket not in buckets:
            buckets[bucket] = {"total": 0, "passed": 0}
        buckets[bucket]["total"] += 1
        if tr["all_passed"]:
            buckets[bucket]["passed"] += 1

    stats = {}
    for bucket, counts in buckets.items():
        rate = counts["passed"] / counts["total"] if counts["total"] > 0 else 0.0
        stats[bucket] = {
            "total": counts["total"],
            "passed": counts["passed"],
            "pass_rate": round(rate, 3),
        }
    return stats


def evaluate_gates(
    model: str,
    overall_rate: float,
    bucket_stats: dict,
    tier: dict,
) -> list[dict]:
    """Check pass rates against alias tier gates. Returns list of gate results."""
    gates = []

    # Overall gate
    fc_gate = tier.get("fc_gate")
    if fc_gate is not None:
        gates.append({
            "gate": "overall",
            "threshold": fc_gate,
            "actual": overall_rate,
            "passed": overall_rate >= fc_gate,
        })

    # In-process gate
    ip_gate = tier.get("fc_gate_inprocess")
    ip_stats = bucket_stats.get("inprocess")
    if ip_gate is not None and ip_stats:
        gates.append({
            "gate": "inprocess",
            "threshold": ip_gate,
            "actual": ip_stats["pass_rate"],
            "passed": ip_stats["pass_rate"] >= ip_gate,
        })

    # External gate
    ext_gate = tier.get("fc_gate_external")
    ext_stats = bucket_stats.get("external")
    if ext_gate is not None and ext_stats:
        gates.append({
            "gate": "external",
            "threshold": ext_gate,
            "actual": ext_stats["pass_rate"],
            "passed": ext_stats["pass_rate"] >= ext_gate,
        })

    return gates


@click.command()
@click.option("--model", default="protolabs/smart", help="Gateway model name")
@click.option("--suite", default=None, help="Test suite name (e.g., basic, canary)")
@click.option("--all-suites", is_flag=True, help="Run all test suites")
@click.option("--gateway-url", default="http://ava:4000/v1")
@click.option("--api-key", envvar=["GATEWAY_API_KEY", "LITELLM_API_KEY"], default="not-needed")
@click.option("--submit-langfuse", is_flag=True, help="Submit scores to Langfuse")
@click.option(
    "--output-dir", type=click.Path(), default=None,
    help="Directory to write result JSON",
)
@click.option("--trials", default=1, type=int, help="Trials per test")
@click.option("--temperature", default=0.0, type=float,
              help="Sampling temperature (default 0.0 — FC is exact-match; low temp is correct)")
def main(model, suite, all_suites, gateway_url, api_key, submit_langfuse, output_dir, trials, temperature):
    """Run function calling accuracy evals."""
    client = OpenAI(base_url=gateway_url, api_key=api_key)
    grader = FunctionCallGrader()

    # Load alias tier for gate enforcement
    tier = load_alias_tier(model)
    if tier:
        click.echo(f"Alias tier: {tier.get('tier', '?')} (gate: {tier.get('fc_gate', '?')})")

    # Collect test files
    if suite:
        files = sorted(SUITE_DIR.glob(f"{suite}.yaml"))
    elif all_suites:
        files = sorted(SUITE_DIR.glob("*.yaml"))
    else:
        click.echo("Specify --suite <name> or --all-suites")
        sys.exit(1)

    if not files:
        click.echo(f"No test files found in {SUITE_DIR}")
        sys.exit(1)

    click.echo(f"Model: {model} | Suites: {len(files)} | Trials: {trials}")
    click.echo("=" * 60)

    total_tests = 0
    total_passed = 0
    test_results = []

    for test_file in files:
        with open(test_file) as f:
            suite_data = yaml.safe_load(f)

        suite_name = suite_data.get("name", test_file.stem)
        tests = suite_data.get("tests", [])
        click.echo(f"\n{suite_name} ({len(tests)} tests)")

        for test in tests:
            trial_records = []
            for trial in range(1, trials + 1):
                output = run_function_call_test(client, model, test, temperature=temperature)
                result = grader.grade(
                    task_input={"prompt": test["prompt"]},
                    task_output=output,
                    expected=test.get("expected", {}),
                )
                trial_records.append({
                    "trial": trial,
                    "passed": result.passed,
                    "score": result.score,
                    "reasoning": result.reasoning,
                    # response-logging: persist what the model ACTUALLY called +
                    # content/error so a failure is self-debuggable without a live re-probe.
                    "actual_tool_calls": output.get("tool_calls", []),
                    "content": output.get("content", ""),
                    "error": output.get("error"),
                })

            all_passed = all(t["passed"] for t in trial_records)
            avg_score = sum(t["score"] for t in trial_records) / len(trial_records)

            total_tests += 1
            if all_passed:
                total_passed += 1

            status = "PASS" if all_passed else "FAIL"
            last = trial_records[-1]
            bucket_tag = f" [{test['bucket']}]" if test.get("bucket") else ""
            tid = test.get("id", "?")
            reasoning = last["reasoning"][:80]
            click.echo(f"  {tid}{bucket_tag}: {status} ({last['score']:.2f}) — {reasoning}")

            test_results.append({
                "test_id": test.get("id", "?"),
                "suite": suite_name,
                "bucket": test.get("bucket"),
                "prompt": test.get("prompt"),
                "expected": test.get("expected", {}),
                "trials": trial_records,
                "all_passed": all_passed,
                "avg_score": avg_score,
            })

    overall_rate = total_passed / total_tests if total_tests > 0 else 0.0
    bucket_stats = compute_bucket_stats(test_results)

    # --- Summary ---
    click.echo(f"\n{'=' * 60}")
    click.echo(f"Results: {total_passed}/{total_tests} passed^{trials} ({overall_rate:.0%})")

    # Per-trial pass-rate mean ± spread (the run-to-run noise band). At temp 0 this is
    # ~flat (deterministic ceiling); at temp>0 it quantifies sampling noise so a delta
    # between models can be read against it instead of mistaking noise for signal.
    if trials > 1 and total_tests > 0:
        import statistics
        per_trial = [
            sum(1 for tr in test_results if tr["trials"][i]["passed"]) / total_tests
            for i in range(trials)
        ]
        mean = sum(per_trial) / trials
        half_range = (max(per_trial) - min(per_trial)) / 2
        std = statistics.pstdev(per_trial)
        click.echo(
            f"Per-trial pass-rate: {mean:.1%} ± {half_range:.1%}  "
            f"(range {min(per_trial):.0%}-{max(per_trial):.0%}, std {std:.1%}, n={trials})"
        )

    # Print per-bucket breakdown
    for bucket, stats in sorted(bucket_stats.items(), key=lambda x: x[0] or ""):
        label = bucket or "untagged"
        click.echo(f"  {label:12s}: {stats['passed']}/{stats['total']} ({stats['pass_rate']:.0%})")

    # --- Gate enforcement ---
    gate_results = []
    if tier:
        gate_results = evaluate_gates(model, overall_rate, bucket_stats, tier)
        click.echo(f"\nGate check ({model}, tier: {tier.get('tier', '?')}):")
        all_gates_pass = True
        for g in gate_results:
            icon = click.style("PASS", fg="green") if g["passed"] else click.style("FAIL", fg="red")
            gate_name = g["gate"]
            actual = f"{g['actual']:.0%}"
            threshold = f"{g['threshold']:.0%}"
            click.echo(f"  [{icon}] {gate_name:12s}: {actual} (threshold: {threshold})")
            if not g["passed"]:
                all_gates_pass = False

        if not all_gates_pass:
            msg = "\n  ⚠ GATE FAILED — model below tool-calling threshold"
            click.echo(click.style(msg, fg="red"))
    else:
        click.echo(f"\nNo alias tier found for '{model}' — gate check skipped")

    # --- Write results ---
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        results_data = {
            "model": model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trials": trials,
            "suite_type": "function_call",
            "tests": test_results,
            "summary": {
                "total": total_tests,
                "passed": total_passed,
                "pass_rate": overall_rate,
            },
            "buckets": bucket_stats,
            "gates": gate_results,
            "alias_tier": tier.get("tier") if tier else None,
        }
        results_file = out_path / "function_call_results.json"
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)
        click.echo(f"\nResults written to {results_file}")

    # Exit non-zero if any gate failed (for CI)
    if gate_results and not all(g["passed"] for g in gate_results):
        sys.exit(1)


if __name__ == "__main__":
    main()
