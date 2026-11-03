"""Build-order step 3: measure LLM time-to-first-token from this network.

This number decides whether the sub-1.5s end-to-end target is comfortable or a
fight (see ARCHITECTURE.md latency budget). If p95 > 1s, pick a fallback
provider now, before integrating.

    DEEPSEEK_API_KEY=... or OPENROUTER_API_KEY=... uv run python scripts/measure_ttft.py [N]
"""

import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import AsyncOpenAI  # noqa: E402

from llm_provider import (  # noqa: E402
    llm_api_key,
    llm_base_url,
    llm_config_errors,
    llm_model,
    resolve_llm_provider,
)


async def one_request(client: AsyncOpenAI, model: str) -> float:
    t0 = time.monotonic()
    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say the word ready and nothing else."}],
        stream=True,
        max_tokens=10,
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            ttft = time.monotonic() - t0
            await stream.close()
            return ttft
    return time.monotonic() - t0


async def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    provider = resolve_llm_provider()
    errors = llm_config_errors(provider)
    if errors:
        raise SystemExit("; ".join(errors))
    client = AsyncOpenAI(
        api_key=llm_api_key(provider),
        base_url=llm_base_url(provider),
    )
    model = llm_model(provider)
    results = []
    for i in range(n):
        try:
            ttft = await one_request(client, model)
            results.append(ttft)
            print(f"{i + 1:3d}: {ttft * 1000:.0f}ms")
        except Exception as e:
            print(f"{i + 1:3d}: ERROR {e}")
        await asyncio.sleep(0.5)

    if results:
        results.sort()
        p50 = statistics.median(results)
        p95 = results[min(len(results) - 1, int(len(results) * 0.95))]
        print(f"\nmodel={model} n={len(results)}")
        print(f"p50 TTFT: {p50 * 1000:.0f}ms")
        print(f"p95 TTFT: {p95 * 1000:.0f}ms")
        print(f"min/max:  {results[0] * 1000:.0f}ms / {results[-1] * 1000:.0f}ms")
        if p95 > 1.0:
            print("WARNING: p95 > 1s — consider a low-TTFT fallback provider")


if __name__ == "__main__":
    asyncio.run(main())
