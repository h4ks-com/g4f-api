#!/usr/bin/env python3
import asyncio
import sys
from backend.background import test_provider
from backend.dependencies import base_working_providers_map


async def test_top_providers():
    """Test all whitelisted providers"""
    priority_providers = sorted(base_working_providers_map.keys())

    print(f"\n{'=' * 60}")
    print(f"Testing {len(priority_providers)} providers...")
    print(f"{'=' * 60}\n")

    results = {"working": [], "failed": []}

    for name in priority_providers:
        provider = base_working_providers_map[name]
        queue = asyncio.Queue()
        semaphore = asyncio.Semaphore(1)

        print(f"Testing {name}...", end=" ", flush=True)

        try:
            result = await test_provider(provider, queue, semaphore)
            if result:
                print("✓ WORKING")
                results["working"].append(name)
            else:
                print("✗ FAILED")
                results["failed"].append(name)
        except Exception as e:
            print(f"✗ ERROR: {str(e)[:50]}")
            results["failed"].append(name)

    print(f"\n{'=' * 60}")
    print(f"Results:")
    print(f"{'=' * 60}")
    print(f"✓ Working: {len(results['working'])}/{len(priority_providers)}")
    print(f"✗ Failed:  {len(results['failed'])}/{len(priority_providers)}")
    print(f"\nWorking providers: {', '.join(results['working'])}")
    if results["failed"]:
        print(f"Failed providers:  {', '.join(results['failed'])}")
    print(f"{'=' * 60}\n")

    return results


if __name__ == "__main__":
    try:
        results = asyncio.run(test_top_providers())
        sys.exit(0 if len(results["working"]) > 0 else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
