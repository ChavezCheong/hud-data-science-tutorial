"""Local test script for data science tutorial scenarios."""

import asyncio
import logging
import sys
import hud
from env import env

# Configure logging
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
    force=True,
)

logger = logging.getLogger(__name__)


async def test_compare_wine_quality():
    """Test the compare-wine-quality scenario."""
    print("\n=== Testing Compare Wine Quality Scenario ===")
    
    # Create the task instance from the scenario
    task = env("compare-wine-quality")
    
    try:
        async with hud.eval(task) as ctx:
            print(f"Prompt: {ctx.prompt[:200]}...")
            print("\nThis scenario requires the agent to:")
            print("1. Load both red and white wine datasets")
            print("2. Calculate average quality for each")
            print("3. Compare and report which is higher")
            print("\nTo run with an agent, use: python run_eval.py")
            
        print("\nStandalone tool test:")
        async with env:
            # Test that we can execute Python code
            result = await env.call_tool("execute_python", code="import pandas as pd; print('Pandas version:', pd.__version__)")
            print(f"Tool Result: {str(result)}")
            
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


async def test_find_correlation():
    """Test the find-quality-correlation scenario."""
    print("\n=== Testing Find Quality Correlation Scenario ===")
    
    task = env("find-quality-correlation")
    
    try:
        async with hud.eval(task) as ctx:
            print(f"Prompt: {ctx.prompt[:200]}...")
            print("\nThis scenario requires the agent to:")
            print("1. Load the red wine dataset")
            print("2. Calculate correlations between features and quality")
            print("3. Identify the feature with highest correlation")
            
    except Exception as e:
        print(f"Test failed with error: {e}")


async def main():
    """Run all local tests."""
    await test_compare_wine_quality()
    await test_find_correlation()
    print("\n" + "=" * 60)
    print("Local tests complete. Use 'python run_eval.py' to run full evaluation with agent.")


if __name__ == "__main__":
    asyncio.run(main())
