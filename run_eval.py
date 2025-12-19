"""Run evaluation tests for data science tutorial scenarios."""

import asyncio
import logging
import sys
import json
import os
from pathlib import Path
from dotenv import load_dotenv
import hud
from hud.agents import OpenAIChatAgent
from env import env

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# Configure logging
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
    force=True,
)

logger = logging.getLogger(__name__)


async def run_evaluation():
    """Run evaluation using tasks from test_tasks.json."""
    print("\n=== Running Data Science Tutorial Evaluation ===")
    
    # 1. Load tasks from JSON
    try:
        with open("test_tasks.json", "r") as f:
            task_data = json.load(f)
    except FileNotFoundError:
        logger.warning("test_tasks.json not found. Creating a default task...")
        # Create a default task for testing
        task_data = [
            {
                "scenario": "compare-wine-quality",
                "args": {}
            }
        ]
    
    # 2. Bind tasks to the local environment
    # Handle both v5 format (with env.name) and simple format
    tasks = []
    for data in task_data:
        scenario = data.get("scenario")
        args = data.get("args", {})
        
        # For v5 format, verify env.name matches our environment
        if "env" in data:
            env_name = data["env"].get("name") if isinstance(data["env"], dict) else data["env"]
            if env_name != "data_science_tutorial":
                logger.warning(f"Task {data.get('id', 'unknown')} has env.name '{env_name}', expected 'data_science_tutorial'")
        
        task = env(scenario, **args)
        tasks.append(task)
    
    print(f"Loaded {len(tasks)} task(s).")

    # 3. Run evaluation with qwen/qwen3-max agent
    async with hud.eval(tasks) as ctx:
        # Get API key from environment variable
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found. Please create a .env file with your API key.\n"
                "See .env.example for reference."
            )
        
        # Create an agent using OpenAIChatAgent with qwen configuration
        agent = OpenAIChatAgent.create(
            model="qwen/qwen3-max",
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        
        print(f"Starting evaluation with agent: {agent.__class__.__name__}")
        print(f"Model: qwen/qwen3-max")
        print(f"Base URL: https://openrouter.ai/api/v1")
        print("-" * 60)
        
        await agent.run(ctx)
        
    print("\n" + "=" * 60)
    print("Evaluation complete.")
    print(f"Total tasks: {len(tasks)}")
    print(f"Results: {ctx.reward if hasattr(ctx, 'reward') else 'N/A'}")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
