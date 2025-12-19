"""data_science_tutorial - HUD Environment with Jupyter Notebook Support"""

import logging
import os
import sys
import subprocess
import asyncio
import signal
import atexit
from pathlib import Path
from typing import Any

from hud import Environment
from hud.tools.jupyter import JupyterTool

# Configure logging to stderr
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
    force=True
)

# Suppress noisy logs
for logger_name in ["tornado", "asyncio", "jupyter_client"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Constants
DATA_PATH = "/app/data"
WINE_QUALITY_PATH = os.path.join(DATA_PATH, "wine_quality")

# Create HUD environment
env = Environment(name="data_science_tutorial")

# Initialize the JupyterTool
# The gateway is started in Docker, so we connect to localhost:8888
jupyter_tool = JupyterTool(url_suffix="localhost:8888", kernel_name="python3")

# Global reference to self-started gateway process (for local testing)
gateway_process = None
gateway_started = False

def cleanup_gateway_sync():
    """Forceful cleanup of background processes."""
    global gateway_process, gateway_started
    if gateway_process:
        try:
            sys.stderr.write("\n[DEBUG] Shutting down Jupyter Gateway (PID: {})\n".format(gateway_process.pid))
            # Kill the process group to ensure all sub-kernels die
            if hasattr(os, 'killpg'):
                try:
                    os.killpg(os.getpgid(gateway_process.pid), signal.SIGKILL)
                except:
                    gateway_process.kill()
            else:
                gateway_process.kill()
            
            gateway_process.wait(timeout=1)
            sys.stderr.write("[DEBUG] Jupyter Gateway terminated.\n")
        except Exception as e:
            sys.stderr.write(f"[DEBUG] Cleanup error: {e}\n")
        finally:
            gateway_process = None
            gateway_started = False

# Register for automatic cleanup on exit
atexit.register(cleanup_gateway_sync)

async def start_gateway_if_needed():
    """Start the Jupyter Kernel Gateway if it's not already running."""
    global gateway_process, gateway_started
    
    if gateway_started:
        if gateway_process and gateway_process.poll() is not None:
            gateway_started = False
        else:
            return

    # 1. Check if healthy (gateway should be running from Docker)
    try:
        await asyncio.wait_for(jupyter_tool._ensure_kernel(), timeout=1.5)
        logger.info("Connected to existing, healthy Jupyter gateway.")
        gateway_started = True
        return
    except Exception:
        logger.info("No healthy Jupyter gateway found. Starting one...")

    # 2. Start gateway in a new process group (for local testing)
    try:
        cmd = [sys.executable, "-m", "jupyter", "kernelgateway", "--KernelGatewayApp.ip=0.0.0.0", "--KernelGatewayApp.port=8888"]
        gateway_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid if hasattr(os, 'setsid') else None
        )
        
        # 3. Wait for connection
        for i in range(15):
            try:
                await asyncio.sleep(1)
                await jupyter_tool._ensure_kernel()
                logger.info("Jupyter gateway started successfully.")
                gateway_started = True
                return
            except Exception:
                if i == 14:
                    raise RuntimeError("Gateway failed to respond after 15 seconds.")
                continue
    except Exception as e:
        logger.error(f"Failed to start gateway: {e}")
        cleanup_gateway_sync()
        raise

@env.initialize
async def init():
    """Environment initialization hook."""
    logger.info("Data science tutorial environment initializing...")
    await start_gateway_if_needed()
    logger.info("Environment initialized successfully.")

@env.shutdown
async def shutdown():
    """Environment shutdown hook."""
    try:
        await jupyter_tool.shutdown()
        logger.info("Jupyter tool shut down successfully.")
    except Exception as e:
        logger.warning(f"Error during shutdown: {e}")

# Register the JupyterTool
env.add_tool(jupyter_tool)

# =============================================================================
# TOOLS - Functions the agent can call
# =============================================================================

@env.tool()
async def execute_python(code: str) -> str:
    """
    Execute Python code in the Jupyter kernel.
    
    Use this tool to run data science operations like:
    - Loading and exploring datasets
    - Performing data analysis
    - Creating visualizations
    - Running machine learning models
    
    Args:
        code: Python code to execute in the Jupyter kernel
        
    Returns:
        The output from executing the code
    """
    await start_gateway_if_needed()
    results = await jupyter_tool(code)
    # Combine all output blocks into a single string
    output = "".join([getattr(b, "text", "") or str(b) for b in results]).strip()
    return output

# =============================================================================
# SCENARIOS - Define prompts and evaluation logic
# =============================================================================

@env.scenario("compare-wine-quality")
async def compare_wine_quality():
    """
    Compare average quality scores between red and white wines.
    
    The agent must calculate and report which wine type has higher average quality.
    """
    red_path = os.path.join(WINE_QUALITY_PATH, "winequality-red.csv")
    white_path = os.path.join(WINE_QUALITY_PATH, "winequality-white.csv")
    
    prompt = f"""You are a data scientist who can analyze datasets through Python code using Jupyter notebooks.

You need to solve the following data analysis question:

### instruction
Compare the average quality scores between red and white wines. Determine which wine type has a higher average quality score and report the exact average quality value for each type.

### dataset_paths
- Red wine dataset: {red_path}
- White wine dataset: {white_path}

### dataset_info
The wine quality dataset contains physicochemical properties of wines and their quality ratings.
Each dataset has the following columns:
- fixed acidity: Concentration of tartaric acid (g/dm³)
- volatile acidity: Amount of acetic acid (affects taste)
- citric acid: Adds freshness and flavor
- residual sugar: Remaining sugar after fermentation
- chlorides: Salt content
- free sulfur dioxide: Unbound SO2
- total sulfur dioxide: Total SO2
- density: Wine density
- pH: Acidity/basicity (0-14 scale)
- sulphates: Wine additives
- alcohol: Alcohol content percentage
- quality: Quality score (0-10 scale, typically 3-9)

### expected_output
Your final answer should clearly state:
1. The average quality score for red wines (as a number)
2. The average quality score for white wines (as a number)
3. Which wine type has higher average quality

The solution can be generated through multiple rounds of interaction. You can do two types of actions:
1. **Data exploration**: Generate Python code to load and explore the datasets. The execution results will be provided to you.
2. **Analysis and calculation**: Generate Python code to calculate the average quality for each wine type and compare them.

Use the `execute_python` tool to run your Python code.

**IMPORTANT**: Provide your final answer with specific numerical values, not just descriptions."""

    agent_response = yield prompt
    
    # Evaluate the response by checking for numerical quality values
    import re
    
    # Extract all numbers (integers and floats)
    numbers = re.findall(r'-?\d+\.\d+|-\d+|\d+', agent_response)
    # Convert to floats, filter valid quality scores (typically 3-9 for wine quality)
    quality_values = []
    for num_str in numbers:
        try:
            val = float(num_str)
            # Wine quality is typically between 0-10, but usually 3-9
            if 0 <= val <= 10:
                quality_values.append(val)
        except ValueError:
            continue
    
    # Check if response contains both red and white wine quality mentions
    response_lower = agent_response.lower()
    has_red = any(word in response_lower for word in ['red', 'red wine'])
    has_white = any(word in response_lower for word in ['white', 'white wine'])
    has_comparison = any(word in response_lower for word in ['higher', 'greater', 'better', 'more', 'compare', 'compared'])
    
    # Should have at least 2 quality values (red and white averages)
    # And they should be reasonable (between 3-9 typically)
    has_sufficient_values = len(quality_values) >= 2
    
    # Reward: 1.0 if all criteria met, 0.0 otherwise
    reward = 1.0 if (has_red and has_white and has_comparison and has_sufficient_values) else 0.0
    yield reward

@env.scenario("find-quality-correlation")
async def find_quality_correlation():
    """
    Find which feature is most correlated with wine quality.
    
    The agent must calculate correlations and identify the feature with highest correlation.
    """
    red_path = os.path.join(WINE_QUALITY_PATH, "winequality-red.csv")
    
    prompt = f"""You are a data scientist who can analyze datasets through Python code using Jupyter notebooks.

You need to solve the following data analysis question:

### instruction
Find which physicochemical property (feature) has the strongest correlation with wine quality. Calculate the correlation coefficient between each feature and the quality score, then identify the feature with the highest absolute correlation value.

### dataset_path
{red_path}

### dataset_info
The red wine quality dataset contains the following features:
- fixed acidity, volatile acidity, citric acid, residual sugar, chlorides
- free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol
- quality: The target variable (quality score 0-10)

### expected_output
Your final answer should clearly state:
1. The feature name with the highest correlation to quality
2. The correlation coefficient value (as a number between -1 and 1)
3. Whether the correlation is positive or negative

The solution can be generated through multiple rounds of interaction:
1. **Data loading and exploration**: Generate Python code to load the dataset and examine its structure.
2. **Correlation calculation**: Generate Python code to calculate correlation coefficients between all features and the quality column.
3. **Result identification**: Identify the feature with the highest absolute correlation.

Use the `execute_python` tool to run your Python code.

**IMPORTANT**: Provide your final answer with the specific feature name and its correlation coefficient value."""

    agent_response = yield prompt
    
    # Evaluate: check for feature name, correlation value, and direction
    import re
    
    # Extract correlation coefficients (numbers between -1 and 1)
    numbers = re.findall(r'-?\d+\.\d+|-\d+|\d+', agent_response)
    correlation_values = []
    for num_str in numbers:
        try:
            val = float(num_str)
            # Correlation coefficients are between -1 and 1
            if -1.0 <= val <= 1.0:
                correlation_values.append(val)
        except ValueError:
            continue
    
    has_correlation_value = len(correlation_values) > 0
    
    # Check for feature names (common features that correlate with quality)
    response_lower = agent_response.lower()
    feature_names = ['alcohol', 'sulphates', 'volatile acidity', 'citric acid', 'residual sugar', 
                     'chlorides', 'density', 'ph', 'fixed acidity', 'sulfur dioxide', 'free sulfur',
                     'total sulfur', 'sulphate']
    has_feature = any(feature.lower() in response_lower for feature in feature_names)
    
    # Check for correlation direction
    has_direction = any(word in response_lower for word in ['positive', 'negative', 'correlated', 'correlation'])
    
    reward = 1.0 if (has_feature and has_correlation_value and has_direction) else 0.0
    yield reward

@env.scenario("calculate-statistics")
async def calculate_statistics(feature: str = "alcohol", dataset: str = "red"):
    """
    Calculate specific statistics for a given feature.
    
    Args:
        feature: The feature to analyze (e.g., "alcohol", "pH", "quality")
        dataset: Which dataset to use ("red" or "white")
    """
    file_path = os.path.join(WINE_QUALITY_PATH, f"winequality-{dataset}.csv")
    
    prompt = f"""You are a data scientist who can analyze datasets through Python code using Jupyter notebooks.

You need to solve the following data analysis question:

### instruction
Calculate and report the following statistics for the '{feature}' feature in the {dataset} wine dataset:
- Mean (average)
- Median
- Standard deviation
- Minimum value
- Maximum value

### dataset_path
{file_path}

### dataset_info
The wine quality dataset contains physicochemical properties of wines.
The '{feature}' column contains numerical values that need to be analyzed.

### expected_output
Your final answer should provide all five statistics as numerical values:
1. Mean: [value]
2. Median: [value]
3. Standard deviation: [value]
4. Minimum: [value]
5. Maximum: [value]

The solution can be generated through multiple rounds:
1. **Data loading**: Generate Python code to load the dataset using pandas.
2. **Statistical calculation**: Generate Python code to calculate the required statistics for the '{feature}' column.

Use the `execute_python` tool to run your Python code.

**IMPORTANT**: Provide your final answer with all five statistics as specific numerical values."""

    agent_response = yield prompt
    
    # Evaluate: check for all 5 statistics
    import re
    
    response_lower = agent_response.lower()
    
    # Extract all numbers
    numbers = re.findall(r'-?\d+\.\d+|-\d+|\d+', agent_response)
    numeric_values = []
    for num_str in numbers:
        try:
            numeric_values.append(float(num_str))
        except ValueError:
            continue
    
    # Check for each statistic keyword
    has_mean = any(word in response_lower for word in ['mean', 'average', 'avg'])
    has_median = 'median' in response_lower
    has_std = any(word in response_lower for word in ['std', 'standard deviation', 'deviation', 'stdev'])
    has_min = any(word in response_lower for word in ['min', 'minimum', 'lowest', 'smallest'])
    has_max = any(word in response_lower for word in ['max', 'maximum', 'highest', 'largest'])
    
    # Should have at least 5 numeric values (one for each statistic)
    has_sufficient_numbers = len(numeric_values) >= 5
    
    # Also check that the feature name is mentioned
    has_feature_mention = feature.lower() in response_lower
    
    reward = 1.0 if (has_mean and has_median and has_std and has_min and has_max and 
                     has_sufficient_numbers and has_feature_mention) else 0.0
    yield reward

@env.scenario("identify-high-quality-wines")
async def identify_high_quality_wines(threshold: float = 7.0, dataset: str = "red"):
    """
    Identify wines above a quality threshold and calculate percentage.
    
    Args:
        threshold: Quality score threshold (default 7.0)
        dataset: Which dataset to use ("red" or "white")
    """
    file_path = os.path.join(WINE_QUALITY_PATH, f"winequality-{dataset}.csv")
    
    prompt = f"""You are a data scientist who can analyze datasets through Python code using Jupyter notebooks.

You need to solve the following data analysis question:

### instruction
Identify how many wines in the {dataset} wine dataset have a quality score greater than or equal to {threshold}, and calculate what percentage of the total dataset this represents.

### dataset_path
{file_path}

### dataset_info
The wine quality dataset contains wine samples with a 'quality' column that ranges from 0 to 10.
You need to filter wines where quality >= {threshold} and calculate the percentage.

### expected_output
Your final answer should provide:
1. The number of wines with quality >= {threshold} (as an integer)
2. The total number of wines in the dataset (as an integer)
3. The percentage of high-quality wines (as a number with 2 decimal places)

The solution can be generated through multiple rounds:
1. **Data loading**: Generate Python code to load the dataset.
2. **Filtering and counting**: Generate Python code to filter wines with quality >= {threshold} and count them.
3. **Percentage calculation**: Calculate the percentage of high-quality wines.

Use the `execute_python` tool to run your Python code.

**IMPORTANT**: Provide your final answer with specific numbers: count of high-quality wines, total count, and percentage."""

    agent_response = yield prompt
    
    # Evaluate: check for count, total, and percentage
    import re
    
    response_lower = agent_response.lower()
    
    # Extract integers (for counts) and floats (for percentage)
    integers = re.findall(r'\d+', agent_response)
    decimals = re.findall(r'\d+\.\d+', agent_response)
    
    # Convert to numbers
    int_values = [int(x) for x in integers if x.isdigit()]
    float_values = [float(x) for x in decimals]
    
    # Check for keywords
    has_count = any(word in response_lower for word in ['count', 'number', 'wines', 'samples'])
    has_total = any(word in response_lower for word in ['total', 'all wines', 'dataset', 'all samples'])
    has_percentage = any(word in response_lower for word in ['percent', '%', 'percentage', 'pct'])
    has_threshold_mention = (str(threshold) in agent_response or 
                            f">= {threshold}" in agent_response or
                            f">={threshold}" in agent_response or
                            f"greater than or equal to {threshold}" in response_lower)
    
    # Should have at least 2 integers (count and total) and at least 1 number for percentage
    # Percentage can be integer (like "15%") or decimal (like "15.5%")
    has_sufficient_numbers = len(int_values) >= 2 and (len(float_values) >= 1 or len(int_values) >= 3)
    
    # Check that percentage is reasonable (0-100)
    has_valid_percentage = any(0 <= val <= 100 for val in float_values) or any(0 <= val <= 100 for val in int_values)
    
    reward = 1.0 if (has_count and has_total and has_percentage and has_threshold_mention and 
                     has_sufficient_numbers and has_valid_percentage) else 0.0
    yield reward

# =============================================================================
# MAIN - Run the environment
# =============================================================================

if __name__ == "__main__":
    try:
        env.run(transport="stdio")
    finally:
        # Manually trigger cleanup and force exit
        cleanup_gateway_sync()
        sys.stderr.write("[DEBUG] Environment process exiting now.\n")
        os._exit(0)
