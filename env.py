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
    
    IMPORTANT: The 'code' parameter must be a STRING containing Python code, not a dictionary or object.
    
    Use this tool to run data science operations like:
    - Loading and exploring datasets
    - Performing data analysis
    - Creating visualizations
    - Running machine learning models
    
    Args:
        code (str): Python code to execute as a single string. The entire code block should be passed as one string.
        
    Returns:
        str: The output from executing the code
        
    Example:
        Correct usage:
        execute_python(code="import pandas as pd\ndf = pd.read_csv('file.csv')\nprint(df.head())")
        
        Wrong usage (DO NOT DO THIS):
        execute_python(code={"file": "data.csv"})  # This is incorrect!
    """
    # Validate that code is actually a string
    if not isinstance(code, str):
        error_msg = (
            f"ERROR: The 'code' parameter must be a STRING, but received {type(code).__name__}. "
            f"Please pass your Python code as a string. Example: execute_python(code='import pandas as pd')"
        )
        logger.error(error_msg)
        return error_msg
    
    # Check if code is empty
    if not code.strip():
        return "ERROR: The 'code' parameter cannot be empty. Please provide Python code to execute."
    
    await start_gateway_if_needed()
    try:
        results = await jupyter_tool(code)
        # Combine all output blocks into a single string
        output = "".join([getattr(b, "text", "") or str(b) for b in results]).strip()
        return output
    except Exception as e:
        error_msg = f"Error executing code: {str(e)}"
        logger.error(error_msg)
        return error_msg

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
The CSV files are located at these EXACT paths:
- Red wine dataset: `{red_path}`
- White wine dataset: `{white_path}`

**IMPORTANT - File Path Usage:**
- These are ABSOLUTE paths - use them exactly as shown
- Use these paths directly in pd.read_csv() - no need to search for files
- Example code: `df_red = pd.read_csv('{red_path}')`
- Example code: `df_white = pd.read_csv('{white_path}')`

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
1. **Data exploration**: Generate Python code to load and explore the datasets using the exact paths: `pd.read_csv('{red_path}')` and `pd.read_csv('{white_path}')`. The execution results will be provided to you.
2. **Analysis and calculation**: Generate Python code to calculate the average quality for each wine type and compare them.

### tool_usage
Use the `execute_python` tool to run your Python code.

**CRITICAL - Tool Calling Format:**
- The `code` parameter MUST be a STRING containing your Python code
- Pass the entire code block as a single string value
- Use newlines (\n) or triple quotes for multi-line code
- Example: execute_python(code="import pandas as pd\ndf = pd.read_csv('file.csv')\nprint(df.head())")
- DO NOT pass a dictionary or object - only a string!

**IMPORTANT**: 
1. After running your code and getting results, you MUST provide a final text answer summarizing your findings
2. Your final answer should include specific numerical values, not just descriptions
3. Make sure to submit your final answer - do not just show tool outputs"""

    agent_response = yield prompt
    
    # Evaluate the response by checking for numerical quality values
    import re
    
    # Handle None or empty response
    if agent_response is None:
        logger.warning("Agent response is None - agent may not have submitted final answer")
        yield 0.0
        return
    
    # Convert to string if needed
    response_text = str(agent_response) if not isinstance(agent_response, str) else agent_response
    
    if not response_text or len(response_text.strip()) == 0:
        logger.warning("Agent response is empty")
        yield 0.0
        return
    
    # Extract all numbers (integers and floats)
    numbers = re.findall(r'-?\d+\.\d+|-\d+|\d+', response_text)
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
    response_lower = response_text.lower()
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
The CSV file is located at this EXACT path:
`{red_path}`

**IMPORTANT - File Path Usage:**
- This is an ABSOLUTE path - use it exactly as shown
- Use this path directly in pd.read_csv() - no need to search for files
- Example code: `df = pd.read_csv('{red_path}')`

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
1. **Data loading and exploration**: Generate Python code to load the dataset using the exact path: `pd.read_csv('{red_path}')` and examine its structure.
2. **Correlation calculation**: Generate Python code to calculate correlation coefficients between all features and the quality column.
3. **Result identification**: Identify the feature with the highest absolute correlation.

### tool_usage
Use the `execute_python` tool to run your Python code. 

**CRITICAL - Tool Calling Format:**
- The `code` parameter MUST be a STRING containing your Python code
- Pass the entire code block as a single string value
- Use newlines (\n) or triple quotes for multi-line code
- Example: execute_python(code="import pandas as pd\ndf = pd.read_csv('file.csv')\nprint(df.head())")
- DO NOT pass a dictionary or object - only a string!

**IMPORTANT**: Provide your final answer with the specific feature name and its correlation coefficient value."""

    agent_response = yield prompt
    
    # Evaluate: check for feature name, correlation value, and direction
    import re
    
    # Handle None or empty response
    if agent_response is None:
        logger.warning("Agent response is None - agent may not have submitted final answer")
        yield 0.0
        return
    
    response_text = str(agent_response) if not isinstance(agent_response, str) else agent_response
    if not response_text or len(response_text.strip()) == 0:
        logger.warning("Agent response is empty")
        yield 0.0
        return
    
    # Extract correlation coefficients (numbers between -1 and 1)
    numbers = re.findall(r'-?\d+\.\d+|-\d+|\d+', response_text)
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
    response_lower = response_text.lower()
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
The CSV file is located at this EXACT path:
`{file_path}`

**IMPORTANT - File Path Usage:**
- This is an ABSOLUTE path - use it exactly as shown
- Use this path directly in pd.read_csv() - no need to search for files
- Example code: `df = pd.read_csv('{file_path}')`

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
1. **Data loading**: Generate Python code to load the dataset using pandas with the exact path: `pd.read_csv('{file_path}')`
2. **Statistical calculation**: Generate Python code to calculate the required statistics for the '{feature}' column.

### tool_usage
Use the `execute_python` tool to run your Python code.

**CRITICAL - Tool Calling Format:**
- The `code` parameter MUST be a STRING containing your Python code
- Pass the entire code block as a single string value
- Use newlines (\n) or triple quotes for multi-line code
- Example: execute_python(code="import pandas as pd\ndf = pd.read_csv('file.csv')\nprint(df.describe())")
- DO NOT pass a dictionary or object - only a string!

**IMPORTANT**: Provide your final answer with all five statistics as specific numerical values."""

    agent_response = yield prompt
    
    # Evaluate: check for all 5 statistics
    import re
    
    # Handle None or empty response
    if agent_response is None:
        logger.warning("Agent response is None - agent may not have submitted final answer")
        yield 0.0
        return
    
    response_text = str(agent_response) if not isinstance(agent_response, str) else agent_response
    if not response_text or len(response_text.strip()) == 0:
        logger.warning("Agent response is empty")
        yield 0.0
        return
    
    response_lower = response_text.lower()
    
    # Extract all numbers
    numbers = re.findall(r'-?\d+\.\d+|-\d+|\d+', response_text)
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
The CSV file is located at this EXACT path:
`{file_path}`

**IMPORTANT - File Path Usage:**
- This is an ABSOLUTE path - use it exactly as shown
- Use this path directly in pd.read_csv() - no need to search for files
- Example code: `df = pd.read_csv('{file_path}')`

### dataset_info
The wine quality dataset contains wine samples with a 'quality' column that ranges from 0 to 10.
You need to filter wines where quality >= {threshold} and calculate the percentage.

### expected_output
Your final answer should provide:
1. The number of wines with quality >= {threshold} (as an integer)
2. The total number of wines in the dataset (as an integer)
3. The percentage of high-quality wines (as a number with 2 decimal places)

The solution can be generated through multiple rounds:
1. **Data loading**: Generate Python code to load the dataset using the exact path: `pd.read_csv('{file_path}')`
2. **Filtering and counting**: Generate Python code to filter wines with quality >= {threshold} and count them.
3. **Percentage calculation**: Calculate the percentage of high-quality wines.

### tool_usage
Use the `execute_python` tool to run your Python code.

**CRITICAL - Tool Calling Format:**
- The `code` parameter MUST be a STRING containing your Python code
- Pass the entire code block as a single string value
- Use newlines (\n) or triple quotes for multi-line code
- Example: execute_python(code="import pandas as pd\ndf = pd.read_csv('file.csv')\nhigh_quality = df[df['quality'] >= 7]\nprint(len(high_quality))")
- DO NOT pass a dictionary or object - only a string!

**IMPORTANT**: Provide your final answer with specific numbers: count of high-quality wines, total count, and percentage."""

    agent_response = yield prompt
    
    # Evaluate: check for count, total, and percentage
    import re
    
    # Handle None or empty response
    if agent_response is None:
        logger.warning("Agent response is None - agent may not have submitted final answer")
        yield 0.0
        return
    
    response_text = str(agent_response) if not isinstance(agent_response, str) else agent_response
    if not response_text or len(response_text.strip()) == 0:
        logger.warning("Agent response is empty")
        yield 0.0
        return
    
    response_lower = response_text.lower()
    
    # Extract integers (for counts) and floats (for percentage)
    integers = re.findall(r'\d+', response_text)
    decimals = re.findall(r'\d+\.\d+', response_text)
    
    # Convert to numbers
    int_values = [int(x) for x in integers if x.isdigit()]
    float_values = [float(x) for x in decimals]
    
    # Check for keywords
    has_count = any(word in response_lower for word in ['count', 'number', 'wines', 'samples'])
    has_total = any(word in response_lower for word in ['total', 'all wines', 'dataset', 'all samples'])
    has_percentage = any(word in response_lower for word in ['percent', '%', 'percentage', 'pct'])
    has_threshold_mention = (str(threshold) in response_text or 
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

@env.scenario("predict-wine-quality-ml")
async def predict_wine_quality_ml(model_type: str = "random_forest", test_size: float = 0.2):
    """
    Complex machine learning scenario: Build a model to predict wine quality.
    
    This scenario requires multiple iterations:
    1. Load and combine datasets
    2. Data preprocessing and feature engineering
    3. Train-test split
    4. Model training
    5. Model evaluation
    6. Report metrics
    
    Args:
        model_type: Type of model to use ("random_forest", "logistic_regression", or "gradient_boosting")
        test_size: Proportion of data for testing (default 0.2)
    """
    red_path = os.path.join(WINE_QUALITY_PATH, "winequality-red.csv")
    white_path = os.path.join(WINE_QUALITY_PATH, "winequality-white.csv")
    
    prompt = f"""You are a data scientist who can analyze datasets and build machine learning models through Python code using Jupyter notebooks.

You need to solve the following machine learning task:

### instruction
Build a machine learning model to predict wine quality using the physicochemical properties. This is a multi-step process that requires careful data preparation, feature engineering, model training, and evaluation.

### dataset_paths
The CSV files are located at these EXACT paths:
- Red wine dataset: `{red_path}`
- White wine dataset: `{white_path}`

**IMPORTANT - File Path Usage:**
- These are ABSOLUTE paths - use them exactly as shown
- Use these paths directly in pd.read_csv() - no need to search for files
- Example code for loading: `df_red = pd.read_csv('{red_path}')`
- Example code for loading: `df_white = pd.read_csv('{white_path}')`
- DO NOT try to search directories or list files - use the paths directly!

### task_requirements
You must complete the following steps:

**Step 1: Data Loading and Combination**
- Load both red and white wine datasets using pandas with the exact paths provided above
- Use: `df_red = pd.read_csv('{red_path}')` and `df_white = pd.read_csv('{white_path}')`
- Add a 'wine_type' column to distinguish between red (value: 'red') and white (value: 'white') wines
- Combine both datasets into a single DataFrame using pd.concat()
- Display the shape and basic info of the combined dataset

**Step 2: Data Preprocessing**
- Check for missing values and handle them if any exist
- Examine the distribution of the target variable (quality)
- Identify and handle any outliers if necessary
- Display summary statistics

**Step 3: Feature Engineering**
- Separate features (X) from target (y = quality column)
- The features should include all physicochemical properties but exclude 'quality'
- Optionally create new features if beneficial (e.g., ratio features, interactions)
- Display the feature names and their count

**Step 4: Train-Test Split**
- Split the data into training and testing sets
- Use a test size of {test_size} (i.e., {int(test_size * 100)}% for testing)
- Use random_state=42 for reproducibility
- Display the shapes of training and testing sets

**Step 5: Model Training**
- Import and initialize a {model_type} model from scikit-learn
  - If "random_forest": use RandomForestClassifier with n_estimators=100, random_state=42
  - If "logistic_regression": use LogisticRegression with max_iter=1000, random_state=42
  - If "gradient_boosting": use GradientBoostingClassifier with n_estimators=100, random_state=42
- Train the model on the training data
- Print a confirmation message that training is complete

**Step 6: Model Evaluation**
- Make predictions on the test set
- Calculate and report the following metrics:
  - Accuracy score
  - Classification report (precision, recall, F1-score)
  - Confusion matrix
- Display these metrics clearly

**Step 7: Feature Importance (if applicable)**
- If using RandomForest or GradientBoosting, extract and display feature importances
- Identify the top 3 most important features for predicting quality

### expected_output
Your final answer should clearly report:
1. The combined dataset shape (rows, columns)
2. The model type used: {model_type}
3. Training set size and test set size
4. Model accuracy on test set (as a decimal between 0 and 1, or as a percentage)
5. Top 3 most important features (if applicable) or confirmation that feature importance was analyzed
6. A brief interpretation of the results

### important_notes
- This task requires multiple rounds of code execution
- You may need to iterate and refine your code if errors occur
### tool_usage
Use the `execute_python` tool for each step, building upon previous results.

**CRITICAL - Tool Calling Format:**
- The `code` parameter MUST be a STRING containing your Python code
- Pass the entire code block as a single string value
- Use newlines (\n) or triple quotes for multi-line code
- Example: execute_python(code="import pandas as pd\nimport numpy as np\ndf = pd.read_csv('file.csv')")
- DO NOT pass a dictionary or object - only a string!
- Import necessary libraries (pandas, numpy, sklearn) at the beginning
- Ensure all code is properly structured and handles edge cases
- The quality column contains integer values (typically 3-9)

### evaluation_criteria
Your solution will be evaluated based on:
- Successful data loading and combination
- Proper preprocessing and feature engineering
- Correct train-test split
- Successful model training
- Complete evaluation metrics
- Clear reporting of results

Use the `execute_python` tool to execute your code step by step. You may need multiple tool calls to complete this task.

**CRITICAL - Tool Calling Format:**
- The `code` parameter MUST be a STRING containing your Python code
- Pass the entire code block as a single string value
- Use newlines (\n) or triple quotes for multi-line code
- Example: execute_python(code="import pandas as pd\ndf = pd.read_csv('file.csv')\nprint(df.shape)")
- DO NOT pass a dictionary or object - only a string!"""

    agent_response = yield prompt
    
    # Evaluate the response - check for all required components
    import re
    
    # Handle None or empty response
    if agent_response is None:
        logger.warning("Agent response is None - agent may not have submitted final answer")
        yield 0.0
        return
    
    response_text = str(agent_response) if not isinstance(agent_response, str) else agent_response
    if not response_text or len(response_text.strip()) == 0:
        logger.warning("Agent response is empty")
        yield 0.0
        return
    
    response_lower = response_text.lower()
    
    # Check for data loading and combination
    has_combined = any(word in response_lower for word in ['combined', 'merge', 'concatenate', 'concat'])
    has_wine_type = 'wine_type' in response_lower or 'wine type' in response_lower
    
    # Check for preprocessing
    has_preprocessing = any(word in response_lower for word in ['preprocess', 'missing', 'null', 'outlier', 'clean'])
    
    # Check for feature engineering
    has_features = any(word in response_lower for word in ['feature', 'x_train', 'x_test', 'y_train', 'y_test'])
    has_split = any(word in response_lower for word in ['train_test_split', 'split', 'train', 'test'])
    
    # Check for model training
    model_keywords = {
        'random_forest': ['randomforest', 'random forest', 'rf'],
        'logistic_regression': ['logistic', 'logisticregression', 'lr'],
        'gradient_boosting': ['gradientboosting', 'gradient boosting', 'gb']
    }
    has_model = any(keyword in response_lower for keyword in model_keywords.get(model_type, []))
    has_trained = any(word in response_lower for word in ['train', 'fit', 'trained', 'training'])
    
    # Check for evaluation metrics
    has_accuracy = any(word in response_lower for word in ['accuracy', 'acc'])
    has_metrics = any(word in response_lower for word in ['precision', 'recall', 'f1', 'f1-score', 'classification report', 'confusion matrix'])
    
    # Extract accuracy value (should be between 0 and 1, or as percentage)
    accuracy_pattern = r'accuracy[:\s]+(\d+\.?\d*)'
    accuracy_matches = re.findall(accuracy_pattern, response_lower)
    has_accuracy_value = False
    if accuracy_matches:
        try:
            acc_val = float(accuracy_matches[0])
            # Accept as decimal (0-1) or percentage (0-100)
            if 0 <= acc_val <= 1 or (0 <= acc_val <= 100 and '%' in agent_response):
                has_accuracy_value = True
        except:
            pass
    
    # Check for feature importance
    has_importance = any(word in response_lower for word in ['importance', 'important feature', 'top feature'])
    
    # Check for dataset shape mention
    has_shape = any(word in response_lower for word in ['shape', 'rows', 'columns', 'samples'])
    
    # Calculate reward based on completeness
    # This is a complex task, so we'll give partial credit for progress
    required_components = [
        has_combined or has_wine_type,  # Data combination
        has_preprocessing,  # Preprocessing
        has_features and has_split,  # Feature engineering and split
        has_model and has_trained,  # Model training
        has_accuracy and has_metrics,  # Evaluation
        has_shape  # Dataset info
    ]
    
    # Count how many components are present
    components_present = sum(required_components)
    
    # Full reward if most components are present and accuracy is reported
    if components_present >= 5 and has_accuracy_value and has_importance:
        reward = 1.0
    elif components_present >= 4 and has_accuracy:
        reward = 0.7  # Partial credit
    elif components_present >= 3:
        reward = 0.4  # Some progress
    else:
        reward = 0.0
    
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
