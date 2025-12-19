# Testing Docker Build

## Step 1: Build the Docker Image

Navigate to the `hud-data-science-tutorial` directory and build:

```bash
cd /home/chavezcheong/AI/jupyter-agent/hud-data-science-tutorial
docker build -f Dockerfile.hud -t data-science-tutorial:test .
```

**Important:** Don't forget the `.` at the end - it specifies the build context (current directory).

**Alternative one-liner (from any directory):**
```bash
docker build -f /home/chavezcheong/AI/jupyter-agent/hud-data-science-tutorial/Dockerfile.hud -t data-science-tutorial:test /home/chavezcheong/AI/jupyter-agent/hud-data-science-tutorial
```

This will:
- Install all Python dependencies from `pyproject.toml`
- Copy `env.py` into the container
- Set up the environment

**Expected output:** Build should complete successfully with no errors.

**Note:** If you see a buildx warning, it's not critical - the build will still work with the legacy builder.

## Step 2: Test Container Startup

Run the container and check if it starts correctly:

```bash
docker run --rm -it data-science-tutorial:test
```

**What to check:**
- Container should start without immediate errors
- You should see logs indicating the Jupyter kernel gateway is starting
- The environment should wait for stdio input (it will appear to hang, which is normal for stdio mode)

Press `Ctrl+C` to stop the container.

## Step 3: Verify Jupyter Kernel Gateway is Running

Run the container in the background and check if port 8888 is accessible:

```bash
# Start container in background
docker run -d --name test-jupyter -p 8888:8888 data-science-tutorial:test

# Wait a few seconds for startup
sleep 6

# Check if kernel gateway is responding
curl http://localhost:8888/api/kernelspecs

# Check container logs
docker logs test-jupyter

# Cleanup
docker stop test-jupyter
docker rm test-jupyter
```

**Expected output:**
- `curl` should return JSON with kernel specs (or at least not a connection error)
- Logs should show Jupyter kernel gateway starting successfully

## Step 4: Test with HUD CLI (if you have HUD set up)

If you have the HUD CLI installed, you can test the environment more thoroughly:

```bash
# Build and tag for HUD
docker build -f Dockerfile.hud -t your-org/data-science-tutorial:latest .

# Test locally with HUD
hud run your-org/data-science-tutorial:latest --local
```

## Step 5: Quick Build Validation

For a quick check that dependencies install correctly:

```bash
docker build -f Dockerfile.hud -t data-science-tutorial:test . 2>&1 | grep -E "(ERROR|error|failed|Successfully)"
```

## Troubleshooting

### Build fails with dependency errors
- Check that `pyproject.toml` has all required dependencies
- Verify Python version compatibility (>=3.11)

### Container exits immediately
- Check logs: `docker logs <container-id>`
- Verify `env.py` has `env.run(transport="stdio")` in the `__main__` block

### Jupyter gateway not accessible
- Check if port 8888 is already in use: `lsof -i :8888`
- Verify the gateway starts in logs: `docker logs <container-id> | grep -i jupyter`
- Try accessing from inside container: `docker exec -it <container-id> curl http://localhost:8888/api/kernelspecs`

### Import errors when running
- Make sure all dependencies are in `pyproject.toml`
- Check that `PYTHONPATH=/app` is set in Dockerfile
