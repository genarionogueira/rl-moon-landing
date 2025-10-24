FROM python:3.10-slim

# Install minimal system dependencies for rendering videos
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    swig \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry

# Create working directory
WORKDIR /app

# Copy Poetry files and install dependencies
COPY pyproject.toml /app/
# If you later add a poetry.lock, copy it too for better caching:
# COPY poetry.lock /app/

# Configure Poetry to install into the system environment (no venv)
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-interaction --no-ansi --no-root

# Copy application code (LunarLander agent only)
COPY agent_class.py /app/
COPY train_agent.py /app/
COPY run_agent.py /app/

# Ensure output directory exists for artifacts and set OUTPUT_DIR
ENV OUTPUT_DIR=/app/output
RUN mkdir -p ${OUTPUT_DIR}
ENV XDG_RUNTIME_DIR=/tmp
ENV SDL_VIDEODRIVER=dummy

CMD ["poetry", "run", "python", "demo.py"]