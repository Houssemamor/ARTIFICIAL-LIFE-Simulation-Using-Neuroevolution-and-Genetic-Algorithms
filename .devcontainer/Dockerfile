FROM mcr.microsoft.com/vscode/devcontainers/python:3.11

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    libgl1-mesa-glx \
    libegl1-mesa \
    libxrandr2 \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxcomposite1 \
    libasound2 \
    libxi6 \
    libxtst6

# Set up Python environment
RUN pip install --upgrade pip

# Set non-root user (optional)
# USER vscode

# Set working directory
WORKDIR /workspace

# Copy only requirements to leverage Docker cache
COPY pyproject.toml .
COPY requirements.txt .

# Install dependencies
RUN pip install -e .[dev,debug]

# Copy the rest of the application
COPY . .

# Make sure files are writable
RUN chown -R vscode:vscode /workspace

# Switch to non-root user if desired
# USER vscode