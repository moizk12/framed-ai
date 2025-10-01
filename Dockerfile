# Dockerfile

# ---- Base Stage ----
# Use a specific, slim Python version for a smaller, more secure base.
FROM python:3.11-slim as base

# Set environment variables for best practices
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Set a non-root home for Hugging Face cache to avoid permission issues
ENV HF_HOME=/home/appuser/.cache/huggingface

# Install system dependencies required for libraries like OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user and group
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Switch to the non-root user
USER appuser

# Set the working directory
WORKDIR /home/appuser/app


# ---- Builder Stage ----
# This stage installs dependencies
FROM base as builder

# Copy only the requirements file to leverage Docker layer caching
COPY --chown=appuser:appgroup requirements.txt.

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt


# ---- Final Stage ----
# This is the final, lean image for production
FROM base as final

# Copy installed packages from the builder stage
COPY --from=builder /home/appuser/.local /home/appuser/.local

# Add the local user's bin to the PATH
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy the pre-downloaded model weights and the rest of the application code
COPY --chown=appuser:appgroup yolov8n.pt.
COPY --chown=appuser:appgroup..

# Expose the port the app will run on
EXPOSE 7860

# Command to run the application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "app:app"]