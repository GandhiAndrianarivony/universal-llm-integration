FROM python:3.12.7-slim-bullseye


# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates
# Download the latest installer
ADD https://astral.sh/uv/0.5.4/install.sh /uv-installer.sh
# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh
# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

ENV UV_LINK_MODE=copy

WORKDIR /apps

ENV PIP_DISABLE_PIP_VERSION CHECK 1
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY pyproject.toml uv.lock ./
RUN uv sync

# Create a non-root user
RUN useradd --create-home qa_rag
USER qa_rag

# Place executables in the environment at the front of the path
ENV PATH="/apps/.venv/bin:$PATH"
EXPOSE 8501

CMD ["uv","run", "streamlit", "run", "apps/app.py"]