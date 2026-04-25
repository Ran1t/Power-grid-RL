FROM python:3.11-slim

# HuggingFace Spaces runs as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

# Install server dependencies
COPY --chown=user powergrid_env/server/requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the full environment package
COPY --chown=user powergrid_env/ powergrid_env/

ENV PROFILE_NOISE=0.05
ENV PYTHONPATH=/home/user/app

EXPOSE 7860

# HuggingFace Spaces expects port 7860
CMD ["uvicorn", "powergrid_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
