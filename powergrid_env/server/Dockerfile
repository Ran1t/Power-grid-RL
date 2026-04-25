FROM python:3.11-slim

WORKDIR /app

# Install server dependencies
COPY server/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment code
COPY models.py models.py
COPY server/ server/

# Make server a package
RUN touch server/__init__.py

ENV PROFILE_NOISE=0.05
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
