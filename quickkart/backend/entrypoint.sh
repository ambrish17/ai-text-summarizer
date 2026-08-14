#!/bin/sh
set -e

# Apply database migrations
python manage.py migrate --noinput || true

# Create a default superuser if DJANGO_SUPERUSER_EMAIL is provided (optional)
# Start server
python manage.py runserver 0.0.0.0:8000
