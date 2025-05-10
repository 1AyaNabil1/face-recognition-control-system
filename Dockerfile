# Use an official Python image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port (optional but good practice)
EXPOSE 5000

# Run the app using Gunicorn
CMD ["gunicorn", "api.app:app", "--bind", "0.0.0.0:5000"]
