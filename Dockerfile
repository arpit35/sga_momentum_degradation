# Python image as the base
FROM python:3.10-bullseye

# Working directory inside the container
WORKDIR /sga_momentum_degradation

# Copy the requirements file to the container
COPY requirements.txt /sga_momentum_degradation/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . /sga_momentum_degradation

# Specify the command to run the main.py script
CMD ["python", "data_loader.py"]
