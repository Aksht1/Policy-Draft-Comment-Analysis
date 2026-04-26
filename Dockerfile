# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8502
ENV STREAMLIT_SERVER_HEADLESS=true

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --timeout=100 -r requirements.txt

# Copy pre-trained models
COPY tfidf_vectorizer.pkl .
COPY lr_model.pkl .
COPY svm_model.pkl .
COPY model_metrics.pkl .

# Copy the application files
COPY app.py .
COPY pr.py .
COPY final_clean_dataset.csv .

# Copy Streamlit config
COPY .streamlit/config.toml /root/.streamlit/config.toml

# Copy entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose the port that Streamlit runs on
EXPOSE 8502

# Run the entrypoint script
CMD ["/app/entrypoint.sh"]