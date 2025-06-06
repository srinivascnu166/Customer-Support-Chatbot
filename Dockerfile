FROM python:3.9-slim

# Set work directory
WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip \
 && pip install -r requirements.txt
 
#to avoid error while unning streamlit app
ENV STREAMLIT_SERVER_RUN_ON_SAVE=false
# Expose port if using a web server (e.g., Streamlit or FastAPI)
EXPOSE 8501

# Run your app
CMD ["streamlit", "run", "app.py"]
