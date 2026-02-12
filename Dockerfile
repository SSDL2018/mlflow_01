# 1️⃣ Base image
FROM python:3.12-slim

# 2️⃣ Set working directory
WORKDIR /app

# 3️⃣ Copy requirements first for caching
COPY requirements.txt .

# 4️⃣ Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5️⃣ Copy the project
COPY . .

# 6️⃣ Expose port
EXPOSE 8000

# 7️⃣ Run FastAPI app
CMD ["uvicorn", "src.fastAPI.app:app", "--host", "0.0.0.0", "--port", "8000"]

