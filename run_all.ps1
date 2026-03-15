Write-Host "Starting FastAPI Backend on port 8000..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd d:\Downloads\cardiovascular_model; uvicorn app.main:app --host localhost --port 8000"

Write-Host "Starting Streamlit Dashboard on port 8501..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd d:\Downloads\cardiovascular_model; python -m streamlit run dashboard/streamlit_app.py --server.port 8501"

Write-Host "Starting Static Frontend on port 8080..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd d:\Downloads\cardiovascular_model\frontend; python -m http.server 8080"

Write-Host "All servers have been started in separate windows."
