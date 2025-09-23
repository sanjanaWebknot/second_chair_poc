Download Ollama from "https://ollama.com/download
Pull the required model [we are currently using Phi3:mini]-> using "ollama pull <model-name>"
Check using "ollama list" 
Create a Virual environemnt using
  -python/python3 -m venv <name of your env>
  -source <name of your env>/bin/activate for linux and mac users
  -.\<name of your env>\Scripts\activate.bat for windows users
Install requirements using "pip install -r requirements.txt"
Run server using "uvicorn ws_server:app --host 0.0.0.0 --port 8002 --reload --log-level info" {change port name as required}


