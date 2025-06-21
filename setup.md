## Setup

### Sample nginx site config /etc/nginx/sites-available (as reverse proxy for python app)
```
server {
	listen <VMPort>;
	server_name <serverIP>;

	location / {
		proxy_pass http://127.0.0.1:<pythonAppPort>;
		proxy_set_header Host $host;
		proxy_set_header X-Real-IP $remote_addr;
		proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
		proxy_set_header X-Forwarded-Proto $scheme;
	}
}
```

### Sample daemon file /etc/systemd/system (to keep the python app running, and restart if it crashes) [Requires gunicorn, venv is advisable]
```
[Unit]
Description=<serviceDescriptionOrname>
After=network.target

[Service]
User=azureuser
WorkingDirectory=/home/azureuser/<appDir>
ExecStart=/home/azureuser/<appDir>/venv/bin/gunicorn -w 4 -b 0.0.0.0:<appPort> --access-logfile /home/azureuser/<appDir>/logs/access.log --error-logfile /home/azureuser/<appDir>/logs/error.log <entryScriptName>:<appVariableName>
Restart=always
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```
