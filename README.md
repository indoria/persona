# persona
Root repository for polyrepo of "Journalist AI Persona"


[Persona Forge](https://github.com/indoria/persona-forge)
```
Contains documentation and study guide for the project
```

[Persona Emulator](https://github.com/indoria/persona-emultor)
```
Playground for understanding persona emulation. Contains one journalist "Barkha Dutt"
```

[Persona Net](https://github.com/indoria/persona-net)
```
Contains POC, NLP using spaCy en_core_web_sm
https://spacy.io/models
```

[Persona RAG](https://github.com/indoria/persona-rag)
```
Contains implementation of RAG. Journalists chosen are "Barkha Dutt" and "Christopher Hitchens"
```

[Persona Press](https://github.com/indoria/persona-press)
```
Production grade collection, containing options for different modules and ideas (chunking methods, embedding methods, NLP methods, LLMs)
```

[Persona Journalist](https://github.com/indoria/persona-journalist)
```
Release candidate
```


## Apps / White papers that help you create persona
- [AI persona org team](https://www.personal.ai/)
- [Synthetic Users](https://www.syntheticusers.com/)
- [Creating synthetic user using LLM](https://medium.com/data-science/creating-synthetic-user-research-using-persona-prompting-and-autonomous-agents-b521e0a80ab6)
- [Creating synthetic user - guide](https://www.weavely.ai/blog/creating-synthetic-users-for-free-with-chatgpt)



```
/etc/systemd/system/persona-net.service
[Unit]
Description=Persona Net
After=network.target

[Service]
User=azureuser
WorkingDirectory=/home/azureuser/persona-net
ExecStart=/home/azureuser/persona-net/venv/bin/gunicorn -w 4 -b 0.0.0.0:8000 --access-logfile /home/azureuser/persona-net/logs/access.log --error-logfile /home/azureuser/persona-net/logs/error.log app:app
Restart=always
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target


/etc/nginx/sites-available/persona-net
server {
	listen 8000;
	server_name 130.131.50.173;

	location / {
		proxy_pass http://127.0.0.1:8000;
		proxy_set_header Host $host;
		proxy_set_header X-Real-IP $remote_addr;
		proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
		proxy_set_header X-Forwarded-Proto $scheme;
	}
}
```

```
/etc/systemd/system/persona_rag.service
[Unit]
Description=Persona Rag 
After=network.target

[Service]
User=azureuser
WorkingDirectory=/home/azureuser/persona_rag
ExecStart=/home/azureuser/persona_rag/venv/bin/gunicorn -w 4 -b 0.0.0.0:8001 --access-logfile /home/azureuser/persona_rag/logs/access.log --error-logfile /home/azureuser/persona_rag/logs/error.log app:app
Restart=always
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target


/etc/nginx/sites-available/persona_rag
server {
	listen 80;
	server_name 130.131.50.173;

	location / {
		proxy_pass http://127.0.0.1:8001;
		proxy_set_header Host $host;
		proxy_set_header X-Real-IP $remote_addr;
		proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
		proxy_set_header X-Forwarded-Proto $scheme;
	}
}
```
