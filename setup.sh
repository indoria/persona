#!/bin/bash


####################################
# 1. Install Prerequisites
####################################
sudo apt update && sudo apt upgrade -y
sudo apt install nginx -y
curl -fsSL https://code-server.dev/install.sh | sh


####################################
# 2. Setup Code Server
####################################
# ExecStart=/usr/bin/code-server --bind-addr 0.0.0.0:8080 --auth password --disable-telemetry --password your_password
cat > "/etc/systemd/system/code-server.service" <<EOF
[Unit]
Description=code-server
After=network.target

[Service]
Type=simple
User=azureuser
ExecStart=/usr/bin/code-server --bind-addr 0.0.0.0:8080
Restart=always
RestartSec=10
Environment=PASSWORD=PersonaNet123

[Install]
WantedBy=multi-user.target
EOF

# sudo systemctl enable --now code-server@$USER
# Creates a symlink => Created symlink /etc/systemd/system/default.target.wants/code-server@azureuser.service → /lib/systemd/system/code-server@.service.
# User is set to User=%i in this file
# sudo systemctl status code-server@$USER
# from "systemctl status" => Using password from /home/azureuser/.config/code-server/config.yaml
# Bind address is also found in "home/azureuser/.config/code-server/config.yaml", change it to 0.0.0.0:8080 to access it from internet



####################################
# 3. Clone Repos
####################################
repos=(
    "https://github.com/indoria/persona.git",
    "https://github.com/indoria/persona-rag.git",
    "https://github.com/indoria/persona-net.git",
    "https://github.com/indoria/persona-forge.git",
    "https://github.com/indoria/persona-emulator.git",
    "https://github.com/indoria/persona-press.git",
    "https://github.com/indoria/persona-journalist.git",
)

CLONE_DIR="/home/azureuser"

cloneRepos() {
  echo "Starting Git repository cloning process..."
  echo "---------------------------------------"

  if [ ${#repos[@]} -eq 0 ]; then
    echo "Error: No repository URLs provided in the 'repos' array."
    echo "Please edit the script and add your repository URLs."
    exit 1
  fi

  if [ -n "${CLONE_DIR}" ]; then
    echo "Cloning repositories into: ${CLONE_DIR}"
    mkdir -p "${CLONE_DIR}"
    cd "${CLONE_DIR}" || { echo "Error: Could not change to directory ${CLONE_DIR}. Exiting."; exit 1; }
  else
    echo "Cloning repositories into the current directory ($(pwd))."
  fi

  for repo_url in "${repos[@]}"; do
    echo ""
    echo "Cloning: ${repo_url}"
    git clone "${repo_url}" || { echo "Warning: Failed to clone ${repo_url}. Continuing with next."; true; }
    echo "---------------------------------------"
  done

  echo ""
  echo "Git repository cloning process completed."
  echo "Check the output above for any warnings or errors."
}

cloneRepos





####################################
# 3. Set up Reverse Proxy and Daemon service
####################################

BASE_DIR="/home/azureuser"
SYSTEMD_DIR="/etc/systemd/system"
NGINX_AVAILABLE="/etc/nginx/sites-available"
NGINX_ENABLED="/etc/nginx/sites-enabled"
DOMAIN="example.com"

PORT=8000

for dir in "$BASE_DIR"/*/; do
    RAW_NAME=$(basename "$dir")
    APP_NAME=$(echo "$RAW_NAME" | tr '_-' ' ' | sed 's/\b\(.\)/\u\1/g')

    APP_PORT=$PORT
    PORT=$((PORT + 1))  # Increment port for next app

    # ----- Create systemd service file -----
    SERVICE_FILE="$SYSTEMD_DIR/${RAW_NAME}.service"
    cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=$APP_NAME
After=network.target

[Service]
User=azureuser
WorkingDirectory=$dir
ExecStart=$dir/venv/bin/gunicorn -w 4 -b 0.0.0.0:$APP_PORT --access-logfile $dir/logs/access.log --error-logfile $dir/logs/error.log run:app 
Restart=always
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

    # ----- Create NGINX config -----
    NGINX_FILE="$NGINX_AVAILABLE/$APP_NAME"
    cat > "$NGINX_FILE" <<EOF
server {
    listen $APP_PORT;
    server_name $DOMAIN;

    location /$APP_NAME/ {
        proxy_pass http://127.0.0.1:$APP_PORT/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

    ln -sf "$NGINX_FILE" "$NGINX_ENABLED/$APP_NAME"
done

# ----- Reload systemd and nginx -----
systemctl daemon-reexec
systemctl daemon-reload

for dir in "$BASE_DIR"/*/; do
    APP_NAME=$(basename "$dir")
    systemctl enable "${APP_NAME}.service"
    systemctl restart "${APP_NAME}.service"
done

nginx -t && systemctl reload nginx

echo "✅ All apps deployed and reverse proxies configured."
