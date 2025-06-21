#!/bin/bash

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
