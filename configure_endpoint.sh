#!/bin/bash

echo "============================================"
echo "  Voice2Voice Endpoint Configuration"
echo "============================================"
echo ""

# Show current configuration if exists
if [ -f .env ]; then
    if grep -q "VOICE2VOICE_ENDPOINT" .env; then
        current_endpoint=$(grep "VOICE2VOICE_ENDPOINT" .env | cut -d'=' -f2-)
        echo "Current endpoint: $current_endpoint"
        echo ""
    fi
fi

# Prompt for new endpoint
echo "Enter your RVC/Voice2Voice server endpoint:"
echo "(Example: http://localhost:8001/voice2voice)"
echo "(Or http://192.168.1.100:8001/voice2voice for remote server)"
read -p "Endpoint: " user_endpoint

# Validate input
if [ -z "$user_endpoint" ]; then
    echo "❌ Error: Endpoint cannot be empty!"
    exit 1
fi

# Update or create .env file
if [ -f .env ]; then
    # Update existing file
    if grep -q "VOICE2VOICE_ENDPOINT" .env; then
        # Replace existing line
        sed -i "s|VOICE2VOICE_ENDPOINT=.*|VOICE2VOICE_ENDPOINT=$user_endpoint|" .env
    else
        # Add new line
        echo "VOICE2VOICE_ENDPOINT=$user_endpoint" >> .env
    fi
else
    # Create new file
    echo "VOICE2VOICE_ENDPOINT=$user_endpoint" > .env
fi

echo ""
echo "✓ Configuration updated successfully!"
echo "  New endpoint: $user_endpoint"
echo ""
echo "Please restart your Audio Separator API server for changes to take effect."
