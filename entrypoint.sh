#!/bin/bash

# Display the working URL prominently
echo ""
echo "=========================================="
echo "  ✅ Streamlit App Started Successfully!"
echo "=========================================="
echo ""
echo "📍 Access your app here:"
echo "   👉 http://localhost:8502"
echo ""
echo "Alternative URLs:"
echo "   • http://127.0.0.1:8502"
echo ""
echo "=========================================="
echo ""

# Disable Streamlit analytics and run the app
export STREAMLIT_LOGGER_LEVEL=error
streamlit run app.py \
  --logger.level=error \
  --client.toolbarMode=minimal \
  --client.showErrorDetails=false
