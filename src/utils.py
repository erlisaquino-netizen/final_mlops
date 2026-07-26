import logging
import os
import json
from datetime import datetime
from src.config import MONITORING_ALERTS_DIR

# Configurar Logs Estructurados
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("MLOps-Pipeline")

def send_alert(message: str, alert_type: str = "DRIFT_ALERT"):
    """
    Simula el envío de una alerta guardando un log estructurado en JSON 
    e imprimiéndolo en consola (fácilmente extensible a Slack o Email).
    """
    os.makedirs(MONITORING_ALERTS_DIR, exist_ok=True)
    alert_payload = {
        "timestamp": datetime.now().isoformat(),
        "type": alert_type,
        "message": message
    }
    logger.warning(f"!!! ALERTA GENERAL DISPARADA: {message} !!!")
    
    alert_file = os.path.join(MONITORING_ALERTS_DIR, f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(alert_file, "w") as f:
        json.dump(alert_payload, f, indent=4) }


---SDWDW