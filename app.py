import os
import re
import logging
import requests
from flask import Flask, request, jsonify, render_template
from urllib.parse import urlparse
from datetime import datetime
from flask_cors import CORS

# Initialize Flask App
app = Flask(__name__, static_folder='static', template_folder='templates')

# Security Headers (HTTPS enforcement, etc.) - Comment out for local HTTP testing
# Talisman(app)

# Cross-Origin Resource Sharing (Allow frontend to call API)
CORS(app)

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Keys (Load from Environment Variables for Security)
# In terminal: export VT_API_KEY="your_key"
VT_API_KEY = os.getenv('VT_API_KEY', None)
GSB_API_KEY = os.getenv('GSB_API_KEY', None)

# ==============================================================================
# DATA & HEURISTICS (The "Model")
# ==============================================================================

SUSPICIOUS_KEYWORDS = [
    'login', 'verify', 'account', 'update', 'secure', 'banking', 'password', 
    'credential', 'webscr', 'confirm', 'suspend', 'alert', 'limited', 'access',
    'signin', 'signup', 'authenticate', 'recovery', 'wallet', 'alert'
]

SUSPICIOUS_TLDS = ['.xyz', '.top', '.club', '.online', '.site', '.info', '.tk', '.ml', '.ga', '.cf', '.gq']

SHORTENERS = [
    'bit.ly', 'goo.gl', 'tinyurl', 'ow.ly', 't.co', 'is.gd', 'buff.ly', 
    'dlvr.it', 'cutt.ly', 'rb.gy', 'bl.ink'
]

# ==============================================================================
# FEATURE EXTRACTION ENGINE
# ==============================================================================

class PhishingAnalyzer:
    def __init__(self, url):
        self.original_url = url
        self.parsed = None
        self.domain = ""
        self.score = 0
        self.reasons = []
        self.features = {}
        
        # Normalize
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        try:
            self.parsed = urlparse(url)
            # netloc may include port; keep host portion if present
            self.domain = self.parsed.netloc
            if not self.domain and self.parsed.path:
                # urlparse can end up treating input as path if it didn't contain scheme properly
                self.domain = self.parsed.path

            self.features['has_https'] = url.startswith('https://')

            if not self.domain:
                raise ValueError('Empty domain')
        except Exception as e:
            logger.error(f"URL Parse Error: {e}")
            self.reasons.append({"type": "danger", "msg": "Invalid URL format."})
            self.score = 100


    def check_ip_address(self):
        """Check if domain is an IP address"""
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        if re.match(ip_pattern, self.domain):
            self.score += 30
            self.reasons.append({
                "type": "danger", 
                "msg": "URL uses an IP address instead of a domain name (High Risk)."
            })
            self.features['uses_ip'] = True
        else:
            self.features['uses_ip'] = False

    def check_at_symbol(self):
        """Check for @ symbol which obfuscates true destination"""
        if '@' in self.original_url:
            self.score += 30
            self.reasons.append({
                "type": "danger", 
                "msg": "URL contains '@' symbol, often used to hide malicious destinations."
            })

    def check_url_length(self):
        """Check for unusually long URLs"""
        if len(self.original_url) > 75:
            self.score += 10
            self.reasons.append({
                "type": "warning", 
                "msg": f"URL is very long ({len(self.original_url)} chars), potential obfuscation."
            })

    def check_suspicious_keywords(self):
        """Scan for phishing keywords in path/query"""
        found = [kw for kw in SUSPICIOUS_KEYWORDS if kw in self.original_url.lower()]
        if found:
            # Cap impact
            add_score = min(len(found) * 5, 20)
            self.score += add_score
            self.reasons.append({
                "type": "warning", 
                "msg": f"Suspicious keywords detected: {', '.join(found[:3])}."
            })
            self.features['keywords_count'] = len(found)
        else:
            self.features['keywords_count'] = 0

    def check_ssl(self):
        """Check HTTPS presence"""
        if not self.features.get('has_https'):
            self.score += 20
            self.reasons.append({
                "type": "danger", 
                "msg": "Connection is not secure (No HTTPS)."
            })

    def check_tld(self):
        """Check for suspicious Top Level Domains"""
        for tld in SUSPICIOUS_TLDS:
            if self.domain.endswith(tld):
                self.score += 15
                self.reasons.append({
                    "type": "warning", 
                    "msg": f"Uses a high-risk TLD ({tld})."
                })
                break

    def check_shorteners(self):
        """Check for URL shorteners"""
        # Prefer exact host match (or host subdomain match) to avoid false positives.
        host = self.domain.lower()
        # remove port if present
        if ':' in host:
            host = host.split(':', 1)[0]

        for short in SHORTENERS:
            short_l = short.lower()
            if host == short_l or host.endswith('.' + short_l):
                self.score += 15
                self.reasons.append({
                    "type": "info", 
                    "msg": "Link is shortened, final destination is hidden."
                })
                break

    
    def check_phishing_db(self):
        """
        Placeholder for checking against local database of known phishing domains.
        In production, this would query a SQL/NoSQL DB.
        """
        # Mock: Simulating a match for demonstration
        known_bad = ["evil-site.com", "paypa1-secure.com"]
        if self.domain in known_bad:
            self.score += 50
            self.reasons.append({
                "type": "danger", 
                "msg": "Domain found in local blacklist database."
            })

    def check_virustotal(self):
        """
        Check VirusTotal API if key is available.
        """
        if not VT_API_KEY or not self.domain:
            return

        try:
            url = f"https://www.virustotal.com/api/v3/domains/{self.domain}"
            headers = {"x-apikey": VT_API_KEY}
            response = requests.get(url, headers=headers, timeout=3)
            
            if response.status_code == 200:
                data = response.json()
                stats = data.get('data', {}).get('attributes', {}).get('last_analysis_stats', {})
                malicious = stats.get('malicious', 0)
                
                if malicious > 0:
                    self.score += min(malicious * 5, 40) # Cap at 40 points
                    self.reasons.append({
                        "type": "danger", 
                        "msg": f"VirusTotal: {malicious} engines flagged this domain as malicious."
                    })
        except Exception as e:
            logger.warning(f"VirusTotal check failed: {e}")

    def analyze(self):
        """Run all checks"""
        self.check_ip_address()
        self.check_at_symbol()
        self.check_url_length()
        self.check_suspicious_keywords()
        self.check_ssl()
        self.check_tld()
        self.check_shorteners()
        self.check_phishing_db()
        
        # Run external checks asynchronously in production
        self.check_virustotal()

        # Cap score at 100
        final_score = min(100, max(0, self.score))
        
        verdict = "Safe"
        if final_score > 30:
            verdict = "Suspicious"
        if final_score > 60:
            verdict = "Dangerous"

        return {
            "url": self.original_url,
            "domain": self.domain,
            "score": final_score,
            "verdict": verdict,
            "reasons": self.reasons,
            "features": self.features,
            "timestamp": datetime.now().isoformat()
        }

# ==============================================================================
# ROUTES
# ==============================================================================

@app.route('/')
def index():
    """Serve the frontend HTML page"""
    # If you have the index.html in a 'templates' folder:
    return render_template('index.html')
    # Or if using a single file approach without templates folder:
    # return send_from_directory('.', 'index.html')

@app.route('/api/scan', methods=['POST'])
def scan_url():
    """API Endpoint for scanning"""
    data = request.get_json()
    
    if not data or 'url' not in data:
        return jsonify({"error": "URL parameter missing"}), 400
    
    url_input = data['url'].strip()
    
    # Basic validation
    if len(url_input) < 3:
        return jsonify({"error": "URL too short"}), 400

    logger.info(f"Scanning: {url_input}")
    
    try:
        analyzer = PhishingAnalyzer(url_input)
        result = analyzer.analyze()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Analysis Error: {e}")
        return jsonify({"error": "Internal Server Error during analysis"}), 500

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == '__main__':
    # Run in debug mode for development
    app.run(debug=True, port=5000)