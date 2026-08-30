"""Network policy enforcement for JARVIS.

Provides centralized network access control including:
- Private/internal network blocking (RFC1918, loopback)
- URL validation and normalization
- DNS resolution validation
- Redirect destination validation
- Domain allowlist/blocklist

All network access from tools should route through this module.
"""
from __future__ import annotations

import ipaddress
import logging
import re
import socket
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Private/internal network ranges (RFC1918 + loopback + link-local)
PRIVATE_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),       # RFC1918
    ipaddress.ip_network("172.16.0.0/12"),    # RFC1918
    ipaddress.ip_network("192.168.0.0/16"),   # RFC1918
    ipaddress.ip_network("127.0.0.0/8"),      # Loopback
    ipaddress.ip_network("169.254.0.0/16"),   # Link-local
    ipaddress.ip_network("::1/128"),          # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),         # IPv6 unique local
    ipaddress.ip_network("fe80::/10"),        # IPv6 link-local
]

# Allowed protocols
ALLOWED_PROTOCOLS = frozenset(["http", "https"])

# Blocked protocols
BLOCKED_PROTOCOLS = frozenset(["file", "ftp", "gopher", "telnet", "ssh", "ldap", "dict"])


def get_network_policy_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get the network policy configuration."""
    if config is None:
        try:
            from hermes_cli.config import load_config_readonly
            config = load_config_readonly()
        except Exception:
            return _default_network_policy()
    
    policy = config.get("security", {}).get("network_policy", {})
    return {
        "enabled": policy.get("enabled", False),
        "block_private_networks": policy.get("block_private_networks", True),
        "allowed_domains": policy.get("allowed_domains", []),
        "blocked_domains": policy.get("blocked_domains", []),
        "allowed_ips": policy.get("allowed_ips", []),
        "blocked_ips": policy.get("blocked_ips", []),
    }


def _default_network_policy() -> Dict[str, Any]:
    """Return default network policy configuration."""
    return {
        "enabled": False,
        "block_private_networks": True,
        "allowed_domains": [],
        "blocked_domains": [],
        "allowed_ips": [],
        "blocked_ips": [],
    }


def is_ip_private(ip_str: str) -> bool:
    """Check if an IP address is in a private/internal range."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    
    if ip.is_loopback:
        return True
    
    for network in PRIVATE_NETWORKS:
        if ip in network:
            return True
    
    return False


def validate_url_format(url: str) -> Tuple[bool, str]:
    """Validate URL format and protocol."""
    if not url or not isinstance(url, str):
        return False, "URL is empty or invalid"
    
    url = url.strip()
    
    try:
        parsed = urlparse(url)
    except Exception as e:
        return False, f"Failed to parse URL: {e}"
    
    if not parsed.scheme:
        return False, "URL missing protocol (http/https)"
    
    if parsed.scheme.lower() in BLOCKED_PROTOCOLS:
        return False, f"Protocol '{parsed.scheme}' is not allowed"
    
    if parsed.scheme.lower() not in ALLOWED_PROTOCOLS:
        return False, f"Protocol '{parsed.scheme}' is not allowed"
    
    if not parsed.hostname:
        return False, "URL missing hostname"
    
    if parsed.username or parsed.password:
        return False, "URL must not contain credentials"
    
    return True, ""


def is_domain_allowed(domain: str, policy: Dict[str, Any]) -> Tuple[bool, str]:
    """Check if a domain is allowed by policy."""
    if not domain:
        return False, "Empty domain"
    
    domain = domain.lower().strip()
    
    for blocked in policy.get("blocked_domains", []):
        if domain == blocked or domain.endswith("." + blocked):
            return False, f"Domain '{domain}' is blocked"
    
    allowed = policy.get("allowed_domains", [])
    if allowed:
        for allow in allowed:
            if domain == allow or domain.endswith("." + allow):
                return True, ""
        return False, f"Domain '{domain}' is not in allowed list"
    
    return True, ""


def is_ip_allowed(ip_str: str, policy: Dict[str, Any]) -> Tuple[bool, str]:
    """Check if an IP address is allowed by policy."""
    if not ip_str:
        return False, "Empty IP"
    
    for blocked in policy.get("blocked_ips", []):
        if ip_str == blocked:
            return False, f"IP '{ip_str}' is blocked"
    
    if policy.get("block_private_networks", True):
        if is_ip_private(ip_str):
            return False, f"IP '{ip_str}' is in a private/internal network range"
    
    allowed = policy.get("allowed_ips", [])
    if allowed:
        for allow in allowed:
            if ip_str == allow:
                return True, ""
        return False, f"IP '{ip_str}' is not in allowed list"
    
    return True, ""


def resolve_and_check(hostname: str, policy: Dict[str, Any]) -> Tuple[bool, str]:
    """Resolve hostname and check all resolved IPs."""
    try:
        addrinfos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as e:
        return False, f"DNS resolution failed for '{hostname}': {e}"
    except Exception as e:
        return False, f"Error resolving '{hostname}': {e}"
    
    if not addrinfos:
        return False, f"No addresses found for '{hostname}'"
    
    for addrinfo in addrinfos:
        ip = addrinfo[4][0]
        allowed, reason = is_ip_allowed(ip, policy)
        if not allowed:
            return False, f"Resolved IP {ip} for '{hostname}': {reason}"
    
    return True, ""


def is_url_allowed(url: str, config: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
    """Check if a URL is allowed by network policy."""
    policy = get_network_policy_config(config)
    
    if not policy.get("enabled", False):
        return True, ""
    
    is_valid, error = validate_url_format(url)
    if not is_valid:
        return False, error
    
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        return False, "URL missing hostname"
    
    domain_allowed, domain_reason = is_domain_allowed(hostname, policy)
    if not domain_allowed:
        return False, domain_reason
    
    ip_allowed, ip_reason = resolve_and_check(hostname, policy)
    if not ip_allowed:
        return False, ip_reason
    
    return True, ""


def validate_redirect_url(original_url: str, redirect_url: str, config: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
    """Validate a redirect destination."""
    return is_url_allowed(redirect_url, config)


def sanitize_url(url: str) -> str:
    """Sanitize a URL for safe use."""
    if not url:
        return ""
    
    url = url.strip()
    url = re.sub(r"[\x00-\x1f\x7f]", "", url)
    return url
