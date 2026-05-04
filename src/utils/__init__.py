"""Safety rails and privacy protection for multi-modal health monitoring.

This module provides safety mechanisms, privacy protections, and medical
disclaimers for the health monitoring system.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import numpy as np

logger = logging.getLogger(__name__)

# Medical disclaimer text
MEDICAL_DISCLAIMER = """
IMPORTANT MEDICAL DISCLAIMER:

This system is for research and educational purposes only. It is NOT intended for:
- Clinical diagnosis
- Medical treatment decisions
- Healthcare recommendations
- Professional medical advice

Always consult qualified healthcare professionals for medical concerns.

This system does not replace professional medical evaluation and should not be
used for making healthcare decisions.
"""

# Privacy notice
PRIVACY_NOTICE = """
PRIVACY NOTICE:

- All data processing is done locally
- No data is transmitted to external servers
- Temporary files are automatically deleted
- No personal information is stored permanently
- Users maintain full control over their data
"""


class SafetyGuardrails:
    """Safety guardrails for health monitoring system.
    
    Provides content filtering, privacy protection, and safety mechanisms
    for the health monitoring application.
    """
    
    def __init__(self) -> None:
        """Initialize safety guardrails."""
        self.blocked_patterns = self._load_blocked_patterns()
        self.privacy_filters = self._load_privacy_filters()
        self.medical_warnings = self._load_medical_warnings()
        
    def _load_blocked_patterns(self) -> List[str]:
        """Load patterns that should be blocked.
        
        Returns:
            List of blocked patterns.
        """
        return [
            r"emergency|urgent|911|ambulance",
            r"suicide|self.harm|self.injury",
            r"overdose|poison|toxic",
            r"pregnant|pregnancy|baby",
            r"child|infant|pediatric",
            r"elderly|senior|geriatric",
        ]
    
    def _load_privacy_filters(self) -> List[str]:
        """Load privacy-sensitive patterns.
        
        Returns:
            List of privacy-sensitive patterns.
        """
        return [
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN pattern
            r"\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b",  # Credit card pattern
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email pattern
            r"\b\d{3}-\d{3}-\d{4}\b",  # Phone number pattern
        ]
    
    def _load_medical_warnings(self) -> Dict[str, str]:
        """Load medical warning messages.
        
        Returns:
            Dictionary of medical warnings.
        """
        return {
            "emergency": "⚠️ EMERGENCY WARNING: This system is NOT for emergencies. Call 911 immediately.",
            "serious": "⚠️ SERIOUS CONDITION: Consult a healthcare professional immediately.",
            "medication": "⚠️ MEDICATION WARNING: Do not change medications without medical supervision.",
            "diagnosis": "⚠️ DIAGNOSIS WARNING: This system cannot provide medical diagnoses.",
        }
    
    def check_text_safety(self, text: str) -> Tuple[bool, List[str]]:
        """Check text for safety issues.
        
        Args:
            text: Input text to check.
            
        Returns:
            Tuple of (is_safe, warnings).
        """
        warnings = []
        text_lower = text.lower()
        
        # Check for blocked patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, text_lower):
                warnings.append(f"Blocked pattern detected: {pattern}")
                return False, warnings
        
        # Check for privacy-sensitive information
        for pattern in self.privacy_filters:
            if re.search(pattern, text):
                warnings.append("Privacy-sensitive information detected")
                # Don't block, just warn
        
        # Check for medical urgency keywords
        urgency_keywords = ["emergency", "urgent", "critical", "severe", "acute"]
        if any(keyword in text_lower for keyword in urgency_keywords):
            warnings.append(self.medical_warnings["emergency"])
        
        # Check for serious condition keywords
        serious_keywords = ["cancer", "stroke", "heart attack", "seizure", "coma"]
        if any(keyword in text_lower for keyword in serious_keywords):
            warnings.append(self.medical_warnings["serious"])
        
        # Check for medication keywords
        medication_keywords = ["medication", "drug", "prescription", "dosage"]
        if any(keyword in text_lower for keyword in medication_keywords):
            warnings.append(self.medical_warnings["medication"])
        
        return True, warnings
    
    def sanitize_text(self, text: str) -> str:
        """Sanitize text by removing privacy-sensitive information.
        
        Args:
            text: Input text to sanitize.
            
        Returns:
            Sanitized text.
        """
        sanitized_text = text
        
        # Remove privacy-sensitive patterns
        for pattern in self.privacy_filters:
            sanitized_text = re.sub(pattern, "[REDACTED]", sanitized_text)
        
        return sanitized_text
    
    def validate_input_data(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate input data for safety and privacy.
        
        Args:
            data: Input data dictionary.
            
        Returns:
            Tuple of (is_valid, warnings).
        """
        warnings = []
        
        # Check text input
        if "text" in data:
            is_safe, text_warnings = self.check_text_safety(data["text"])
            warnings.extend(text_warnings)
            if not is_safe:
                return False, warnings
        
        # Check file paths for privacy
        for key in ["audio_path", "image_path"]:
            if key in data and data[key]:
                path = Path(data[key])
                if path.exists():
                    # Check file size (prevent extremely large files)
                    file_size = path.stat().st_size
                    if file_size > 100 * 1024 * 1024:  # 100MB limit
                        warnings.append(f"File {key} is too large ({file_size / 1024 / 1024:.1f}MB)")
                        return False, warnings
        
        return True, warnings


class PrivacyProtection:
    """Privacy protection mechanisms for health data.
    
    Provides data anonymization, secure processing, and privacy controls
    for health monitoring applications.
    """
    
    def __init__(self) -> None:
        """Initialize privacy protection."""
        self.temp_files: Set[Path] = set()
        self.processed_data: Dict[str, Any] = {}
        
    def generate_anonymous_id(self, data: str) -> str:
        """Generate anonymous ID for data.
        
        Args:
            data: Input data to anonymize.
            
        Returns:
            Anonymous ID.
        """
        # Create hash of data for anonymous identification
        hash_object = hashlib.sha256(data.encode())
        return hash_object.hexdigest()[:16]
    
    def anonymize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize metadata by removing identifying information.
        
        Args:
            metadata: Original metadata.
            
        Returns:
            Anonymized metadata.
        """
        anonymized = {}
        
        # Keep only non-identifying information
        safe_fields = ["age_group", "gender", "timestamp"]
        for field in safe_fields:
            if field in metadata:
                if field == "age_group":
                    # Convert age to age group
                    age = metadata.get("age", 0)
                    if age < 18:
                        anonymized[field] = "under_18"
                    elif age < 30:
                        anonymized[field] = "18_29"
                    elif age < 50:
                        anonymized[field] = "30_49"
                    elif age < 70:
                        anonymized[field] = "50_69"
                    else:
                        anonymized[field] = "over_70"
                else:
                    anonymized[field] = metadata[field]
        
        return anonymized
    
    def secure_data_processing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data securely with privacy protection.
        
        Args:
            data: Input data.
            
        Returns:
            Securely processed data.
        """
        # Generate anonymous ID
        anonymous_id = self.generate_anonymous_id(str(data))
        
        # Anonymize metadata
        if "metadata" in data:
            data["metadata"] = self.anonymize_metadata(data["metadata"])
        
        # Add privacy flags
        data["privacy_flags"] = {
            "anonymous_id": anonymous_id,
            "processed_locally": True,
            "no_external_transmission": True,
            "temporary_storage": True,
        }
        
        return data
    
    def cleanup_temp_files(self) -> None:
        """Clean up temporary files."""
        for temp_file in self.temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    logger.info(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Could not clean up {temp_file}: {e}")
        
        self.temp_files.clear()
    
    def add_temp_file(self, file_path: Path) -> None:
        """Add temporary file for cleanup.
        
        Args:
            file_path: Path to temporary file.
        """
        self.temp_files.add(file_path)


class MedicalCompliance:
    """Medical compliance and regulatory considerations.
    
    Provides compliance checks, regulatory warnings, and medical
    safety mechanisms for health monitoring applications.
    """
    
    def __init__(self) -> None:
        """Initialize medical compliance."""
        self.regulatory_warnings = self._load_regulatory_warnings()
        self.compliance_flags = self._load_compliance_flags()
        
    def _load_regulatory_warnings(self) -> Dict[str, str]:
        """Load regulatory warning messages.
        
        Returns:
            Dictionary of regulatory warnings.
        """
        return {
            "fda": "⚠️ FDA WARNING: This device is not FDA approved for medical use.",
            "hipaa": "⚠️ HIPAA NOTICE: Ensure compliance with healthcare privacy regulations.",
            "clinical": "⚠️ CLINICAL WARNING: Not approved for clinical decision-making.",
            "liability": "⚠️ LIABILITY WARNING: Use at your own risk. No medical liability assumed.",
        }
    
    def _load_compliance_flags(self) -> Dict[str, bool]:
        """Load compliance flags.
        
        Returns:
            Dictionary of compliance flags.
        """
        return {
            "research_only": True,
            "educational_purpose": True,
            "not_clinical": True,
            "not_diagnostic": True,
            "not_therapeutic": True,
            "no_medical_liability": True,
        }
    
    def check_regulatory_compliance(self, use_case: str) -> Tuple[bool, List[str]]:
        """Check regulatory compliance for use case.
        
        Args:
            use_case: Intended use case.
            
        Returns:
            Tuple of (is_compliant, warnings).
        """
        warnings = []
        
        # Check for non-compliant use cases
        non_compliant_uses = [
            "clinical_diagnosis",
            "medical_treatment",
            "therapeutic_decision",
            "emergency_response",
            "patient_care",
        ]
        
        if use_case.lower() in non_compliant_uses:
            warnings.append(self.regulatory_warnings["clinical"])
            warnings.append(self.regulatory_warnings["fda"])
            return False, warnings
        
        # Add general compliance warnings
        warnings.append(self.regulatory_warnings["fda"])
        warnings.append(self.regulatory_warnings["liability"])
        
        return True, warnings
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report.
        
        Returns:
            Compliance report dictionary.
        """
        return {
            "compliance_flags": self.compliance_flags,
            "regulatory_warnings": self.regulatory_warnings,
            "medical_disclaimer": MEDICAL_DISCLAIMER,
            "privacy_notice": PRIVACY_NOTICE,
            "recommended_actions": [
                "Use only for research and educational purposes",
                "Do not use for clinical decision-making",
                "Consult healthcare professionals for medical advice",
                "Ensure data privacy and security",
                "Follow applicable regulations and guidelines",
            ],
        }


class SafetyManager:
    """Main safety manager for health monitoring system.
    
    Coordinates all safety mechanisms, privacy protection, and compliance
    checks for the health monitoring application.
    """
    
    def __init__(self) -> None:
        """Initialize safety manager."""
        self.guardrails = SafetyGuardrails()
        self.privacy = PrivacyProtection()
        self.compliance = MedicalCompliance()
        
    def validate_and_process_input(
        self, 
        data: Dict[str, Any], 
        use_case: str = "research"
    ) -> Tuple[bool, Dict[str, Any], List[str]]:
        """Validate and process input with full safety checks.
        
        Args:
            data: Input data.
            use_case: Intended use case.
            
        Returns:
            Tuple of (is_valid, processed_data, warnings).
        """
        warnings = []
        
        # Safety validation
        is_safe, safety_warnings = self.guardrails.validate_input_data(data)
        warnings.extend(safety_warnings)
        
        if not is_safe:
            return False, data, warnings
        
        # Compliance check
        is_compliant, compliance_warnings = self.compliance.check_regulatory_compliance(use_case)
        warnings.extend(compliance_warnings)
        
        if not is_compliant:
            return False, data, warnings
        
        # Privacy protection
        processed_data = self.privacy.secure_data_processing(data)
        
        # Add safety metadata
        processed_data["safety_metadata"] = {
            "validated": True,
            "privacy_protected": True,
            "compliance_checked": True,
            "use_case": use_case,
            "timestamp": torch.cuda.Event().elapsed_time() if torch.cuda.is_available() else 0,
        }
        
        return True, processed_data, warnings
    
    def get_safety_warnings(self) -> List[str]:
        """Get all safety warnings.
        
        Returns:
            List of safety warnings.
        """
        warnings = []
        
        # Add medical disclaimer
        warnings.append(MEDICAL_DISCLAIMER)
        
        # Add privacy notice
        warnings.append(PRIVACY_NOTICE)
        
        # Add regulatory warnings
        warnings.extend(self.compliance.regulatory_warnings.values())
        
        return warnings
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.privacy.cleanup_temp_files()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
