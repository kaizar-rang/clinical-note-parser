"""
Shared noise filters reused across MRI, EEG, and clinical parsers.
"""
import re

MIN_LENGTH = 30

BOILERPLATE_PATTERN = re.compile(
    r'forwarded to an automated communication'
    r'|findings were (discussed|communicated)'
    r'|carotid stenosis reference'
    r'|i,? the (attending|teaching) physician'
    r'|alert notification of critical'
    r'|critical results were communicated'
    r'|this report has been'
    r'|electronically notify'
    r'|study (?:terminated|aborted|discontinued) (?:early|due to|at patient)',
    re.IGNORECASE
)

HEADER_PATTERN = re.compile(
    r'^(?:brain mri|head mra|neck mra|mri brain|intracranial mra'
    r'|anterior circulation|posterior circulation|collateral circulation'
    r'|aortic\s+\[redacted\].*origin of major cervical'
    r'|impression|technique|comparison|indication|methodology'
    r'|detail|findings|neck):?\s*$',
    re.IGNORECASE
)

NEGATIVE_PATTERN = re.compile(
    r'^(?:there is |there are )?(?:no evidence of|no acute|no new|no abnormal'
    r'|no significant|unremarkable|within normal limits|no hemorrhage'
    r'|no infarct|no mass effect|no hydrocephalus|no midline shift'
    r'|patent|intact|clear|stable and unremarkable)'
    r'|^the (?:major |main )?(?:intracranial )?(?:flow voids|arterial flow voids|vessels).{0,60}(?:intact|preserved|patent)'
    r'|^(?:extracranial structures|paranasal sinuses|mastoid|orbits).{0,80}(?:clear|normal|unremarkable)',
    re.IGNORECASE
)

EXTRACRANIAL_PATTERN = re.compile(
    r'^(?:paranasal sinuses|mastoid air cells|orbits|skull base'
    r'|extracranial soft tissues|cervical|vertebral arter'
    r'|aortic|subclavian|carotid)',
    re.IGNORECASE
)


def is_boilerplate(text: str) -> bool:
    return bool(BOILERPLATE_PATTERN.search(text))

def is_header(text: str) -> bool:
    return bool(HEADER_PATTERN.match(text))

def is_too_short(text: str) -> bool:
    return len(text.strip()) < MIN_LENGTH

def is_negative(text: str) -> bool:
    return bool(NEGATIVE_PATTERN.match(text))

def is_extracranial(text: str) -> bool:
    return bool(EXTRACRANIAL_PATTERN.match(text))

def is_noise(text: str) -> bool:
    """Returns True if the paragraph should be dropped entirely."""
    return is_too_short(text) or is_header(text) or is_boilerplate(text)
