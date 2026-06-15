import re
import pandas as pd

# ── Filters v8 ────────────────────────────────────────────────────────────────

BOILERPLATE_PATTERN = re.compile(
    r'forwarded to an automated communication'
    r'|findings were (?:discussed|communicated)'
    r'|(?:the )?(?:critical )?findings in this report were reported to'
    r'|who responded indicating that the communication was understood'
    r'|alert notification of critical'
    r'|critical results were communicated'
    r'|this report has been'
    r'|electronically notify'
    r'|per epic note'
    r'|carotid stenosis reference'
    r'|(?:\[redacted\][\w\s,\.]*)?(?:md|do|phd|np|pa)[\s,]+the (?:attending|teaching) physician'
    r'|i,? the (?:attending|teaching) physician'
    r'|have reviewed the images and agree'
    r'|study (?:terminated|aborted|discontinued) (?:early|due to|at patient)'
    r'|post contrast[:\s]+following (?:iv|intravenous) administration'
    r'|following (?:iv|intravenous) administration of (?:gadolinium|contrast)'
    r'|(?:exam|study|images?|sequences?|imaging quality|image quality) (?:is|are) (?:limited|degraded) (?:by|due to|secondary to) (?:motion|artifact|patient)'
    r'|(?:patient was not able|patient could not) (?:to )?tolerate'
    r'|(?:patient was )?removed from the magnet'
    r'|only (?:diffusion|flair|dwi|t1|t2)[\w\s]+ (?:sequence|sequences|images?) (?:were|was) obtained'
    r'|(?:images?|sequences?) (?:are|were|is) degraded by'
    r'|pending postprocessing at the time of this dictation'
    r'|perfusion images? (?:are )?pending'
    r'|study quality is limited'
    r'|there is motion artifact'
    r'|(?:asl|dwi) sequence is nondiagnostic'
    r'|you can find out more about our efforts'
    r'|visiting \[redacted\]'
    r'|suboptimal evaluation (?:for|of|secondary to).{0,60}(?:motion|artifact|technique)'
    r'|follow.?up (?:is )?recommended'
    r'|contact was made at the time of interpr'
    r'|(?:communicated|documented) via(?: a)? closed.?loop communication'
    r'|(?:results? were|findings? were) communicated to the referring (?:provider|physician|clinician)s? via'
    r'|communicated to the referring providers? via \w+'
    r'|(?:swan|t2 star|t2\*).{0,40}(?:motion degraded|markedly.{0,10}degraded|nondiagnostic)',
    re.IGNORECASE
)

HEADER_PATTERN = re.compile(
    r'^(?:brain mri|head mra|neck mra|mri brain|intracranial mra'
    r'|anterior circulation|posterior circulation|collateral circulation'
    r'|aortic\s+\[redacted\].*origin of major cervical'
    r'|impression|technique|comparison|indication|methodology'
    r'|detail|findings|neck'
    r'|ventricular system[\w\s\-]+:'
    r'|extra-axial spaces?:'
    r'|brain parenchyma:'
    r'|vascular system:'
    r'|vascular structures?:'
    r'|extracranial structures?:?'
    r'|flow.related signal'
    r'|synopsis for clinical management'
    r'|miscellaneous:'
    r'|calvarium:'
    r'|anterior circulation:):?\s*$',
    re.IGNORECASE
)

NEGATIVE_PATTERN = re.compile(
    r'^(?:there (?:is|are) )?(?:no evidence of|no acute|no new|no abnormal'
    r'|no significant|unremarkable|within normal limits|no hemorrhage'
    r'|no infarct|no mass effect|no hydrocephalus|no midline shift'
    r'|no space.occupying|no enhancing lesion|no abnormal enhancement'
    r'|no restricted diffusion|no acute intracranial|no brain parenchymal mass'
    r'|no interval change|no intracranial mass|no mass lesion'
    r'|no gross abnormalities)'
    r'|^(?:the )?(?:intracranial )?(?:major )?(?:flow.?voids?|arterial flow voids|vessels).{0,60}(?:intact|preserved|patent|maintained|present)'
    r'|^intracranial (?:major )?(?:dural sinus and )?arterial flow voids are present'
    r'|^flow.related signal is observed.{0,120}without (?:occlusion|stenosis)'
    r'|^the (?:visualized )?(?:paranasal sinuses|mastoid air cells?|orbits|bones and extracranial|extracranial soft tissues).{0,80}(?:clear|normal|unremarkable|unchanged)'
    r'|^the (?:remaining )?ventricles? (?:are|is) (?:stable|normal|unremarkable|unchanged)'
    r'|^(?:the )?ventricles?[,\s]+sulci[,\s]+and (?:cisterns?|basal cisterns?).{0,60}(?:unremarkable|normal|stable|within normal limits)'
    r'|^ventricles?[,\s]+sulci[,\s]+and cisterns? are stable'
    r'|^there has been no (?:significant )?interval change in (?:size|the size)'
    r'|^(?:\[redacted\]\s+)?venous sinuses? (?:enhance|are|appear).{0,40}(?:normal|patent|intact|homogenous)'
    r'|^vascular system:.{0,120}(?:intact|preserved|patent|normal|homogenous)'
    r'|^(?:extracranial structures|paranasal sinuses|mastoid).{0,80}(?:clear|normal|unremarkable)'
    r'|^(?:otherwise,?\s+)?the bones(?: and extracranial soft tissues)?.{0,60}(?:unremarkable|normal|unchanged)'
    r'|^(?:post contrast|following contrast).{0,60}no (?:abnormal|significant)'
    r'|^(?:there are )?no regions of abnormal restricted diffusivity'
    r'|^diffusion.weighted imaging demonstrates no acute'
    r'|^diffusion shows no signal abnormality'
    r'|^asl (?:sequence )?does not show'
    r'|^calvarium:.{0,60}(?:normal|within normal limits|unremarkable)',
    re.IGNORECASE
)

EXTRACRANIAL_PATTERN = re.compile(
    r'^(?:paranasal sinuses|mastoid air cells?|orbits|skull base'
    r'|extracranial soft tissues|cervical|vertebral arter'
    r'|aortic|subclavian|carotid|nasopharynx|external carotid'
    r'|mra\s+\[?redacted\]?)'
    r'|^(?:otherwise,?\s+)?the bones'
    r'|^the (?:visualized )?(?:paranasal sinuses|mastoid|orbits)'
    r'|^the (?:intracranial )?vertebral arteries'
    r'|^(?:partial|bilateral|mild|moderate|severe|minimal|scattered)?\s*(?:opacification|thickening|mucosal).{0,40}(?:sinus|mastoid|orbit)'
    r'|(?:mastoid air cells?|middle ear cav|maxillary sinus|sphenoid sinus|ethmoid sinus|paranasal sinus).{0,50}(?:opacif|thicken|fluid|effusion|clear|normal|air cell)'
    r'|(?:superior sagittal|transverse sinus|sigmoid sinus|straight sinus|cavernous sinus|dural (?:venous )?sinus|venous sinus).{0,80}(?:patent|thrombus|laceration|signal|enhance|filling defect|persistent|resolution|confluence)'
    r'|(?:nasopharynx|pharynx).{0,60}(?:edema|mucosal|thickening|fluid)'
    r'|(?:edema|mucosal thickening).{0,40}(?:nasopharynx|pharynx)'
    r'|(?:spinal canal|neuroforaminal|neural foramen|spinal cord|cauda equina|conus terminates|c\d-\d|l\d[ -]|cervical spine|thoracic spine|cervicomedullary)'
    r'|^(?:the )?(?:intracranial vertebral arteries|basilar artery).{0,40}(?:patent|normal)'
    r'|^(?:left|right)? vertebral artery (?:demonstrates|shows)'
    r'|^(?:the )?intracranial (?:ica|internal carotid).{0,80}(?:normal|patent|caliber)'
    r'|^(?:the )?(?:anterior|middle|posterior) cerebral arter.{0,60}(?:normal|patent|caliber)'
    r'|^(?:in )?the circle of.{0,80}(?:patent|atherosclerotic|anterior|posterior)'
    r'|^anterior circulation:'
    r'|^(?:no )?hemodynamically.significant stenosis'
    r'|(?:aortic arch|three.vessel arch|subclavian arter|vertebral arter(?:ies)? arise)'
    r'|(?:extracranial and intracranial|extracranial).{0,40}(?:internal carotid|carotid artery|vertebral)'
    r'|(?:flow.related enhancement|flow related enhancement).{0,60}(?:internal carotid|carotid|vertebral|middle cerebral)'
    r'|(?:\bright\b|\bleft\b|\bbilateral\b)?\s*(?:parietal bone|frontal bone|occipital bone|temporal bone|sphenoid bone|ethmoid bone)'
    r'|(?:calvarium|calvarial).{0,60}(?:focus|lesion|metastas|hyperintens|signal)'
    r'|(?:maxillary sinus|sphenoid sinus|ethmoid).{0,80}(?:asymmetr|diminutive|bowing|deviat|retention cyst)'
    r'|(?:nasogastric|nasoenteric|feeding).{0,30}(?:tube|ng tube).{0,60}(?:oropharynx|pharynx|stomach|esophagus|coiled|positioned)'
    r'|(?:oropharynx|pharynx).{0,60}(?:tube|coiled)',
    re.IGNORECASE
)

INLINE_SUBSECTION_PATTERN = re.compile(
    r'^(?:brain parenchyma|ventricular system[\w\s\-]*|extra-axial spaces?'
    r'|vascular system|extracranial structures?|calvarium|miscellaneous'
    r'|synopsis for clinical management)\s*:\s*',
    re.IGNORECASE
)

# ── Public API ────────────────────────────────────────────────────────────────

def strip_inline_header(text: str) -> str:
    return INLINE_SUBSECTION_PATTERN.sub('', text).strip()

def is_noise(t: str) -> bool:
    if len(t.strip()) < 30:
        return True
    if HEADER_PATTERN.match(t):
        return True
    if BOILERPLATE_PATTERN.search(t):
        return True
    return False

def is_negative(t: str) -> bool:
    return bool(NEGATIVE_PATTERN.match(t))

def is_extracranial(t: str) -> bool:
    return bool(EXTRACRANIAL_PATTERN.search(t))

# ── Splitter ──────────────────────────────────────────────────────────────────

SECTION_SPLIT_PATTERN = re.compile(
    r'\n(?=(?:BRAIN MRI|HEAD MRA|NECK MRA|MRI BRAIN AND NECK'
    r'|SPINE|CERVICAL|THORACIC|LUMBAR|STRUCTURAL MRI|FUNCTIONAL MRI)\s*[:\n])',
    re.IGNORECASE
)
BRAIN_SECTION_PATTERN = re.compile(r'^(?:BRAIN MRI|MRI BRAIN|STRUCTURAL MRI)', re.IGNORECASE)
INLINE_HEADER_PATTERN = re.compile(r'^(?:BRAIN MRI|HEAD MRA|MRI BRAIN)\s*:?\s*\n', re.IGNORECASE)


def _extract_brain_section(text: str) -> str:
    if not SECTION_SPLIT_PATTERN.search(text):
        return text
    sections = SECTION_SPLIT_PATTERN.split(text)
    brain = [s for s in sections if BRAIN_SECTION_PATTERN.match(s.strip())]
    return brain[0] if brain else sections[0]


def split_report(text: str) -> list[str]:
    """Split a findings string into cleaned paragraphs."""
    if pd.isna(text):
        return []
    text = _extract_brain_section(text)
    result = []
    for p in re.split(r'\n[ \t]*\n', text):
        p = INLINE_HEADER_PATTERN.sub('', p.strip()).strip()
        p = strip_inline_header(p)
        if len(p) >= 30:
            result.append(p)
    return result