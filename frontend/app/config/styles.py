"""
Custom CSS Styles
==================
Professional styling for the Streamlit frontend.
"""

import streamlit as st
from .theme import get_theme_css, get_theme_colors


def get_custom_css() -> str:
    """Generate complete custom CSS"""
    colors = get_theme_colors()
    theme_vars = get_theme_css()

    return f"""
    {theme_vars}

    /* ============================================
       GLOBAL STYLES
       ============================================ */

    /* Main container */
    .main .block-container {{
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }}

    /* Force dark/light background on main app */
    [data-testid="stAppViewContainer"] {{
        background-color: {colors.bg_primary} !important;
    }}

    .main {{
        background-color: {colors.bg_primary} !important;
    }}

    [data-testid="stMain"] {{
        background-color: {colors.bg_primary} !important;
    }}

    /* Ensure proper text colors */
    .main .block-container,
    .main .block-container p,
    .main .block-container span,
    .main .block-container label {{
        color: {colors.text_primary};
    }}

    /* Hide Streamlit branding and auto-navigation */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    /* Hide automatic page navigation tabs */
    [data-testid="stSidebarNav"] {{
        display: none !important;
    }}

    /* Hide header decoration if present */
    header[data-testid="stHeader"] {{
        background: transparent;
    }}

    /* Scrollbar styling */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}

    ::-webkit-scrollbar-track {{
        background: var(--color-bg-secondary);
        border-radius: var(--radius-full);
    }}

    ::-webkit-scrollbar-thumb {{
        background: var(--color-border);
        border-radius: var(--radius-full);
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: var(--color-text-muted);
    }}

    /* ============================================
       SIDEBAR STYLES
       ============================================ */

    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {colors.bg_secondary} 0%, {colors.bg_primary} 100%);
        border-right: 1px solid var(--color-border);
    }}

    [data-testid="stSidebar"] .block-container {{
        padding-top: 1rem;
    }}

    /* Sidebar header */
    .sidebar-header {{
        padding: var(--space-md);
        margin-bottom: var(--space-md);
        border-bottom: 1px solid var(--color-border);
    }}

    .sidebar-title {{
        font-size: var(--font-size-lg);
        font-weight: 600;
        color: var(--color-text-primary);
        margin: 0;
    }}

    .sidebar-subtitle {{
        font-size: var(--font-size-sm);
        color: var(--color-text-muted);
        margin-top: var(--space-xs);
    }}

    /* ============================================
       TAB STYLES
       ============================================ */

    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background-color: var(--color-bg-secondary);
        padding: var(--space-xs);
        border-radius: var(--radius-lg);
        margin-bottom: var(--space-lg);
    }}

    .stTabs [data-baseweb="tab"] {{
        padding: var(--space-sm) var(--space-lg);
        background-color: transparent;
        border-radius: var(--radius-md);
        color: var(--color-text-secondary);
        font-weight: 500;
        border: none;
        transition: all var(--transition-fast);
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        background-color: var(--color-bg-tertiary);
        color: var(--color-text-primary);
    }}

    .stTabs [aria-selected="true"] {{
        background-color: var(--color-primary) !important;
        color: white !important;
    }}

    .stTabs [data-baseweb="tab-highlight"] {{
        display: none;
    }}

    .stTabs [data-baseweb="tab-border"] {{
        display: none;
    }}

    /* ============================================
       BUTTON STYLES
       ============================================ */

    .stButton > button {{
        border-radius: var(--radius-md);
        font-weight: 500;
        padding: var(--space-sm) var(--space-lg);
        transition: all var(--transition-fast);
        border: 1px solid transparent;
    }}

    .stButton > button[kind="primary"] {{
        background-color: var(--color-primary);
        color: white;
        border-color: var(--color-primary);
    }}

    .stButton > button[kind="primary"]:hover {{
        background-color: var(--color-primary-hover);
        border-color: var(--color-primary-hover);
        transform: translateY(-1px);
        box-shadow: var(--shadow-lg);
    }}

    .stButton > button[kind="secondary"] {{
        background-color: var(--color-bg-secondary);
        color: var(--color-text-primary);
        border-color: var(--color-border);
    }}

    .stButton > button[kind="secondary"]:hover {{
        background-color: var(--color-bg-tertiary);
        border-color: var(--color-primary);
    }}

    /* ============================================
       INPUT STYLES
       ============================================ */

    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea {{
        background-color: var(--color-bg-primary);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-md);
        color: var(--color-text-primary);
        padding: var(--space-sm) var(--space-md);
        transition: border-color var(--transition-fast);
    }}

    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {{
        border-color: var(--color-primary);
        box-shadow: 0 0 0 2px var(--color-primary-light);
    }}

    /* Select box */
    .stSelectbox > div > div {{
        background-color: var(--color-bg-primary);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-md);
    }}

    /* Multiselect */
    .stMultiSelect > div > div {{
        background-color: var(--color-bg-primary);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-md);
    }}

    /* ============================================
       CARD STYLES
       ============================================ */

    .metric-card {{
        background: var(--color-bg-card);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-lg);
        padding: var(--space-lg);
        transition: all var(--transition-normal);
    }}

    .metric-card:hover {{
        border-color: var(--color-primary);
        box-shadow: var(--shadow-lg);
    }}

    .metric-card-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: var(--space-sm);
    }}

    .metric-card-title {{
        font-size: var(--font-size-sm);
        color: var(--color-text-muted);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    .metric-card-value {{
        font-size: var(--font-size-2xl);
        font-weight: 700;
        color: var(--color-text-primary);
        line-height: 1.2;
    }}

    .metric-card-delta {{
        font-size: var(--font-size-sm);
        margin-top: var(--space-xs);
    }}

    .metric-card-delta.positive {{
        color: var(--color-success);
    }}

    .metric-card-delta.negative {{
        color: var(--color-error);
    }}

    /* Status cards */
    .status-card {{
        padding: var(--space-md);
        border-radius: var(--radius-md);
        display: flex;
        align-items: center;
        gap: var(--space-sm);
    }}

    .status-card.success {{
        background-color: var(--color-success-bg);
        border-left: 4px solid var(--color-success);
    }}

    .status-card.warning {{
        background-color: var(--color-warning-bg);
        border-left: 4px solid var(--color-warning);
    }}

    .status-card.error {{
        background-color: var(--color-error-bg);
        border-left: 4px solid var(--color-error);
    }}

    .status-card.info {{
        background-color: var(--color-info-bg);
        border-left: 4px solid var(--color-info);
    }}

    /* ============================================
       SERVICE CARD STYLES
       ============================================ */

    .service-card {{
        background: var(--color-bg-card);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-lg);
        padding: var(--space-lg);
        position: relative;
        overflow: hidden;
    }}

    .service-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
    }}

    .service-card.healthy::before {{
        background: linear-gradient(90deg, var(--color-success), var(--color-accent-2));
    }}

    .service-card.degraded::before {{
        background: linear-gradient(90deg, var(--color-warning), var(--color-accent-3));
    }}

    .service-card.unhealthy::before {{
        background: linear-gradient(90deg, var(--color-error), var(--color-accent-1));
    }}

    .service-card-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: var(--space-md);
    }}

    .service-name {{
        font-weight: 600;
        font-size: var(--font-size-lg);
        color: var(--color-text-primary);
    }}

    .service-status {{
        padding: var(--space-xs) var(--space-sm);
        border-radius: var(--radius-full);
        font-size: var(--font-size-xs);
        font-weight: 600;
        text-transform: uppercase;
    }}

    .service-status.healthy {{
        background-color: var(--color-success-bg);
        color: var(--color-success);
    }}

    .service-status.degraded {{
        background-color: var(--color-warning-bg);
        color: var(--color-warning);
    }}

    .service-status.unhealthy {{
        background-color: var(--color-error-bg);
        color: var(--color-error);
    }}

    /* ============================================
       PROGRESS & LOADING
       ============================================ */

    .stProgress > div > div {{
        background-color: var(--color-bg-tertiary);
        border-radius: var(--radius-full);
    }}

    .stProgress > div > div > div {{
        background: linear-gradient(90deg, var(--color-primary), var(--color-accent-2));
        border-radius: var(--radius-full);
    }}

    /* ============================================
       EXPANDER STYLES
       ============================================ */

    .streamlit-expanderHeader {{
        background-color: var(--color-bg-secondary);
        border-radius: var(--radius-md);
        border: 1px solid var(--color-border);
        font-weight: 500;
        color: var(--color-text-primary);
    }}

    .streamlit-expanderHeader:hover {{
        border-color: var(--color-primary);
        color: var(--color-primary);
    }}

    .streamlit-expanderContent {{
        border: 1px solid var(--color-border);
        border-top: none;
        border-radius: 0 0 var(--radius-md) var(--radius-md);
        padding: var(--space-md);
    }}

    /* ============================================
       ALERT STYLES
       ============================================ */

    .stAlert {{
        border-radius: var(--radius-md);
        border: none;
    }}

    [data-baseweb="notification"] {{
        border-radius: var(--radius-md) !important;
    }}

    /* ============================================
       DATAFRAME STYLES
       ============================================ */

    .stDataFrame {{
        border: 1px solid var(--color-border);
        border-radius: var(--radius-md);
        overflow: hidden;
    }}

    /* ============================================
       HEADER STYLES
       ============================================ */

    .page-header {{
        margin-bottom: var(--space-xl);
        padding-bottom: var(--space-lg);
        border-bottom: 1px solid var(--color-border);
    }}

    .page-title {{
        font-size: var(--font-size-3xl);
        font-weight: 700;
        color: var(--color-text-primary);
        margin: 0;
    }}

    .page-subtitle {{
        font-size: var(--font-size-md);
        color: var(--color-text-muted);
        margin-top: var(--space-sm);
    }}

    /* Section headers */
    .section-header {{
        display: flex;
        align-items: center;
        gap: var(--space-sm);
        margin: var(--space-xl) 0 var(--space-lg) 0;
        padding-bottom: var(--space-sm);
        border-bottom: 2px solid var(--color-primary);
    }}

    .section-title {{
        font-size: var(--font-size-xl);
        font-weight: 600;
        color: var(--color-text-primary);
        margin: 0;
    }}

    .section-icon {{
        font-size: var(--font-size-xl);
    }}

    /* ============================================
       BADGE STYLES
       ============================================ */

    .badge {{
        display: inline-flex;
        align-items: center;
        padding: var(--space-xs) var(--space-sm);
        border-radius: var(--radius-full);
        font-size: var(--font-size-xs);
        font-weight: 600;
    }}

    .badge.primary {{
        background-color: var(--color-primary-light);
        color: var(--color-primary);
    }}

    .badge.success {{
        background-color: var(--color-success-bg);
        color: var(--color-success);
    }}

    .badge.warning {{
        background-color: var(--color-warning-bg);
        color: var(--color-warning);
    }}

    .badge.error {{
        background-color: var(--color-error-bg);
        color: var(--color-error);
    }}

    /* ============================================
       TOOLTIP STYLES
       ============================================ */

    [data-baseweb="tooltip"] {{
        background-color: var(--color-bg-tertiary) !important;
        border-radius: var(--radius-md) !important;
        box-shadow: var(--shadow-lg) !important;
    }}

    /* ============================================
       FILE UPLOADER
       ============================================ */

    [data-testid="stFileUploader"] > div {{
        background-color: var(--color-bg-secondary);
        border: 2px dashed var(--color-border);
        border-radius: var(--radius-lg);
        padding: var(--space-xl);
        transition: all var(--transition-normal);
    }}

    [data-testid="stFileUploader"] > div:hover {{
        border-color: var(--color-primary);
        background-color: var(--color-primary-light);
    }}

    /* ============================================
       SLIDER STYLES
       ============================================ */

    /* Slider track (background) */
    .stSlider > div > div {{
        background-color: {colors.bg_tertiary} !important;
    }}

    /* Slider filled part */
    .stSlider > div > div > div {{
        background-color: {colors.primary} !important;
    }}

    /* Target specific slider elements */
    [data-testid="stSlider"] [data-baseweb="slider"] > div {{
        background-color: {colors.bg_tertiary} !important;
    }}

    [data-testid="stSlider"] [data-baseweb="slider"] > div > div {{
        background-color: {colors.primary} !important;
    }}

    /* Slider thumb */
    [data-testid="stSlider"] [role="slider"] {{
        background-color: {colors.primary} !important;
        border-color: {colors.primary} !important;
    }}

    /* Slider track - more specific selectors */
    [data-baseweb="slider"] [data-testid="stTickBar"] {{
        background: {colors.bg_tertiary} !important;
    }}

    div[data-baseweb="slider"] > div:first-child {{
        background-color: {colors.bg_tertiary} !important;
    }}

    div[data-baseweb="slider"] > div:first-child > div {{
        background-color: {colors.primary} !important;
    }}

    /* ============================================
       CHECKBOX & RADIO
       ============================================ */

    .stCheckbox > label {{
        color: var(--color-text-primary);
    }}

    .stRadio > label {{
        color: var(--color-text-primary);
    }}

    /* ============================================
       ANIMATIONS
       ============================================ */

    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
    }}

    @keyframes slideIn {{
        from {{ transform: translateX(-20px); opacity: 0; }}
        to {{ transform: translateX(0); opacity: 1; }}
    }}

    .animate-fade-in {{
        animation: fadeIn var(--transition-normal) ease-out;
    }}

    .animate-pulse {{
        animation: pulse 2s infinite;
    }}

    .animate-slide-in {{
        animation: slideIn var(--transition-normal) ease-out;
    }}

    /* ============================================
       UTILITY CLASSES
       ============================================ */

    .text-center {{ text-align: center; }}
    .text-right {{ text-align: right; }}
    .text-muted {{ color: var(--color-text-muted); }}
    .text-primary {{ color: var(--color-primary); }}
    .text-success {{ color: var(--color-success); }}
    .text-warning {{ color: var(--color-warning); }}
    .text-error {{ color: var(--color-error); }}

    .font-bold {{ font-weight: 700; }}
    .font-semibold {{ font-weight: 600; }}
    .font-medium {{ font-weight: 500; }}

    .mt-0 {{ margin-top: 0; }}
    .mt-1 {{ margin-top: var(--space-sm); }}
    .mt-2 {{ margin-top: var(--space-md); }}
    .mt-3 {{ margin-top: var(--space-lg); }}
    .mt-4 {{ margin-top: var(--space-xl); }}

    .mb-0 {{ margin-bottom: 0; }}
    .mb-1 {{ margin-bottom: var(--space-sm); }}
    .mb-2 {{ margin-bottom: var(--space-md); }}
    .mb-3 {{ margin-bottom: var(--space-lg); }}
    .mb-4 {{ margin-bottom: var(--space-xl); }}

    .p-0 {{ padding: 0; }}
    .p-1 {{ padding: var(--space-sm); }}
    .p-2 {{ padding: var(--space-md); }}
    .p-3 {{ padding: var(--space-lg); }}
    .p-4 {{ padding: var(--space-xl); }}

    .rounded {{ border-radius: var(--radius-md); }}
    .rounded-lg {{ border-radius: var(--radius-lg); }}
    .rounded-full {{ border-radius: var(--radius-full); }}

    .shadow {{ box-shadow: var(--shadow); }}
    .shadow-lg {{ box-shadow: var(--shadow-lg); }}

    .border {{ border: 1px solid var(--color-border); }}
    .border-primary {{ border-color: var(--color-primary); }}

    .bg-primary {{ background-color: var(--color-bg-primary); }}
    .bg-secondary {{ background-color: var(--color-bg-secondary); }}
    .bg-card {{ background-color: var(--color-bg-card); }}

    /* Flexbox utilities */
    .flex {{ display: flex; }}
    .flex-col {{ flex-direction: column; }}
    .items-center {{ align-items: center; }}
    .justify-center {{ justify-content: center; }}
    .justify-between {{ justify-content: space-between; }}
    .gap-1 {{ gap: var(--space-sm); }}
    .gap-2 {{ gap: var(--space-md); }}
    .gap-3 {{ gap: var(--space-lg); }}

    /* Grid utilities */
    .grid {{ display: grid; }}
    .grid-cols-2 {{ grid-template-columns: repeat(2, 1fr); }}
    .grid-cols-3 {{ grid-template-columns: repeat(3, 1fr); }}
    .grid-cols-4 {{ grid-template-columns: repeat(4, 1fr); }}
    """


def inject_styles():
    """Inject custom CSS into Streamlit"""
    css = get_custom_css()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
