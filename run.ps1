# Suppress transformers warnings and run Streamlit
$env:TRANSFORMERS_VERBOSITY = 'error'
$env:TF_CPP_MIN_LOG_LEVEL = '3'
$env:PYTHONWARNINGS = 'ignore'

# Run Streamlit with stderr redirected
streamlit run app.py 2>$null
