# 🏛️ MonuVision AI - Streamlit Deployment Guide

## Streamlit Cloud Deployment Settings

Use these exact settings when deploying to Streamlit Cloud:

### Basic Settings
- **Repository**: `NikhilMana/Monument-Classification` (or your GitHub username/repo)
- **Branch**: `master` (or `main` if you renamed it)
- **Main file path**: `streamlit_app.py`
- **App URL (optional)**: `monument-classification` (or your preferred subdomain)

### Advanced Settings (Click "Advanced settings" on deploy page)

#### Python Version
- **Python version**: `3.10` (recommended)

#### Secrets (if needed)
If you're using any API keys or secrets, add them in the Secrets section. For this app, you likely don't need any.

## Important Notes

### Model File
⚠️ **CRITICAL**: Make sure your trained model file is committed to GitHub:
- The app expects the model at: `models/best_model.keras`
- If your model has a different name, update line 38 in `streamlit_app.py`
- GitHub has a 100MB file size limit. If your model is larger:
  - Use Git LFS (Large File Storage)
  - Or host the model elsewhere and download it in the app

### Dataset Configuration
The `config.py` file tries to download the Kaggle dataset. For Streamlit Cloud deployment:
- The dataset download is only needed for training, not inference
- The app will use fallback class names if config fails
- Consider removing the kagglehub dependency from requirements.txt if not needed

## Testing Locally

Before deploying, test the app locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

## Deployment Steps

1. **Commit all files to GitHub**:
   ```bash
   git add .
   git commit -m "Add Streamlit deployment"
   git push origin master
   ```

2. **Go to Streamlit Cloud**:
   - Visit https://share.streamlit.io/
   - Sign in with GitHub
   - Click "New app"

3. **Fill in the deployment form** with the settings above

4. **Click "Deploy"**

5. **Wait for deployment** (usually 2-5 minutes)

## Files Created for Deployment

- ✅ `streamlit_app.py` - Main Streamlit application
- ✅ `requirements.txt` - Updated with streamlit dependency
- ✅ `.streamlit/config.toml` - Custom theme configuration
- ✅ `packages.txt` - System-level dependencies

## Troubleshooting

### Model Not Found Error
- Ensure `models/best_model.keras` exists in your repo
- Check the file path in `streamlit_app.py` line 38

### Memory Issues
- Streamlit Cloud free tier has 1GB RAM limit
- Consider using a smaller model or model quantization

### Dependency Errors
- Check that all versions in `requirements.txt` are compatible
- TensorFlow 2.13+ should work on Streamlit Cloud

### kagglehub Issues
- If you get errors about kagglehub, you can remove it from requirements.txt
- The app will use fallback class names

## Post-Deployment

After successful deployment:
- Test with various monument images
- Share the URL with others
- Monitor app performance in Streamlit Cloud dashboard

## Need Help?

- Streamlit Docs: https://docs.streamlit.io/
- Streamlit Community: https://discuss.streamlit.io/
