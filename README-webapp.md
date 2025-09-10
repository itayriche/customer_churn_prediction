# Customer Churn Prediction Web Application

A comprehensive, production-ready Streamlit web application for predicting customer churn using machine learning models with advanced analytics, model performance metrics, and business impact analysis.

## 🚀 Features

### 🎯 Interactive Prediction Interface
- **Real-time churn probability calculation** with visual risk assessment
- **Comprehensive input form** with validation for all customer attributes
- **Risk categorization** (Low: 0-30%, Medium: 30-70%, High: 70%+)
- **SHAP explainability** showing feature contributions to predictions
- **Business recommendations** based on risk level

### 📊 Model Performance Dashboard
- **Comprehensive metrics** (accuracy, precision, recall, F1, ROC AUC)
- **Interactive visualizations** including ROC curves and confusion matrices
- **Cross-validation analysis** with confidence intervals
- **Feature importance analysis** for tree-based models
- **Model comparison** with business impact metrics

### 🔍 Data Insights & Analytics
- **Exploratory data analysis** with correlation matrices
- **Customer segmentation** visualization and analysis
- **Business impact calculations** with revenue projections
- **High-risk segment identification** and recommendations
- **Data quality assessment** and completeness checks

### 📁 Batch Prediction Capability
- **CSV file upload** with automatic validation
- **Bulk prediction processing** for thousands of customers
- **Results export** in multiple formats (full results, high-risk only, summary)
- **Business impact analysis** for batch predictions
- **Data cleaning options** for missing values

### ⚖️ Model Comparison Dashboard
- **Side-by-side comparison** of multiple ML models
- **Performance radar charts** and business impact analysis
- **ROC curves comparison** and confusion matrices
- **Feature importance comparison** across models
- **Business recommendations** based on different objectives

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9+
- pip package manager
- Git

### Local Installation

1. **Clone the repository:**
```bash
git clone https://github.com/itayriche/customer_churn_prediction.git
cd customer_churn_prediction
```

2. **Install dependencies:**
```bash
# Install ML dependencies
pip install -r requirements.txt

# Install web dependencies  
pip install -r requirements-web.txt
```

3. **Train models (if not already available):**
```bash
python examples/train_model.py --quick
```

4. **Run the web application:**
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Docker Deployment

1. **Build Docker image:**
```bash
docker build -t churn-prediction-app .
```

2. **Run container:**
```bash
docker run -p 8501:8501 churn-prediction-app
```

3. **Access application:**
Open `http://localhost:8501` in your browser

### Production Deployment

#### Using Docker Compose

Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  churn-app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - STREAMLIT_SERVER_HEADLESS=true
      - STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
    volumes:
      - ./models:/app/models:ro
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

#### Cloud Deployment Options

**Streamlit Cloud:**
1. Push code to GitHub repository
2. Connect to Streamlit Cloud
3. Deploy automatically from repository

**Heroku:**
1. Create `Procfile`:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```
2. Deploy using Heroku CLI or GitHub integration

**AWS/GCP/Azure:**
- Use container services (ECS, Cloud Run, Container Instances)
- Deploy Docker image to cloud container registries
- Configure load balancing and auto-scaling as needed

## 📖 User Guide

### Navigation
The application uses Streamlit's multi-page architecture:
- **Home**: Overview and quick metrics
- **🎯 Prediction**: Individual customer churn prediction
- **📊 Performance**: Model performance analysis  
- **🔍 Insights**: Data exploration and insights
- **📁 Batch**: Bulk prediction from CSV files
- **⚖️ Compare**: Model comparison dashboard

### Making Predictions

1. **Navigate to Prediction page**
2. **Select a model** from the dropdown
3. **Fill customer information** using the interactive form
4. **Click "Predict Churn"** to get results
5. **Review risk assessment** and recommendations
6. **Explore explanation** using SHAP values

### Batch Processing

1. **Navigate to Batch page**
2. **Download sample CSV** to see required format
3. **Upload your CSV file** with customer data
4. **Validate and clean data** if needed
5. **Run batch prediction** 
6. **Download results** in preferred format

### Model Comparison

1. **Navigate to Compare page**
2. **Select models** to compare (2-5 recommended)
3. **Choose comparison focus** (performance, business, technical)
4. **Review visualizations** and metrics
5. **Follow recommendations** for model selection

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set custom paths
export CHURN_DATA_PATH="/path/to/your/data.csv"
export CHURN_MODEL_PATH="/path/to/models/"

# Streamlit configuration
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_HEADLESS=true
```

### Customization

**Adding New Models:**
1. Train model using existing pipeline
2. Save in `models/` directory with `.joblib` extension
3. Model will automatically appear in web interface

**Styling:**
- Edit `static/style.css` for custom CSS
- Modify `utils/web_utils.py` for UI components
- Update page layouts in individual page files

**Data Source:**
- Update `DATA_PATH` in `src/config.py`
- Ensure new data follows same schema
- Retrain models if data format changes

## 🎨 UI Components & Features

### Interactive Elements
- **Risk gauges** with color-coded probability meters
- **Feature input widgets** with validation and help text
- **Download buttons** for exporting results and reports
- **Progress indicators** for long-running operations
- **Error handling** with user-friendly messages

### Visualizations
- **Plotly charts** for interactive exploration
- **Confusion matrices** with heatmap visualization
- **ROC curves** for model comparison
- **Feature importance** bar charts
- **Business impact** charts and metrics

### Responsive Design
- **Mobile-friendly** interface that adapts to screen sizes
- **Professional styling** with gradient themes
- **Consistent branding** across all pages
- **Loading animations** and smooth transitions

## 🚦 Performance & Scalability

### Caching Strategy
- **Model loading** cached using `@st.cache_resource`
- **Data loading** cached using `@st.cache_data`
- **Computation results** cached for repeated operations

### Memory Management
- Models loaded once and reused across sessions
- Large datasets handled efficiently with pandas
- Garbage collection for temporary computations

### Scalability Considerations
- **Horizontal scaling** with container orchestration
- **Load balancing** for multiple instances  
- **Session state management** for user data
- **Database integration** for production data sources

## 🔒 Security & Privacy

### Data Protection
- **Input validation** for all user inputs
- **No data persistence** by default (stateless)
- **Secure file handling** for uploads
- **Error message sanitization** to prevent data leaks

### Production Considerations
- **HTTPS enforcement** in production deployments
- **Authentication** integration (OAuth, LDAP, etc.)
- **Access logging** and monitoring
- **Rate limiting** for API protection

## 🐛 Troubleshooting

### Common Issues

**Models not loading:**
```bash
# Check if models exist
ls -la models/

# Retrain if necessary
python examples/train_model.py --quick
```

**Memory issues:**
```bash
# Increase Docker memory limits
docker run -m 4g -p 8501:8501 churn-prediction-app
```

**Port conflicts:**
```bash
# Use different port
streamlit run app.py --server.port 8502
```

**Dependencies errors:**
```bash
# Reinstall dependencies
pip install -r requirements-web.txt --force-reinstall
```

### Performance Optimization

**For large datasets:**
- Implement data sampling for exploration
- Use database connections instead of CSV loading
- Add pagination for large result sets

**For multiple users:**
- Deploy with container orchestration
- Implement proper session management
- Add caching layer (Redis, Memcached)

## 📊 Monitoring & Analytics

### Application Metrics
- **User interactions** and page views
- **Prediction volumes** and accuracy
- **Model usage** statistics
- **Error rates** and performance metrics

### Business Metrics
- **Revenue saved** through predictions
- **Customer retention** improvements  
- **Campaign effectiveness** tracking
- **ROI calculations** for ML initiatives

## 🤝 Contributing

### Development Setup
```bash
# Clone and setup
git clone https://github.com/itayriche/customer_churn_prediction.git
cd customer_churn_prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-web.txt

# Run in development mode
streamlit run app.py --server.runOnSave=true
```

### Adding Features
1. **Create feature branch**: `git checkout -b feature/new-feature`
2. **Implement changes** following existing patterns
3. **Test thoroughly** with different scenarios
4. **Update documentation** as needed
5. **Submit pull request** with description

### Code Style
- Follow PEP 8 Python style guidelines
- Use type hints where appropriate
- Add docstrings to functions and classes
- Include error handling and validation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

### Getting Help
- **Issues**: Report bugs or request features on GitHub
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check this README and code comments
- **Examples**: See `examples/` directory for usage patterns

### Contact Information
- **Project Repository**: https://github.com/itayriche/customer_churn_prediction
- **Documentation**: Available in repository README files
- **Community**: GitHub Discussions and Issues

---

## 🎉 Acknowledgments

Built with ❤️ using:
- **Streamlit** for the web framework
- **Plotly** for interactive visualizations  
- **Scikit-learn** for machine learning
- **SHAP** for model explainability
- **Pandas** for data processing

**Customer Churn Prediction System** - Empowering businesses with AI-driven customer retention insights.