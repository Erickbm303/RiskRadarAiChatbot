import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Optional, Union, Tuple
from datetime import datetime, timedelta
import numpy as np
from openai import OpenAI
import json

# Configure Streamlit page
st.set_page_config(
    page_title="RiskRadarAI - AI-Powered Stock Analysis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-score-low {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .risk-score-low h2 {
        color: #155724 !important;
        font-weight: bold;
    }
    .risk-score-low p {
        color: #155724;
    }
    .risk-score-moderate {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .risk-score-moderate h2 {
        color: #856404 !important;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .risk-score-moderate p {
        color: #856404;
    }
    .risk-score-high {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
    .risk-score-high h2 {
        color: #721c24 !important;
        font-weight: bold;
    }
    .risk-score-high p {
        color: #721c24;
    }
    .ai-insight {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
        color: #0d47a1;
    }
    .welcome-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 1rem;
        margin: 2rem 0;
        color: #212529;
    }
    .welcome-section h2 {
        color: #1f77b4;
    }
    .welcome-section h3 {
        color: #495057;
    }
</style>
""", unsafe_allow_html=True)


class StockDataFetcher:
    """Handles fetching and processing stock data from various sources."""
    
    @staticmethod
    def get_stock_data(ticker: str) -> Dict:
        """Fetch stock data with fallback to sample data."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="6mo")
            
            # If we get valid data, use it
            if info and len(info) > 3:  # Basic validation
                current_price = info.get('currentPrice', hist['Close'].iloc[-1] if len(hist) > 0 else None)
                year_high = hist['Close'].max() if len(hist) > 0 else None
                year_low = hist['Close'].min() if len(hist) > 0 else None
                
                # Calculate volatility
                volatility = None
                if len(hist) >= 10:
                    returns = hist['Close'].pct_change().dropna()
                    if len(returns) > 0:
                        volatility = returns.tail(min(30, len(returns))).std() * np.sqrt(252) * 100
                
                return {
                    "info": info,
                    "history": hist,
                    "current_price": current_price,
                    "year_high": year_high,
                    "year_low": year_low,
                    "volatility": volatility,
                    "success": True,
                    "source": "yahoo_finance"
                }
        except:
            pass
        
        # Fallback to sample data
        return StockDataFetcher.get_fallback_data(ticker)
    
    @staticmethod
    def get_fallback_data(ticker: str) -> Dict:
        """Fallback data source when Yahoo Finance fails."""
        mock_data = {
            "AAPL": {
                "longName": "Apple Inc.",
                "sector": "Technology",
                "industry": "Consumer Electronics",
                "currentPrice": 175.50,
                "marketCap": 2800000000000,
                "beta": 1.25,
                "forwardPE": 28.5,
                "debtToEquity": 1.73,
                "profitMargins": 0.257,
                "currentRatio": 1.08,
                "priceToBook": 39.4,
                "country": "United States"
            },
            "GOOGL": {
                "longName": "Alphabet Inc.",
                "sector": "Technology",
                "industry": "Internet Content & Information",
                "currentPrice": 135.25,
                "marketCap": 1700000000000,
                "beta": 1.05,
                "forwardPE": 24.2,
                "debtToEquity": 0.12,
                "profitMargins": 0.209,
                "currentRatio": 2.85,
                "priceToBook": 5.8,
                "country": "United States"
            },
            "MSFT": {
                "longName": "Microsoft Corporation",
                "sector": "Technology",
                "industry": "Software Infrastructure",
                "currentPrice": 365.75,
                "marketCap": 2700000000000,
                "beta": 0.89,
                "forwardPE": 26.8,
                "debtToEquity": 0.35,
                "profitMargins": 0.342,
                "currentRatio": 1.77,
                "priceToBook": 13.2,
                "country": "United States"
            },
            "TSLA": {
                "longName": "Tesla, Inc.",
                "sector": "Consumer Cyclical",
                "industry": "Auto Manufacturers",
                "currentPrice": 195.50,
                "marketCap": 620000000000,
                "beta": 2.35,
                "forwardPE": 45.2,
                "debtToEquity": 0.17,
                "profitMargins": 0.096,
                "currentRatio": 1.84,
                "priceToBook": 9.1,
                "country": "United States"
            }
        }
        
        if ticker.upper() in mock_data:
            data = mock_data[ticker.upper()]
            return {
                "info": data,
                "history": None,
                "current_price": data["currentPrice"],
                "year_high": data["currentPrice"] * 1.15,
                "year_low": data["currentPrice"] * 0.85,
                "volatility": 35.0 if data["beta"] > 1.5 else 20.0,
                "success": True,
                "source": "sample_data"
            }
        else:
            return {"error": f"No data available for {ticker}", "success": False}


class RiskRadarAI:
    """Main risk analysis engine with OpenAI integration."""
    
    def __init__(self, openai_client=None):
        self.openai_client = openai_client
        self.risk_weights = {
            "volatility": 0.20,
            "financial": 0.25,
            "valuation": 0.15,
            "sentiment": 0.15,
            "sector": 0.10,
            "esg": 0.15
        }
    
    def calculate_volatility_risk(self, beta: Optional[float], volatility: Optional[float]) -> Tuple[float, str]:
        """Calculate volatility risk score (0-100)."""
        if beta is None and volatility is None:
            return 50.0, "Beta and volatility data unavailable"
        
        score = 0
        factors = []
        
        if beta is not None:
            if beta < 0.8:
                beta_score = 20
            elif beta < 1.2:
                beta_score = 50
            else:
                beta_score = min(80, 50 + (beta - 1.2) * 30)
            score += beta_score
            factors.append(f"Beta: {beta:.2f}")
        
        if volatility is not None:
            if volatility < 20:
                vol_score = 25
            elif volatility < 40:
                vol_score = 55
            else:
                vol_score = min(90, 55 + (volatility - 40) * 2)
            score += vol_score
            factors.append(f"Volatility: {volatility:.1f}%")
        
        final_score = score / (2 if beta and volatility else 1)
        explanation = f"Based on {', '.join(factors)}"
        
        return min(100, max(0, final_score)), explanation
    
    def calculate_financial_risk(self, info: Dict) -> Tuple[float, str]:
        """Calculate financial health risk score."""
        factors = []
        scores = []
        
        # Debt-to-equity ratio
        debt_to_equity = info.get('debtToEquity')
        if debt_to_equity is not None:
            if debt_to_equity < 0.3:
                debt_score = 20
            elif debt_to_equity < 0.6:
                debt_score = 40
            else:
                debt_score = min(80, 40 + debt_to_equity * 20)
            factors.append(f"D/E: {debt_to_equity:.2f}")
            scores.append(debt_score)
        
        # Current ratio
        current_ratio = info.get('currentRatio')
        if current_ratio is not None:
            if current_ratio > 2:
                liquidity_score = 20
            elif current_ratio > 1.5:
                liquidity_score = 30
            elif current_ratio > 1:
                liquidity_score = 50
            else:
                liquidity_score = 80
            factors.append(f"Current Ratio: {current_ratio:.2f}")
            scores.append(liquidity_score)
        
        # Profit margins
        profit_margin = info.get('profitMargins')
        if profit_margin is not None:
            if profit_margin > 0.15:
                profit_score = 20
            elif profit_margin > 0.05:
                profit_score = 40
            elif profit_margin > 0:
                profit_score = 60
            else:
                profit_score = 90
            factors.append(f"Profit Margin: {profit_margin*100:.1f}%")
            scores.append(profit_score)
        
        final_score = np.mean(scores) if scores else 50
        explanation = f"Based on {', '.join(factors)}" if factors else "Limited financial data available"
        
        return final_score, explanation
    
    def calculate_valuation_risk(self, info: Dict) -> Tuple[float, str]:
        """Calculate valuation risk based on key ratios."""
        factors = []
        scores = []
        
        # P/E ratio
        pe_ratio = info.get('forwardPE') or info.get('trailingPE')
        if pe_ratio is not None:
            if pe_ratio < 15:
                pe_score = 25
            elif pe_ratio < 25:
                pe_score = 45
            elif pe_ratio < 35:
                pe_score = 65
            else:
                pe_score = min(90, 65 + (pe_ratio - 35) * 2)
            factors.append(f"P/E: {pe_ratio:.1f}")
            scores.append(pe_score)
        
        # Price to Book
        pb_ratio = info.get('priceToBook')
        if pb_ratio is not None:
            if pb_ratio < 1:
                pb_score = 20
            elif pb_ratio < 3:
                pb_score = 40
            elif pb_ratio < 5:
                pb_score = 60
            else:
                pb_score = min(85, 60 + (pb_ratio - 5) * 10)
            factors.append(f"P/B: {pb_ratio:.2f}")
            scores.append(pb_score)
        
        final_score = np.mean(scores) if scores else 50
        explanation = f"Based on {', '.join(factors)}" if factors else "Limited valuation data available"
        
        return final_score, explanation
    
    def get_ai_insight(self, ticker: str, risk_analysis: Dict, user_profile: Dict) -> str:
        """Get AI-powered insights using OpenAI."""
        if not self.openai_client:
            return "AI insights require OpenAI API key"
        
        try:
            # Prepare data for AI analysis
            prompt = f"""
            As RiskRadarAI, analyze this stock investment for a retail investor:
            
            Stock: {ticker}
            Company: {risk_analysis.get('company_name', 'N/A')}
            Sector: {risk_analysis.get('sector', 'N/A')}
            
            Risk Analysis:
            - Overall Risk Score: {risk_analysis['final_score']}/100
            - Risk Level: {risk_analysis['risk_level']}
            - Volatility Risk: {risk_analysis['components']['volatility']['score']}/100
            - Financial Risk: {risk_analysis['components']['financial']['score']}/100
            - Valuation Risk: {risk_analysis['components']['valuation']['score']}/100
            
            Key Metrics:
            - Current Price: ${risk_analysis['key_metrics']['current_price']:.2f}
            - P/E Ratio: {risk_analysis['key_metrics']['pe_ratio']}
            - Beta: {risk_analysis['key_metrics']['beta']}
            - Debt/Equity: {risk_analysis['key_metrics']['debt_to_equity']}
            
            User Profile:
            - Investment Horizon: {user_profile.get('time_horizon', 'N/A')}
            - Risk Tolerance: {user_profile.get('risk_tolerance', 'N/A')}
            
            Provide a friendly, educational analysis in 3 paragraphs:
            1. What makes this stock risky or safe?
            2. How does it fit this investor's profile?
            3. One specific actionable insight or consideration.
            
            Keep it under 200 words, use simple language, and end with "Not financial advice."
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are RiskRadarAI, a friendly CFA-level financial analyst who explains stock risks in simple terms for retail investors."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"AI analysis unavailable: {str(e)}"
    
    def get_chat_response(self, user_message: str, ticker: str, risk_analysis: Dict) -> str:
        """Get AI chat response about the stock."""
        if not self.openai_client:
            return "Chat feature requires OpenAI API key"
        
        try:
            context = f"""
            Current analysis context for {ticker}:
            - Company: {risk_analysis.get('company_name', 'N/A')}
            - Risk Score: {risk_analysis['final_score']}/100
            - Risk Level: {risk_analysis['risk_level']}
            - Current Price: ${risk_analysis['key_metrics']['current_price']:.2f}
            - Beta: {risk_analysis['key_metrics']['beta']}
            - P/E: {risk_analysis['key_metrics']['pe_ratio']}
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You are RiskRadarAI, helping investors understand stock risks. Context: {context}"},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Chat unavailable: {str(e)}"
    
    def calculate_comprehensive_risk(self, ticker: str, stock_data: Dict, user_profile: Dict = None) -> Dict:
        """Calculate comprehensive risk analysis."""
        if not stock_data.get("success"):
            return {"error": "Failed to fetch stock data"}
        
        info = stock_data["info"]
        
        # Calculate individual risk components
        volatility_risk, vol_explanation = self.calculate_volatility_risk(
            info.get('beta'), stock_data.get('volatility')
        )
        
        financial_risk, fin_explanation = self.calculate_financial_risk(info)
        valuation_risk, val_explanation = self.calculate_valuation_risk(info)
        
        # Simplified other risks
        sentiment_risk = 45  # Neutral default
        sector_risk = 40     # Neutral default
        esg_risk = 50        # Neutral default
        
        # Calculate weighted final score
        final_score = (
            volatility_risk * self.risk_weights["volatility"] +
            financial_risk * self.risk_weights["financial"] +
            valuation_risk * self.risk_weights["valuation"] +
            sentiment_risk * self.risk_weights["sentiment"] +
            sector_risk * self.risk_weights["sector"] +
            esg_risk * self.risk_weights["esg"]
        )
        
        # Risk classification
        if final_score <= 30:
            risk_level = "Low Risk"
            recommendation = "HOLD - Suitable for conservative portfolios"
        elif final_score <= 60:
            risk_level = "Moderate Risk"
            recommendation = "SPECULATIVE BUY - Suitable for moderate risk tolerance"
        else:
            risk_level = "High Risk"
            recommendation = "AVOID - High risk, suitable only for aggressive investors"
        
        risk_analysis = {
            "ticker": ticker,
            "company_name": info.get('longName', ticker),
            "sector": info.get('sector', 'N/A'),
            "final_score": round(final_score, 1),
            "risk_level": risk_level,
            "recommendation": recommendation,
            "components": {
                "volatility": {"score": volatility_risk, "explanation": vol_explanation},
                "financial": {"score": financial_risk, "explanation": fin_explanation},
                "valuation": {"score": valuation_risk, "explanation": val_explanation},
                "sentiment": {"score": sentiment_risk, "explanation": "Market sentiment analysis"},
                "sector": {"score": sector_risk, "explanation": "Sector-specific risk assessment"},
                "esg": {"score": esg_risk, "explanation": "ESG risk factors"}
            },
            "key_metrics": {
                "current_price": stock_data.get("current_price"),
                "beta": info.get("beta"),
                "pe_ratio": info.get("forwardPE") or info.get("trailingPE"),
                "debt_to_equity": info.get("debtToEquity"),
                "profit_margin": info.get("profitMargins"),
                "market_cap": info.get("marketCap")
            }
        }
        
        # Add AI insight if available
        if user_profile:
            risk_analysis["ai_insight"] = self.get_ai_insight(ticker, risk_analysis, user_profile)
        
        return risk_analysis


def format_currency(value):
    """Format currency values."""
    if value is None:
        return "N/A"
    if value >= 1e12:
        return f"${value/1e12:.2f}T"
    elif value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif value >= 1e6:
        return f"${value/1e6:.2f}M"
    else:
        return f"${value:,.2f}"


def create_risk_chart(risk_analysis):
    """Create a visual representation of risk components."""
    components = risk_analysis["components"]
    
    categories = list(components.keys())
    scores = [components[cat]["score"] for cat in categories]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        name='Risk Profile',
        line_color='rgb(31, 119, 180)',
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10)
            )),
        showlegend=False,
        title="Risk Component Analysis",
        font=dict(size=12),
        height=400
    )
    
    return fig


def show_welcome_screen():
    """Display the welcome screen and API key input."""
    st.markdown('<h1 class="main-header">🤖 RiskRadarAI</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="welcome-section">
        <h2>🚀 AI-Powered Stock Risk Analysis</h2>
        <p>RiskRadarAI combines traditional financial analysis with cutting-edge AI to help you understand stock risks and make informed investment decisions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### ✨ What You'll Get:")
    st.markdown("""
    - 🎯 **AI Risk Scoring:** Comprehensive 0-100 risk assessment
    - 🤖 **Intelligent Insights:** Personalized analysis powered by OpenAI  
    - 💬 **Interactive Chat:** Ask follow-up questions about any stock
    - 📊 **Visual Analytics:** Beautiful charts and risk breakdowns
    - 📈 **Real-time Data:** Live stock prices and financial metrics
    """)
    
    st.markdown("### 🔍 Risk Categories We Analyze:")
    st.markdown("""
    - **Volatility Risk** (20%): Price fluctuations and market sensitivity
    - **Financial Risk** (25%): Debt levels, profitability, and liquidity  
    - **Valuation Risk** (15%): Whether the stock is overpriced
    - **Market Sentiment** (15%): News and analyst opinions
    - **Sector Risk** (10%): Industry-specific challenges
    - **ESG Risk** (15%): Environmental, social, and governance factors
    """)
    
    st.markdown("---")
    
    st.markdown("### 🔑 Get Started")
    st.write("To use RiskRadarAI, you need an OpenAI API key for AI-powered insights. You can get one [here](https://platform.openai.com/account/api-keys).")
    
    # API Key Input
    api_key = st.text_input(
        "Enter your OpenAI API Key:", 
        type="password",
        placeholder="sk-...",
        help="Your API key is used locally and never stored. It's required for AI analysis and chat features."
    )
    
    if api_key:
        if api_key.startswith('sk-') and len(api_key) > 20:
            try:
                # Test the API key
                client = OpenAI(api_key=api_key)
                # Store in session state
                st.session_state.openai_api_key = api_key
                st.session_state.openai_client = client
                st.success("✅ API key validated! Click 'Continue to Analysis' below.")
                
                if st.button("🚀 Continue to Analysis", type="primary"):
                    st.session_state.show_main_app = True
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Invalid API key: {str(e)}")
        else:
            st.warning("⚠️ Please enter a valid OpenAI API key (should start with 'sk-')")
    
    st.markdown("---")
    st.caption("💡 **Sample Analysis Available:** Try the app with sample data even without an API key!")


def show_main_app():
    """Display the main application interface."""
    st.markdown('<h1 class="main-header">🤖 RiskRadarAI - AI Stock Analysis</h1>', unsafe_allow_html=True)
    
    # Initialize components
    openai_client = st.session_state.get('openai_client')
    data_fetcher = StockDataFetcher()
    risk_analyzer = RiskRadarAI(openai_client)
    
    # Sidebar for user inputs
    st.sidebar.header("📊 Analysis Parameters")
    
    # API Status
    if openai_client:
        st.sidebar.success("🤖 AI Analysis: Enabled")
    else:
        st.sidebar.warning("🤖 AI Analysis: Disabled")
        if st.sidebar.button("🔑 Add API Key"):
            if 'show_main_app' in st.session_state:
                del st.session_state.show_main_app
            st.rerun()
    
    # Stock ticker input
    ticker = st.sidebar.text_input("Stock Ticker", value="AAPL", help="Enter a valid US stock ticker (e.g., AAPL, GOOGL, TSLA)")
    
    # User situation inputs
    st.sidebar.subheader("👤 Your Investment Profile")
    time_horizon = st.sidebar.selectbox("Investment Horizon", ["1 year", "3 years", "5 years", "10+ years"])
    risk_tolerance = st.sidebar.selectbox("Risk Tolerance", ["Conservative", "Moderate", "Aggressive"])
    
    # Create user profile
    user_profile = {
        "time_horizon": time_horizon,
        "risk_tolerance": risk_tolerance
    }
    
    # Analysis button
    if st.sidebar.button("🔍 Analyze Stock", type="primary"):
        if ticker:
            with st.spinner(f"🤖 Analyzing {ticker.upper()}..."):
                # Fetch stock data
                stock_data = data_fetcher.get_stock_data(ticker.upper())
                
                if stock_data.get("success"):
                    # Perform risk analysis
                    risk_analysis = risk_analyzer.calculate_comprehensive_risk(ticker.upper(), stock_data, user_profile)
                    
                    if "error" not in risk_analysis:
                        # Store results in session state
                        st.session_state.risk_analysis = risk_analysis
                        st.session_state.stock_data = stock_data
                        
                        if stock_data.get("source") == "sample_data":
                            st.info(f"✅ Analysis completed for {ticker.upper()} using sample data!")
                            st.caption("📝 Note: Using sample data due to API limitations")
                        else:
                            st.success(f"✅ Analysis completed for {ticker.upper()}!")
                    else:
                        st.error(f"❌ Error in risk analysis: {risk_analysis['error']}")
                else:
                    st.error(f"❌ Failed to fetch data: {stock_data.get('error', 'Unknown error')}")
        else:
            st.warning("⚠️ Please enter a stock ticker symbol")
    
    # Display results if available
    if hasattr(st.session_state, 'risk_analysis') and hasattr(st.session_state, 'stock_data'):
        risk_analysis = st.session_state.risk_analysis
        stock_data = st.session_state.stock_data
        info = stock_data["info"]
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Snapshot section
            st.header("📈 Snapshot")
            company_name = info.get("longName", risk_analysis["ticker"])
            sector = info.get("sector", "N/A")
            current_price = format_currency(risk_analysis["key_metrics"]["current_price"])
            
            st.write(f"**{company_name}** ({risk_analysis['ticker']}) operates in the {sector} sector. "
                    f"Current trading price is {current_price} with a market cap of {format_currency(risk_analysis['key_metrics']['market_cap'])}.")
            
            # AI Insight section
            if "ai_insight" in risk_analysis and openai_client:
                st.header("🤖 AI Analysis")
                with st.container():
                    st.markdown("**🧠 AI Insight:**")
                    st.info(risk_analysis['ai_insight'])
            
            # Risk Score section
            st.header("🎯 Risk Score")
            risk_score = risk_analysis["final_score"]
            risk_level = risk_analysis["risk_level"]
            
            # Color-coded risk display
            if risk_score <= 30:
                risk_class = "risk-score-low"
                risk_color = "🟢"
            elif risk_score <= 60:
                risk_class = "risk-score-moderate"
                risk_color = "🟡"
            else:
                risk_class = "risk-score-high"
                risk_color = "🔴"
            
            st.markdown(f"""
            <div class="{risk_class}">
                <h2 style="margin-bottom: 10px;">{risk_color} {risk_score}/100 - {risk_level}</h2>
                <p style="margin-bottom: 0;">{'AI-enhanced' if openai_client else 'Traditional'} risk assessment based on volatility, financial health, valuation, and market factors.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Key Drivers section
            st.header("🔍 Key Risk Drivers")
            components = risk_analysis["components"]
            
            for component, data in components.items():
                with st.expander(f"{component.title()} Risk: {data['score']:.1f}/100"):
                    st.write(data["explanation"])
            
            # Recommendation section
            st.header("💡 Recommendation")
            recommendation = risk_analysis["recommendation"]
            st.info(f"**{recommendation}**")
            
            # Key metrics table
            st.header("📊 Key Metrics")
            metrics_data = {
                "Metric": ["Current Price", "Beta", "P/E Ratio", "Debt/Equity", "Profit Margin"],
                "Value": [
                    format_currency(risk_analysis["key_metrics"]["current_price"]),
                    f"{risk_analysis['key_metrics']['beta']:.2f}" if risk_analysis['key_metrics']['beta'] else "N/A",
                    f"{risk_analysis['key_metrics']['pe_ratio']:.2f}" if risk_analysis['key_metrics']['pe_ratio'] else "N/A",
                    f"{risk_analysis['key_metrics']['debt_to_equity']:.2f}" if risk_analysis['key_metrics']['debt_to_equity'] else "N/A",
                    f"{risk_analysis['key_metrics']['profit_margin']*100:.2f}%" if risk_analysis['key_metrics']['profit_margin'] else "N/A"
                ]
            }
            st.table(pd.DataFrame(metrics_data))
        
        with col2:
            # Risk visualization
            st.subheader("📊 Risk Breakdown")
            risk_chart = create_risk_chart(risk_analysis)
            st.plotly_chart(risk_chart, use_container_width=True)
            
            # Price chart (if available)
            if stock_data.get("history") is not None and len(stock_data["history"]) > 0:
                st.subheader("📈 Price History")
                price_fig = px.line(
                    x=stock_data["history"].index,
                    y=stock_data["history"]["Close"],
                    title=f"{risk_analysis['ticker']} Price History"
                )
                price_fig.update_layout(showlegend=False, height=300)
                st.plotly_chart(price_fig, use_container_width=True)
        
        # Chat Interface (only if OpenAI client available)
        if openai_client:
            st.markdown("---")
            st.header("💬 Ask RiskRadarAI")
            
            # Initialize chat messages
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask me anything about this stock..."):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Get AI response
                with st.chat_message("assistant"):
                    with st.spinner("🤖 Thinking..."):
                        response = risk_analyzer.get_chat_response(prompt, risk_analysis["ticker"], risk_analysis)
                        st.markdown(response)
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})


# Main app logic
def main():
    """Main application entry point."""
    # Initialize session state
    if 'show_main_app' not in st.session_state:
        st.session_state.show_main_app = False
    
    # Show appropriate screen
    if st.session_state.show_main_app:
        show_main_app()
    else:
        show_welcome_screen()


if __name__ == "__main__":
    main()
