import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import io

# Set page config
st.set_page_config(
    page_title="Student Performance Nexus",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #FFD166, #FF6B6B, #4ECDC4, #FFD166);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: rgba(20, 25, 45, 0.6);
        backdrop-filter: blur(12px);
        border-radius: 28px;
        border: 1px solid rgba(255,255,255,0.15);
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFD166, #FF6B6B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
</style>
""", unsafe_allow_html=True)

# Generate synthetic dataset
@st.cache_data
def generate_dataset(n=450):
    np.random.seed(42)  # For reproducibility
    data = []

    for i in range(1, n+1):
        # Generate realistic correlated features
        attendance = np.clip(np.random.normal(75, 15), 20, 100)
        homework = np.clip(attendance * 0.6 + np.random.normal(0, 15), 20, 100)
        midterm = np.clip(homework * 0.5 + np.random.normal(0, 20), 20, 100)
        study_hours = np.clip(attendance * 0.2 + np.random.normal(0, 8), 2, 35)

        # Calculate weighted total score
        total_score = attendance * 0.25 + homework * 0.35 + midterm * 0.4

        # Logistic function for pass probability
        logit = (total_score - 62) / 12
        prob_pass = 1 / (1 + np.exp(-logit))
        pass_fail = 1 if np.random.random() < prob_pass else 0

        data.append({
            'student_id': f'STU{i:03d}',
            'attendance_pct': round(attendance, 1),
            'homework_pct': round(homework, 1),
            'midterm_score': round(midterm, 1),
            'study_hours_per_week': round(study_hours, 1),
            'total_score': round(total_score, 1),
            'pass': pass_fail
        })

    return pd.DataFrame(data)

# Train ML model
@st.cache_data
def train_model(df):
    features = ['attendance_pct', 'homework_pct', 'midterm_score', 'study_hours_per_week']
    X = df[features]
    y = df['pass']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy, X_train, X_test, y_train, y_test

# Load data and train model
df = generate_dataset()
model, accuracy, X_train, X_test, y_train, y_test = train_model(df)

# Main app
def main():
    st.markdown('<h1 class="main-header">⚡ Student Performance Nexus</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">AI-driven predictions · Deep analytics · Real-time insights</p>', unsafe_allow_html=True)

    # Navigation
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📈 Analytics", "🤖 Smart Predictor", "📁 Batch Processor"])

    with tab1:
        show_dashboard()

    with tab2:
        show_analytics()

    with tab3:
        show_predictor()

    with tab4:
        show_batch_processor()

def show_dashboard():
    st.header("📊 Dashboard Overview")

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    total_students = len(df)
    pass_count = df['pass'].sum()
    pass_rate = (pass_count / total_students * 100)
    avg_attendance = df['attendance_pct'].mean()
    avg_study = df['study_hours_per_week'].mean()

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_students}</div>
            <div>Total Students</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{pass_rate:.1f}%</div>
            <div>Pass Rate</div>
            <div style="font-size:0.8rem;">{int(pass_count)} students</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_attendance:.1f}%</div>
            <div>Avg Attendance</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_study:.1f} hrs</div>
            <div>Avg Study Hours</div>
        </div>
        """, unsafe_allow_html=True)

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Score Distribution")
        fig_hist = px.histogram(
            df, x='total_score', color='pass',
            color_discrete_map={1: '#4ECDC4', 0: '#FF6B6B'},
            labels={'pass': 'Result', 'total_score': 'Total Score'},
            title="Pass vs Fail Distribution"
        )
        fig_hist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        st.subheader("Attendance vs Total Score")
        fig_scatter = px.scatter(
            df, x='attendance_pct', y='total_score',
            size='study_hours_per_week', color='pass',
            color_discrete_map={1: '#4ECDC4', 0: '#FF6B6B'},
            labels={'pass': 'Result'},
            title="Attendance vs Total Score (bubble size = study hours)"
        )
        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Top/Bottom performers
    st.subheader("🏆 Top 5 Performers & ⚠️ At-Risk Students")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🏆 Top 5 Performers**")
        top_5 = df.nlargest(5, 'total_score')[['student_id', 'total_score', 'attendance_pct', 'study_hours_per_week']]
        st.dataframe(top_5, use_container_width=True)

    with col2:
        st.markdown("**⚠️ At-Risk (Bottom 5)**")
        bottom_5 = df.nsmallest(5, 'total_score')[['student_id', 'total_score', 'attendance_pct', 'study_hours_per_week']]
        st.dataframe(bottom_5, use_container_width=True)

def show_analytics():
    st.header("📈 Advanced Analytics")

    # Feature distributions
    st.subheader("Feature Distributions (Pass vs Fail)")

    features = ['attendance_pct', 'homework_pct', 'midterm_score', 'study_hours_per_week']
    feature_names = ['Attendance (%)', 'Homework (%)', 'Midterm Score', 'Study Hours/Week']

    fig_box = go.Figure()
    for i, (feature, name) in enumerate(zip(features, feature_names)):
        pass_data = df[df['pass'] == 1][feature]
        fail_data = df[df['pass'] == 0][feature]

        fig_box.add_trace(go.Box(y=pass_data, name=f'{name} (Pass)', marker_color='#4ECDC4'))
        fig_box.add_trace(go.Box(y=fail_data, name=f'{name} (Fail)', marker_color='#FF6B6B'))

    fig_box.update_layout(
        title="Distribution by Feature and Pass/Fail",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        showlegend=True
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")

    corr_cols = ['attendance_pct', 'homework_pct', 'midterm_score', 'study_hours_per_week', 'total_score', 'pass']
    corr_matrix = df[corr_cols].corr()

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='Viridis',
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))

    fig_heatmap.update_layout(
        title="Correlation Matrix",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

def show_predictor():
    st.header("🤖 Smart Predictor")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input Student Data")

        attendance = st.slider("📋 Attendance (%)", 0, 100, 75, 1)
        homework = st.slider("📚 Homework (%)", 0, 100, 70, 1)
        midterm = st.slider("📝 Midterm Score", 0, 100, 65, 1)
        study_hours = st.slider("⏰ Study Hours/Week", 0, 40, 10, 1)

        if st.button("✨ PREDICT NOW", type="primary"):
            # Make prediction
            input_data = pd.DataFrame({
                'attendance_pct': [attendance],
                'homework_pct': [homework],
                'midterm_score': [midterm],
                'study_hours_per_week': [study_hours]
            })

            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]

            st.session_state.prediction = prediction
            st.session_state.confidence = probability[1] if prediction == 1 else probability[0]

    with col2:
        st.subheader("Prediction Result")

        if 'prediction' in st.session_state:
            prediction = st.session_state.prediction
            confidence = st.session_state.confidence

            if prediction == 1:
                st.success("✅ PASS")
                st.metric("Confidence", f"{confidence:.1%}")
            else:
                st.error("❌ FAIL")
                st.metric("Confidence", f"{confidence:.1%}")

            # Progress bar
            st.progress(confidence)
        else:
            st.info("🔮 Adjust sliders and click Predict")

def show_batch_processor():
    st.header("📁 Batch Processor")

    st.markdown("""
    Upload a CSV file with the following columns:
    - `attendance_pct`
    - `homework_pct`
    - `midterm_score`
    - `study_hours_per_week`
    """)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)

            # Validate columns
            required_cols = ['attendance_pct', 'homework_pct', 'midterm_score', 'study_hours_per_week']
            if not all(col in batch_df.columns for col in required_cols):
                st.error(f"CSV must contain columns: {', '.join(required_cols)}")
                return

            st.subheader("Data Preview")
            st.dataframe(batch_df.head(), use_container_width=True)

            if st.button("🚀 Run Predictions"):
                # Make predictions
                predictions = model.predict(batch_df[required_cols])
                probabilities = model.predict_proba(batch_df[required_cols])

                # Add results to dataframe
                batch_df['prediction'] = ['Pass' if p == 1 else 'Fail' for p in predictions]
                batch_df['confidence'] = [prob[1] if p == 1 else prob[0] for p, prob in zip(predictions, probabilities)]

                st.subheader("Results Preview")
                st.dataframe(batch_df.head(), use_container_width=True)

                # Download button
                csv_buffer = io.StringIO()
                batch_df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()

                st.download_button(
                    label="💾 Download Results CSV",
                    data=csv_data,
                    file_name="batch_predictions.csv",
                    mime="text/csv"
                )

                # Summary statistics
                pass_count = (predictions == 1).sum()
                total_count = len(predictions)
                pass_rate = pass_count / total_count * 100

                st.subheader("Batch Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Predictions", total_count)
                with col2:
                    st.metric("Predicted Passes", pass_count)
                with col3:
                    st.metric("Pass Rate", f"{pass_rate:.1f}%")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
